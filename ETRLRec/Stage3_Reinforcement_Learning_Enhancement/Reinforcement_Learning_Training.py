from copy import deepcopy
from tqdm import tqdm
import time
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model
import concurrent.futures
import re
from datetime import datetime
from .Conventional_SR_Model_SASRec.main import Trained_CRM
from .Scoring_LLM import Scoring_LLM 
from collections import defaultdict
from transformers import TrainerCallback
import torch
import gc
import unicodedata
from ..Stage2_Temporal_Reasoning_Distillation.load_smaller_LLM import _load_smaller_LLM
from ..Stage2_Temporal_Reasoning_Distillation.Offline_Knowledge_Distillation import look_model
from ..utils import GradientSafeMemoryCleanCallback, ValidationCallback
import logging

current_time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")  

from datasets import load_dataset  
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    set_seed, 
    BitsAndBytesConfig, 
    TrainerCallback, 
    TrainingArguments, 
    pipeline,
)
from peft import ( 
    LoraConfig,  
    PeftModel,  
    get_peft_model, 
    prepare_model_for_kbit_training, 
    PeftConfig,
)
from trl import GRPOTrainer, GRPOConfig
import torch
import os
import numpy as np
import gc
import json

set_seed(42)


def _load_distilled_smaller_scale_llm(args, distilled_smaller_scale_llm_path):

    initial_smaller_llm, tokenizer = _load_smaller_LLM(args)
    distilled_smaller_llm = PeftModel.from_pretrained(initial_smaller_llm, str(distilled_smaller_scale_llm_path), is_trainable=True)
    
    look_model(distilled_smaller_llm)
    if hasattr(distilled_smaller_llm, 'print_trainable_parameters'):
        distilled_smaller_llm.print_trainable_parameters()
    else:
        trainable_params = sum(p.numel() for p in distilled_smaller_llm.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in distilled_smaller_llm.parameters())
        print(f"Trainable: {trainable_params} | Proportion: {100 * trainable_params / total_params:.2f}%")

    return distilled_smaller_llm, tokenizer

def is_positive_integer(s):
    """Determine whether a string is a positive integer"""
    if s:
        return s.isdigit()
    else:
        return False
def __parse_prompt(prompt):

    HISTORY_PATTERN = re.compile(r"history is:\n(.*?)\n\nThe candidate", re.DOTALL)
    CANDIDATE_PATTERN = re.compile(r"\n(.*?)\n\nAlthough you", re.DOTALL)
    
    history_match = HISTORY_PATTERN.search(prompt)
    history = []
    if history_match:
        history_block = history_match.group(1).strip()
        history = [
            line[2:].strip()
            for line in history_block.split('\n') 
            if line[2:].strip()
        ]
    
    
    candidate_match = CANDIDATE_PATTERN.search(prompt)
    candidates = []
    if candidate_match:
        candidate_block = candidate_match.group(1).strip()
        for line in candidate_block.split('\n'):
            stripped_line = line.strip()
            if not stripped_line:  
                continue
            

            if stripped_line[0].isdigit():
                
                idx = 0
                while idx < len(stripped_line) and stripped_line[idx].isdigit():
                    idx += 1
                item_name = stripped_line[idx:].lstrip('. ').strip()
            elif stripped_line.startswith(('- ', '. ')):
                item_name = stripped_line[2:].strip()
            else:
                item_name = stripped_line
                
            candidates.append(item_name)
    return history, candidates
def __extract_recommendation(completion):
    """Extract the names of the prediction next item """

    ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    match = ANSWER_PATTERN.search(completion)
    return match.group(1).strip() if match else None
def __extract_reasoning(completion):
    """Extract the output temporal reasoning """
    REASON_PATTERN = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL)
    match = REASON_PATTERN.search(completion)
    return match.group(1).strip() if match else "None"
def __extract_ids(args, interaction_history):
    """Extract item IDs using global mapping"""
    item_name_to_ids = defaultdict(list)
    with open(args.item_data_path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:  
                item_id = int(parts[0].strip())  
                item_name = parts[2].strip()  
                item_name_to_ids[item_name].append(item_id)


    cleaned_list = [m.lstrip('- ').strip() for m in interaction_history]
    matched_ids = []
    for item in cleaned_list:
        if item in item_name_to_ids:
            matched_ids.extend(item_name_to_ids[item])
    return matched_ids
def __extract_rating_score(rating_response):
    """Extract the score from the scoring llm"""

    RATING_PATTERN = re.compile(r'<rating>(.*?)</rating>')
    match = RATING_PATTERN.search(rating_response)
    if match:
        try:
            rating = float(match.group(1).strip())
            return rating, rating_response
        except ValueError:
            return 0.0, rating_response
    return 0.0, rating_response
def _CRM_reward(args, interaction_history, candidate_item_set, actual_next_item, prediction_next_item):
    his_ids = __extract_ids(args, interaction_history)
    can_ids = __extract_ids(args, candidate_item_set)

    label_id = int(can_ids[int(actual_next_item)-1])
    
    rec_pos_in_can = -1
    r_CRM = -1.0  
    
    if is_positive_integer(str(prediction_next_item)):
        if int(prediction_next_item)-1 < len(can_ids) and int(prediction_next_item)-1 >= 0:
            rec_pos_in_can = prediction_next_item - 1
            SR_Logits, _, _, _, _ = Trained_CRM(
                llist=his_ids, 
                candi_sas=can_ids,
                label=label_id,state_dict_path=args.trained_CRM_path)
            r_CRM = float(SR_Logits[rec_pos_in_can])
        else:
            r_CRM = -0.5
    else:
        r_CRM = -1.0
    
    return r_CRM, SR_Logits
def _label_reward(prediction_next_item, actual_next_item):
    r_label = 1.0 if type(prediction_next_item) == int and prediction_next_item == actual_next_item else 0.0
    return r_label
def _reasoning_reward(args, output_temporal_reasoning, interaction_history, candidate_item_set, actual_next_item, prediction_next_item, SR_Logits):
    try:
        if output_temporal_reasoning != 'None':
            r_reasoning, _ = __extract_rating_score(
                Scoring_LLM(args, interaction_history, candidate_item_set, actual_next_item, output_temporal_reasoning, prediction_next_item, SR_Logits)
            )
        else:
            r_reasoning = -args.scaling_K

        return r_reasoning*1.0/args.scaling_K
        
    except Exception as e:
        print(f"Score extraction error: {e}")
        return float('nan')

def _curriculum_learning(args, current_step):
    if current_step < args.warm_up_stage:
        return 0.4, 0.4, 0.2
    else:
        return 0.2, 0.2, 0.6

def _process_single_sample(args, prompt, completion,prompt_to_references,current_step):
    
    # Get reference response  
    interaction_history, candidate_item_set = __parse_prompt(prompt)
    reference = prompt_to_references.get(prompt, {"reasoning": "", "answer": ""})
    constructed_temporal_reasoning = reference["reasoning"]
    actual_next_item = int(reference["answer"])

    prediction_next_item = __extract_recommendation(completion)
    if prediction_next_item is None:
        prediction_next_item = "None"
    else:   
        if is_positive_integer(prediction_next_item):
            prediction_next_item = int(prediction_next_item) 
        else:
            prediction_next_item = str(prediction_next_item)

    output_temporal_reasoning = str(__extract_reasoning(completion))
    
    
    # 1. R_CRM
    r_CRM, SR_Logits = _CRM_reward(args, interaction_history, candidate_item_set, actual_next_item, prediction_next_item)
    
    # 2. R_label
    r_label = _label_reward(prediction_next_item, actual_next_item)

    # 3. R_reasoning
    r_reasoning = _reasoning_reward(args, output_temporal_reasoning, interaction_history, candidate_item_set, actual_next_item, prediction_next_item, SR_Logits)
    
    alpha, beta, gamma = _curriculum_learning(args,current_step)
    
    sum_reward =  alpha * r_reasoning + beta * r_CRM + gamma * r_label
    
    debug_text = (
        f"Prompt: {prompt}\n"
        f"Completion: {completion}\n"
        f"Reasoning: {output_temporal_reasoning}\n"
        f"r_label: {r_label:.2f}, r_reasoning: {r_reasoning:.2f}, r_CRM: {r_CRM:.2f}, sum reward: {sum_reward:.2f}"
    )
    
    return sum_reward, debug_text

def _elaborate_reward_module(args, prompts, completions,  prompt_to_references):

    global global_step
    with torch.no_grad():
        rewards = []
        debug_info = []
        skipped_count = 0
        valid_samples = 0
        first_valid_debug = None
        first_valid_prompt = None
        first_valid_completion = None
        
        # Process samples in parallel using a thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=40) as executor:
            futures = []
            for prompt, completion in zip(prompts, completions):
                future = executor.submit(
                    _process_single_sample,
                    args,
                    prompt, 
                    completion,
                    prompt_to_references,
                    global_step,
                )
                futures.append(future)
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    total_reward, sample_debug = future.result()
                    rewards.append(total_reward)
                    debug_info.append(sample_debug)
                    if torch.isnan(torch.tensor(total_reward)):
                        skipped_count += 1
                except Exception as e:
                    print(f"Thread processing failed: {str(e)}")
                    rewards.append(float('nan'))
                    debug_info.append(f"Thread processing failed: {str(e)}")
                    skipped_count += 1
        
        total_samples = len(prompts)
        print(f"\nSample statistics: Total number={total_samples}, valid={valid_samples}, skip={skipped_count}")
        
        # Print the debugging information of the first valid sample
        if debug_info:
            print("\n==== Example of Reward Calculation ====")
            if first_valid_debug is not None:
                print(first_valid_debug)
            else:
                print("No valid samples, Skip all")
            print("====================\n")
        
        # Print sample example
        if total_samples > 0:
            print(f"Prompt example:\n{prompts[0]}")
            print(f"Completion example:\n{completions[0]}")
            
            # If there are valid samples, print the first valid sample as well
            if first_valid_prompt is not None:
                print(f"\nThe first valid sample prompt:\n{first_valid_prompt}")
                print(f"The first valid sample Completion:\n{first_valid_completion}")
        
        print(f'save path: {args.rf_smaller_llm_path}')
        global_step += 1
        print(f"Current training step number: {global_step}")
        
        
    return rewards



def prepare_rl_data(examples):
    
    prompts = []
    for messages in examples['messages']:
        user_message = next((msg['content'] for msg in messages if msg['role'] == 'user'), "")
        message = f"User: {user_message}\nAssistant: "
        prompts.append(message)
        
    return {"prompt": prompts}

def reinforcement_learning_enhancement(args, distilled_smaller_scale_llm_path, train_dataset, eval_dataset, prompt_to_reference, eval_input_prompts_for_ref):
    
    distilled_smaller_llm, tokenizer = _load_distilled_smaller_scale_llm(args, distilled_smaller_scale_llm_path)

    train_dataset = train_dataset.map(
        prepare_rl_data,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    text_dataset = text_dataset.map(
        prepare_rl_data,
        batched=True,
        remove_columns=text_dataset.column_names
    )
    eval_texts = eval_dataset.map(
        prepare_rl_data,
        batched=True,
        remove_columns=eval_dataset.column_names
    )

    generation_kwargs = {
        "max_tokens": args.output_token_max_length,      
        "temperature": args.smaller_llm_temperature_rl,
        "top_p": args.top_p_rl, 
        }

    grpo_config = GRPOConfig(
        learning_rate=args.rl_learning_rate,
        per_device_train_batch_size=args.rl_per_device_train_batch_size,
        gradient_accumulation_steps=args.rl_gradient_accumulation_steps,
        num_train_epochs=1,
        output_dir=args.rl_smaller_llm_out_path,
        logging_steps=args.rl_logging_steps,
        ds3_gather_for_generation=args.ds3_gather_for_generation,
        seed=42,
        gradient_checkpointing=args.gradient_checkpointing,
        num_generations=args.num_generations,
        beta=args.kl_weight,
        epsilon=args.cliprange,
        max_prompt_length=args.prompt_max_length,
        max_completion_length=args.output_token_max_length,
        remove_unused_columns=args.remove_unused_columns,
        #fp16=True,
        bf16 = args.use_bf16,
        optim=args.rl_optim,
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        weight_decay=args.rl_weight_decay,
        save_total_limit=args.save_total_limit,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        generation_kwargs=generation_kwargs,
        #dataloader_num_workers=0,
        group_by_length=args.group_by_length,
        logging_dir=f"{args.rl_smaller_llm_out_path}/logs",
        save_strategy=args.rl_save_strategy,
    )
    
    RL_trainer = GRPOTrainer(
        model=distilled_smaller_llm,
        reward_funcs=_elaborate_reward_module, 
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer, 
        reward_processing_classes=None,
    )

    validation_callback = ValidationCallback(
        args=args,
        trainer=RL_trainer,
        model=distilled_smaller_llm,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,  
        eval_prompts=eval_input_prompts_for_ref,  
    )

    clean_callback = GradientSafeMemoryCleanCallback(
        gradient_accumulation_steps=args)
    
    RL_trainer.add_callback(validation_callback)
    RL_trainer.add_callback(clean_callback)

    look_model(distilled_smaller_llm)
    logging.basicConfig(level=logging.INFO)
    distilled_smaller_llm.train()
    RL_trainer.train()

    os.makedirs(args.rl_smaller_llm_out_path, exist_ok=True)
    RL_trainer.save_model(args.rl_smaller_llm_out_path)
    tokenizer.save_pretrained(args.rl_smaller_llm_out_path)

    print(f"The RL trained smaller-scale LLM has been saved to {args.rl_smaller_llm_out_path}")

    
    return args.rl_smaller_llm_out_path