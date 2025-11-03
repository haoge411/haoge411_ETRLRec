from ..utils import parse_constructed_temporal_reasoning, ValidationCallback
from process_data import _split_dataset
from load_smaller_LLM import _load_smaller_LLM
import logging
from datasets import load_dataset
from bigmodelvis import Visualization
import re
import os
import sys
import json
import torch
from transformers import ( 
    AutoTokenizer,  
    AutoModelForCausalLM, 
    set_seed, 
    BitsAndBytesConfig,  
    TrainerCallback, 
    TrainingArguments, 
    Trainer,  
    DataCollatorForLanguageModeling,  
    pipeline,
)

from peft import ( 
    AdaLoraConfig,
    LoraConfig,  
    PeftModel,  
    get_peft_model, 
    prepare_model_for_kbit_training
)

from transformers import (
    AutoTokenizer,  
    AutoModelForCausalLM,  
    set_seed,  
    BitsAndBytesConfig,  
    TrainerCallback, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling, 
    pipeline,
)

set_seed(42)

global pod

pod = []


def _create_input_prompt(dataset):
    old_format_str = "\n\nAlthough you are asked to recommnend item, I still provide the actual next interaction item to you, which is: <answer>(.*?)</answer>. Please provide a sequential recommendation"
    new_format_str = ""

    pattern = re.compile(
        r'''\n\nAlthough you are asked to recommnend item,.*?a single recommendation sentence.''',
        flags=re.DOTALL
    )
    for message in dataset['conversation']:
        
        if message['role'] == 'user' and isinstance(message['content'], str):
            print("content", message['content'])
            message['content'] = pattern.sub("",message['content'])
            print(message['content'])
    
            
    return dataset


def _create_prompt_reference(train_dataset, test_dataset, eval_dataset, prompt_to_reference):
    
    for example in train_dataset:
        prompt = example['conversation'][0]['content']
        assistant_content = next((msg['content'] for msg in example['conversation'] if msg['role'] == 'assistant'), "")

        answer_end_index = assistant_content.find("</answer>") # After checking, see if there is any additional content
        if answer_end_index != -1:
            
            end_position = answer_end_index + len("</answer>") # Calculate the end position of the calculation (including the tag itself)
            
            after_content = assistant_content[end_position:].strip() # Check if there is any non-blank content after the tag

            if after_content:
                pod.append(example.get('user_id', 'No ID'))
                assistant_content = assistant_content[:end_position].rstrip() # Only retain until the end of the </answer> tag

        else:
            pod.append(example.get('user_id', 'No ID'))
        
        # Extract the content within the <answer> tag from assistant_content
        answer_match = re.search(r"<answer>(.*?)</answer>", assistant_content)
        answer = answer_match.group(1).strip() if answer_match else example['label']

        prompt_to_reference[prompt] = {
            "reasoning": assistant_content.split("<answer>")[0].strip() if "<answer>" in assistant_content else assistant_content,
            "answer": int(answer)
        }
        
        if answer is None or answer == "None":
            continue
    
    for example in test_dataset:
        prompt = example['conversation'][0]['content']
        assistant_content = next((msg['content'] for msg in example['conversation'] if msg['role'] == 'assistant'), "")

        answer_end_index = assistant_content.find("</answer>") # After checking, see if there is any additional content
        if answer_end_index != -1:
            
            end_position = answer_end_index + len("</answer>") # Calculate the end position of the calculation (including the tag itself)
            
            after_content = assistant_content[end_position:].strip() # Check if there is any non-blank content after the tag

            if after_content:
                pod.append(example.get('user_id', 'No ID'))
                assistant_content = assistant_content[:end_position].rstrip() # Only retain until the end of the </answer> tag

        else:
            pod.append(example.get('user_id', 'No ID'))
        
        # Extract the content within the <answer> tag from assistant_content
        answer_match = re.search(r"<answer>(.*?)</answer>", assistant_content)
        answer = answer_match.group(1).strip() if answer_match else example['label']

        prompt_to_reference[prompt] = {
            "reasoning": assistant_content.split("<answer>")[0].strip() if "<answer>" in assistant_content else assistant_content,
            "answer": int(answer)
        }
        
        if answer is None or answer == "None":
            continue

    for example in eval_dataset:
        prompt = example['conversation'][0]['content']
        assistant_content = next((msg['content'] for msg in example['conversation'] if msg['role'] == 'assistant'), "")

        answer_end_index = assistant_content.find("</answer>") # After checking, see if there is any additional content
        if answer_end_index != -1:
            
            end_position = answer_end_index + len("</answer>") # Calculate the end position of the calculation (including the tag itself)
            
            after_content = assistant_content[end_position:].strip() # Check if there is any non-blank content after the tag

            if after_content:
                pod.append(example.get('user_id', 'No ID'))
                assistant_content = assistant_content[:end_position].rstrip() # Only retain until the end of the </answer> tag

        else:
            pod.append(example.get('user_id', 'No ID'))
        
        # Extract the content within the <answer> tag from assistant_content
        answer_match = re.search(r"<answer>(.*?)</answer>", assistant_content)
        answer = answer_match.group(1).strip() if answer_match else example['label']

        prompt_to_reference[prompt] = {
            "reasoning": assistant_content.split("<answer>")[0].strip() if "<answer>" in assistant_content else assistant_content,
            "answer": int(answer)
        }
        
        if answer is None or answer == "None":
            continue
    
    test_input_prompts_for_ref = [example['conversation'][0]['content'] for example in test_dataset]
    eval_input_prompts_for_ref = [example['conversation'][0]['content'] for example in eval_dataset]     
    
    return prompt_to_reference, test_input_prompts_for_ref, eval_input_prompts_for_ref

def _create_distillation_dataset(args, constructed_temporal_reasoning_path):
    try:
        dis_dataset = load_dataset(
            'csv', 
            data_files=str(constructed_temporal_reasoning_path),
            split='train',
        )
        
        dis_dataset = dis_dataset.map(parse_constructed_temporal_reasoning)

        print(f"The size of temporal reasonings: {len(dis_dataset)}")
        train_dataset, test_dataset, eval_dataset = _split_dataset(dis_dataset)
        train_dataset = train_dataset.map(_create_input_prompt)
        test_dataset = test_dataset.map(_create_input_prompt)
        eval_dataset = eval_dataset.map(_create_input_prompt)
        print(train_dataset[0])

        prompt_to_reference = {}
        prompt_to_reference, test_input_prompts_for_ref, eval_input_prompts_for_ref  = _create_prompt_reference(train_dataset, test_dataset,eval_dataset, prompt_to_reference)
        print(f"Built {len(prompt_to_reference)} reference label")
        print(f"Built {len(test_input_prompts_for_ref)} input prompt for test")
        print(f"Built {len(eval_input_prompts_for_ref)} input prompt for evaluate")

        return train_dataset, test_dataset, eval_dataset, prompt_to_reference, test_input_prompts_for_ref, eval_input_prompts_for_ref
    except Exception as e:
        print(e)
        
        

def look_model(model):  
    Visualization(model).structure_graph() # Show the structure of LLM

def _evaluate_(args, smaller_llm, tokenizer, eval_dataset, eval_prompts, config, prompt_to_reference):
    """Evaluate the model performance on the validation set"""
    smaller_llm.eval()  
    total_correct = 0
    num_samples = min(len(eval_dataset), config["num_eval_samples"])  
    
    
    # Create a text generation pipeline
    text_generator = pipeline(
        "text-generation",
        model=smaller_llm,
        tokenizer=tokenizer,
        #device=model.device.index if torch.cuda.is_available() else -1,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        batch_size=config["eval_batch_size"],
        # Add generation control parameters
        early_stopping=True,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
        num_beams=3,
        length_penalty=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    for i in range(0, num_samples, config["eval_batch_size"]):
        
        batch_prompts = eval_prompts[i:i+config["eval_batch_size"]]
        
        full_prompts = [f"User: {prompt}\nAssistant:" for prompt in batch_prompts]
        
        # Generate text using pipeline
        with torch.no_grad():
            outputs = text_generator(
                full_prompts,
                max_new_tokens=args.output_token_max_length,
                do_sample=args.if_output_do_sample,  
                temperature=args.smaller_llm_temperature,  
                top_p=args.smaller_llm_top_p,
                return_full_text=False
            )
        
        # Extract the generated text
        completions = [output[0]['generated_text'] for output in outputs]
        
        # Extract the generated answer
        for j, completion in enumerate(completions):
            prompt_idx = i + j
            if prompt_idx < len(eval_prompts):
                prompt = eval_prompts[prompt_idx]
                
                # Get the reference answer from the mapping
                ref_data = prompt_to_reference.get(prompt, {"answer": ""})
                ref_answer = str(ref_data["answer"])
                ref_answer_reasoning = str(ref_data["reasoning"])
                
                # Extract the answer from the generated text
                answer_match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
               
                gen_answer = answer_match.group(1).strip() if answer_match else ""
                
                is_correct = gen_answer == ref_answer
                total_correct += 1 if is_correct else 0
                
                # Extract the reasoning part
                reason_match = re.search(r"<reason>(.*?)</reason>", completion, re.DOTALL)
                gen_reasoning = reason_match.group(1).strip() if reason_match else ""
                
                # Printing sample (Batch 1)
                if (i == 0 and j == 0) or (i == 1 and j == 1):
                    # Print the complete generated result
                    print(f"\nVerification sample #{prompt_idx + 1}:")
                    print(f"Prompt: {prompt}")
                    print(f"Full generation: {completion}")
                    print(f"Generation reasoning: {gen_reasoning}")
                    print(f"Generation answer: {gen_answer}")
                    print(f"Reference item answer: {ref_answer}")
                    print(f"Reference reasoning answer: {ref_answer_reasoning}")
                    print(f"Reference total answer: {str(ref_data)}")
                    print(f"Is correct: {'Yes' if is_correct else 'No'}")

    accuracy = total_correct / num_samples if num_samples > 0 else 0.0
    
    del text_generator
    torch.cuda.empty_cache()
    
    return accuracy

def distill_temporal_reasoning (args, constructed_temporal_reasoning_path):
 
    train_dataset, test_dataset, eval_dataset, prompt_to_reference, test_input_prompts_for_ref, eval_input_prompts_for_ref = _create_distillation_dataset(args, str(constructed_temporal_reasoning_path))

    smaller_llm, tokenizer = _load_smaller_LLM(args)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )


    def preprocess_function(example): 
        user_message = example['messages'][0]['content']

        reasoning = prompt_to_reference[user_message]["reasoning"] 
        answer = prompt_to_reference[user_message]["answer"] 
        formatted_output = f"<reason>{reasoning}</reason>\n <answer>{answer}</answer>"

        full_text = f"User: {user_message}\nAssistant: {formatted_output}"
        
        tokenized = tokenizer(
            full_text,
            max_length=args.prompt_max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_offsets_mapping=True,
            return_length=True,
        )    

        labels = tokenized["input_ids"].clone()
        
        assistant_prefix = "\nAssistant: "
        assistant_start = full_text.find(assistant_prefix)
        
        if assistant_start != -1:
            content_start = assistant_start + len(assistant_prefix)
            
            offsets = tokenized["offset_mapping"][0]
            
            start_index = None
            for i, (start, end) in enumerate(offsets):
                if start <= content_start < end:
                    start_index = i
                    break
            
            if start_index is not None:
                labels[0, :start_index] = -100
            
        return {
            "input_ids": tokenized["input_ids"][0],
            "attention_mask": tokenized["attention_mask"][0],
            "labels": labels[0]
        }
    
    #Loading smaller-scale LLM

    train_dataset = train_dataset.map(
        preprocess_function,
        batched = True,
        remove_columns=train_dataset.column_names
    )

    test_dataset = test_dataset.map(
        preprocess_function,
        batched = True,
        remove_columns=test_dataset.column_names
    )

    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched = True,
        remove_columns=eval_dataset.column_names
    )

    training_args = TrainingArguments(
    #ddp_backend="nccl",
    bf16_full_eval=True,
    output_dir=args.saved_distilled_smaller_llm_path,
    per_device_train_batch_size=args.per_device_distill_batch_size,
    per_device_eval_batch_size=args.per_device_eval_distill_batch_size,
    gradient_accumulation_steps=args.distill_gradient_accumulation_steps,
    learning_rate=args.distill_learning_rate,
    num_train_epochs=args.distill_num_epochs,
    weight_decay=args.distill_weight_decay,
    logging_dir=f"{args.saved_distilled_smaller_llm_path}/logs",
    gradient_checkpointing=args.distill_gradient_checkpointing,
    #gradient_checkpointing_kwargs={"use_reentrant": True},
    logging_steps=args.distill_logging_steps,
    save_strategy=args.distill_save_strategy,
    save_steps=args.distill_save_steps,
    save_total_limit=args.distill_save_total_limit,
    bf16=args.distill_if_use_bf16,
    ddp_find_unused_parameters=args.distill_ddp_find_unused_parameters,
    remove_unused_columns=args.distill_remove_unused_columns,
    report_to="none",
    optim=args.distill_optimizer,
    #warmup_ratio=0.08,
    group_by_length=True,
    eval_steps=args.distill_eval_steps,
    neftune_noise_alpha=args.distill_neftune_noise_alpha,
    include_tokens_per_second=args.distill_include_tokens_per_second,
    )

    offline_distillor = Trainer(
    model=smaller_llm,
    args=training_args,
    train_dataset=train_dataset.shuffle(seed=42),
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    #label_names=["labels"]
    )

    validation_callback = ValidationCallback(
        args=args,
        trainer=offline_distillor,
        model=smaller_llm,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,  
        eval_prompts=eval_input_prompts_for_ref,  
    )
    offline_distillor.add_callback(validation_callback)

    look_model(smaller_llm)
    logging.basicConfig(level=logging.INFO)
    smaller_llm.train()
    offline_distillor.train()

    os.makedirs(args.distilled_smaller_llm_path, exist_ok=True)
    offline_distillor.save_model(args.distilled_smaller_llm_path)
    tokenizer.save_pretrained(args.distilled_smaller_llm_path)

    print(f"The distilled smaller-scale LLM has been saved to {args.distilled_smaller_llm_path}")


    return args.distilled_smaller_llm_path, train_dataset, test_dataset, eval_dataset, prompt_to_reference, eval_input_prompts_for_ref
