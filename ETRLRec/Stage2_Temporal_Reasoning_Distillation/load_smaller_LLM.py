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

import json

import torch

import os

def _load_smaller_LLM(args):
    if args.use_int8:
        bnb_config = BitsAndBytesConfig(
        load_in_8bit=args.use_int8,  # Enable 8-bit loading
        llm_int8_threshold=args.smaller_llm_int8_threshold 
    )
    else:
        bnb_config = None

    base_smaller_llm = AutoModelForCausalLM.from_pretrained(
        args.smaller_llm_ori_path,  
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map=args.device,
        trust_remote_code=True, 
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.smaller_llm_ori_path,
        padding_side=args.padding_side, 
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if args.use_int8:
        base_smaller_llm = prepare_model_for_kbit_training(base_smaller_llm)


    if args.if_load_model:
        lora_path = args.saved_distilled_smaller_llm_path
        
        if not os.path.exists(args.saved_distilled_smaller_llm_path):
            print(f"Load smaller LLM failed")
            exit(1)
        
        # Check the file content
        print("LoRA Content:")
        print(os.listdir(lora_path))

        # Check the configuration file
        config_file = os.path.join(lora_path, "adapter_config.json")
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                adapter_config = json.load(f)
        else:
            print("Can not find adapter_config.json!")
        
        # Check the weight file
        weights_file = os.path.join(lora_path, "adapter_model.safetensors")
        if os.path.exists(weights_file):
            print(f"Find: {weights_file}, Size: {os.path.getsize(weights_file)/1024/1024:.2f} MB")
        else:
            weights_file = os.path.join(lora_path, "adapter_model.bin")
            if os.path.exists(weights_file):
                print(f"Find: {weights_file}, Size: {os.path.getsize(weights_file)/1024/1024:.2f} MB")
            else:
                print("Can not find weight file!")
                exit(1)
        
        smaller_llm = PeftModel.from_pretrained(base_smaller_llm, args.if_load_model, is_trainable=True)
        print("Loaded saved smaller-llm")
    else:
        if args.if_use_lora == False:
            
            peft_config = AdaLoraConfig( # or LoraConfig
                r=args.lora_rank,  
                lora_alpha=args.lora_alpha,  
                lora_dropout=args.lora_dropout, 
                bias="none",  
                task_type="CAUSAL_LM",  
                target_modules=args.target_modules
            )
            smaller_llm = get_peft_model(base_smaller_llm, peft_config)
            print("Used new lora")
        else:
            smaller_llm = base_smaller_llm
            print("Used without lora")
    
    if hasattr(smaller_llm, 'print_trainable_parameters'):
        smaller_llm.print_trainable_parameters()
    else:
        trainable_params = sum(p.numel() for p in smaller_llm.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in smaller_llm.parameters())
        print(f"Trainable: {trainable_params} | Proportion: {100 * trainable_params / total_params:.2f}%")

    return smaller_llm, tokenizer
