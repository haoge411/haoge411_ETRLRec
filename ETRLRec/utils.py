import openai
from openai import OpenAI
import json
import os
import torch
from datetime import datetime
import sys
import re
import gc 


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



class LLMClient:
    def __init__(self, api_key, base_url, model_version, temperature=0.7):

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_version = model_version
        self.system_prompt = "You are an expert in the field of sequential recommendation systems."
        self.temperature = temperature
    
    def generate_response(self, prompt, temperature=None):
        try:
            print('start input!')
            temp = temperature if temperature is not None else self.temperature
            print("prompt: ",prompt,'\n')
            response = self.client.chat.completions.create(
                model=self.model_version,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temp,
                stream=False
            )
            
            print("result: ",response.choices[0].message.content)
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"The API call failed: {str(e)}")
            return f"Response generation failed: {str(e)}"
    
    def __call__(self, prompt, temperature=None):
        return self.generate_response(prompt, temperature)


def parse_constructed_temporal_reasoning(example):
    parsed = json.loads(example['messages'])

    return {
        'messages': parsed,
        'label': example['label']
    }

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

    accuracy = total_correct / num_samples if num_samples > 0 else 0.0 # To save computational efficiency, we use HR@1 for verification
    
    del text_generator
    torch.cuda.empty_cache()
    
    return accuracy



class ValidationCallback(TrainerCallback):
    def __init__(self, args, trainer,model, tokenizer, eval_dataset, eval_prompts, config):
        self.args =args
        self.model = model
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.eval_prompts = eval_prompts
        self.config = config
        self.best_accuracy = -1.0 
        self.best_model_path = os.path.join(self.args.saved_best_distilled_smaller_llm_path, "best_model")
        self.eval_counter = 0
        #self.last_eval_step = -1

    def on_step_end(self, state):

        if isinstance(self.config["eval_frequency"], int):
            if state.global_step % self.config["eval_frequency"] == 0:
                self.evaluate_and_save(state)

    # def on_epoch_end(self, args, state, control, **kwargs):
    #     """Evaluate at the end of each epoch"""
    #     self.evaluate_and_save(state)

    def evaluate_and_save(self, state):
        self.eval_counter += 1
        print(f"\n{'=' * 50}")
        print(f"Start verification and assessment {self.eval_counter} (Steps {state.global_step})")

        self.trainer.save_model(os.path.join(self.args.saved_distilled_smaller_llm_path,str(state.global_step)))

        try:

            accuracy = _evaluate_(
                self.args,
                self.model,
                self.tokenizer,
                self.eval_dataset,
                self.eval_prompts,
                self.config
            )

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                print(f"Find better model: {accuracy:.4f}")

                if self.args.use_lora:
                    os.makedirs(self.best_model_path, exist_ok=True)

                    self.trainer.save_model(self.best_model_path)

                    self.tokenizer.save_pretrained(self.best_model_path)
                    print(f"The best LoRA adapter has been saved to {self.best_model_path}")
                else:
                    self.model.save_pretrained(self.best_model_path)
                    self.tokenizer.save_pretrained(self.best_model_path)
                    print(f"The best model has been saved to {self.best_model_path}")

                with open(os.path.join(self.config["output_dir"], "best_metrics.txt"), "w") as f:
                    f.write(f"Time of evaluating: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            else:
                print(f"Current model accuracy: {accuracy:.4f} (Best: {self.best_accuracy:.4f})")
            

            with open(os.path.join(self.config["output_dir"], "eval_results.txt"), "a") as f:
                f.write(f"Assess {self.eval_counter} | Steps {state.global_step} | Accuracy {accuracy:.4f}\n")

        except Exception as e:
            print(f"Verification and evaluation failed: {e}")
            import traceback
            traceback.print_exc()

        print(f"{'=' * 50}\n")

        torch.cuda.empty_cache()
        gc.collect()
        self.model.train()

