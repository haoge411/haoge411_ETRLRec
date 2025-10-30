from ..utils import parse_constructed_temporal_reasoning

from datasets import load_dataset
import re
import sys
import json
global pod

pod = []

def parse_constructed_temporal_reasoning(example):
    parsed = json.loads(example['conversation'])

    return {
        'conversation': parsed,
        'label': example['label']
    }

def _split_dataset(dis_dataset):
    try:
        total_size = len(dis_dataset)
        train_end = int(0.6 * total_size)  
        test_end = int(0.9 * total_size)
        
        all_indices = list(range(total_size))
        train_indices = all_indices[:train_end]
        test_indices = all_indices[train_end:test_end]
        eval_indices = all_indices[test_end:]

        train_dataset = dis_dataset.select(train_indices)
        test_dataset = dis_dataset.select(test_indices)
        eval_dataset = dis_dataset.select(eval_indices)
        return train_dataset, test_dataset, eval_dataset
    except Exception as e:
        print(f"Split dataset failed: {e}")
        return e

def _create_input_prompt(dataset):
    old_format_str = "\n\nAlthough you are asked to recommnend item, I still provide the actual next interaction item to you, which is: <answer>(.*?)</answer>. Please provide a sequential recommendation"
    new_format_str = "[new]"

    pattern = re.compile(
        r'''\n\nAlthough you are asked to recommnend item,.*?a single recommendation sentence.''',
        flags=re.DOTALL
    )
    for message in dataset['conversation']:
        print("i:  ",message)
        if message['role'] == 'user' and isinstance(message['content'], str):
            print("content", message['content'])
            message['content'] = pattern.sub("",message['content'])
            print(message['content'])
        print("i2: ",message)
    
            
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
        
        


        
        
        


    # except Exception as e:
    #     print(f"Create dataset failed: {e}")
    #     exit(1)


def distill_temporal_reasoning (args, constructed_temporal_reasoning_path):

    train_dataset, test_dataset, eval_dataset, prompt_to_reference, test_input_prompts_for_ref, eval_input_prompts_for_ref = _create_distillation_dataset(args, str(constructed_temporal_reasoning_path))

    #Loading smaller-scale LLM

    return

_create_distillation_dataset(0,'ETRLRec/Stage1_Temporal_Reasoning_Construction/Constructed_Temporal_Reasoning.csv')