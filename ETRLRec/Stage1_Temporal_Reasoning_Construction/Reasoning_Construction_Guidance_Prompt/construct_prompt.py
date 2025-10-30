#HF_ENDPOINT=https://hf-mirror.com python your_script.py
__all__ = ['construct_guidance_prompt']
from .import_data_files import import_interaction,import_item
import sys
import os
from collections import defaultdict
import csv
from tqdm import tqdm
import random
from itertools import islice



def _group_his_txt_by_user(file_path):
    user_history = defaultdict(list)
    with open(file_path, 'r', encoding='utf-8') as f:
        next(f)  
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                user_id, item_title = parts
                user_history[user_id].append(item_title)
    return user_history

def _load_info_txt_titles(file_path):
    titles = []
    with open(file_path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                titles.append(parts[2])
    return titles

def _load_template(template_path, template_type):
    """
    Load the specified type of template from template.txt
    """
    with open(template_path, 'r', encoding='utf-8') as f:
        templates = f.read().strip().split('\n')
    
    template_map = {
        "movie": templates[0],
        "video_game": templates[1], 
        "sport_and_out": templates[2]
    }
    
    template_str = template_map.get(template_type, templates[0])

    template_str = template_str.replace('\\n', '\n')

    return template_str

def _concatenate_three_dimensions():
    """
    Concatenate temporal knowledge of three dimensions
    """
    file = 'ETRLRec/Stage1_Temporal_Reasoning_Construction/Three_Dimensions_of_Temporal_Knowledge'
    with open(os.path.join(file, "Interaction_Sequentiality.txt"), 'r', encoding='utf-8') as f:
        Interaction_Sequentiality = f.read().strip()
    
    with open(os.path.join(file, "Item_Attribute_Transition.txt"), 'r', encoding='utf-8') as f:
        Item_Attribute_Transition = f.read().strip()
    
    with open(os.path.join(file, "User_Evolving_Thought.txt"), 'r', encoding='utf-8') as f:
        User_Evolving_Thought = f.read().strip()
    
    three_dimensions = f"\n\n{Interaction_Sequentiality}\n\n{Item_Attribute_Transition}\n\n{User_Evolving_Thought}"
    return three_dimensions

def construct_guidance_prompt(args): #写批量的形式，传入文件，以Sports & outdoors为例
    
    three_dimensions = _concatenate_three_dimensions()
    
    interaction_history_path_txt = args.interaction_history_out_path #import_interaction(args.user_data_path, args.item_data_path, args.interaction_history_out_path)
    item_path_txt = args.item_out_path #import_item(args.item_data_path, args.item_out_path)
    template = str(_load_template(args.template_path, args.template_type)) 

    interaction_history = _group_his_txt_by_user(interaction_history_path_txt)
    all_titles = _load_info_txt_titles(item_path_txt)

    with open(args.guidance_prompt_out_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['user_id', 'output_block', 'label']
        )

        for user_id, items in tqdm(islice(interaction_history.items(),3)):

            if len(items) < 2:
                continue

            history_str = "\n".join([f"- {item}" for item in items[:-1]])

            user_all_items = set(items)
            candidate_pool = [title for title in all_titles if title not in user_all_items]
            if len(candidate_pool) < args.candidate_num-1:
                continue
            selected_candidates = random.sample(candidate_pool, args.candidate_num-1)

            actual_next_interaction_item = items[-1]
            insert_pos = random.randint(0, args.candidate_num-1)
            selected_candidates.insert(insert_pos, actual_next_interaction_item)
            label = insert_pos + 1

            candidate_item_set = "\n".join([f"{i+1}. {item}" for i, item in enumerate(selected_candidates)])

            complete_prompt = template.format(
            interaction_history=history_str, 
            candidate_item_set=candidate_item_set,
            actual_next_interaction_item=actual_next_interaction_item,
            three_dimensions=three_dimensions,
            _label_ = label
            )
            
            writer.writerow([user_id, complete_prompt,label])
    guidance_prompt_out_path = args.guidance_prompt_out_path
    print(f"Successfully constructed guidance prompt! Saved in {guidance_prompt_out_path}")    
    return guidance_prompt_out_path

