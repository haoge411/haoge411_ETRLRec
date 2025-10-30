#from Reasoning_Construction_Guidance_Prompt.construct_prompt import construct_guidance_prompt
from ..Reasoning_Construction_Guidance_Prompt.construct_prompt import construct_guidance_prompt 

import pandas as pd
from openai import OpenAI
import concurrent.futures
import json
from utils import LLMClient

# def Large_Scale_LLM(prompt, args):
#     try:
#         client = OpenAI(api_key=args.LLM_key, base_url=args.LLM_url,)
#         system_prompt = "You are an expert in the field of sequential recommendation systems."
#         user_prompt = prompt
        
#         response = client.chat.completions.create(
#             model=args.LLM_version,
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt},
#             ],
#             stream=False
#         )
#         return response.choices[0].message.content
    
#     except Exception as e:
#         print(f"The API call failed: {str(e)}")
#         return f"Response generation failed: {str(e)}"

def _construct_single_temporal_reasoning(args,row):
    try:
        Large_Scale_LLM = LLMClient(api_key=args.LLM_key, base_url=args.LLM_url,model_version=args.LLM_version,temperature=args.temperature)

        print(f"Processing user {row['user_id']}")
        
        constructed_temporal_reasoning = Large_Scale_LLM(row['output_block'])
        

        conversation = [
            {"content": row['output_block'], "role": "user"},
            {"content": constructed_temporal_reasoning, "role": "assistant"}
        ]
        
        return {
            'user_id': row['user_id'],
            'conversation': json.dumps(conversation, ensure_ascii=False),
            'label': row['label']
        }
    
    except Exception as e:
        print(f"User {row['user_id']} failed: {str(e)}")
        conversation = [
            {"content": row['output_block'], "role": "user"},
            {"content": f"Response generation failed: {str(e)}", "role": "assistant"}
        ]
        return {
            'user_id': row['user_id'],
            'conversation': json.dumps(conversation, ensure_ascii=False),
            'label': row['label']
        }

def input_LLM(args):

    reasoning_construction_guidance_prompt_path = str(construct_guidance_prompt(args))

    df = pd.read_csv(reasoning_construction_guidance_prompt_path)
    results = []
    processed_count = 0

    '''
    Use a thread pool to accelerate processing
    '''
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.thread_num) as executor:

        futures = [executor.submit(_construct_single_temporal_reasoning, args, row) for _, row in df.iterrows()]
        

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                processed_count += 1
                

                if processed_count % 100 == 0:

                    temp_df = pd.DataFrame(results)
                    temp_df.to_csv(args.constructed_temporal_reasoning_path, index=False, encoding='utf-8')
                    print(f"Saved {processed_count} temporal reasonings in {args.constructed_temporal_reasoning_path}")
            
            except Exception as e:
                print(f"Failed: {str(e)}")
    

    All_temporal_reasoning = pd.DataFrame(results)
    All_temporal_reasoning.to_csv(args.constructed_temporal_reasoning_path, index=False, encoding='utf-8')
    print(f"Successfully processed! Saved in {args.constructed_temporal_reasoning_path}")
    constructed_temporal_reasoning_out_path = args.constructed_temporal_reasoning_path

    return str(constructed_temporal_reasoning_out_path)