import sys
from argparse import ArgumentParser
import torch
import os
from Stage1_Temporal_Reasoning_Construction.Construct_Temporal_Reasoning.input_LLM import input_LLM
from Stage2_Temporal_Reasoning_Distillation.Offline_Knowledge_Distillation import distill_temporal_reasoning, test_ETRLRec
from Stage3_Reinforcement_Learning_Enhancement.Reinforcement_Learning_Training import reinforcement_learning_enhancement

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

def main(args):
    
    # logger = TensorBoardLogger(save_dir='./log/', name=args.log_dir)
    # args.logger = logger
    # if not os.path.exists(args.second_model_path):
    #     os.makedirs(args.second_model_path)
    global test, t_prompt, prompt_to_reference
    test = []
    t_prompt = []
    prompt_to_reference = []

    if args.mode == 'train':
        constructed_temporal_reasoning_path = input_LLM(args)
        print("the constructed temporal reasonings are saved in: ", str(constructed_temporal_reasoning_path))
        distilled_smaller_scale_llm_path, train_dataset, test_dataset, eval_dataset, prompt_to_reference, eval_input_prompts_for_ref, test_input_prompts_for_ref = distill_temporal_reasoning(args,str(constructed_temporal_reasoning_path))
        rl_smaller_llm_out_path = reinforcement_learning_enhancement(args, distilled_smaller_scale_llm_path, train_dataset, eval_dataset, prompt_to_reference, eval_input_prompts_for_ref)
        print(f"The RL trained smaller-scale LLM has been saved to {rl_smaller_llm_out_path}")
        test = test_dataset
        t_prompt = test_input_prompts_for_ref
        sys.exit()
    else:
        ht1, ht10, nd10 = test_ETRLRec(args, args.rl_smaller_llm_out_path, test, t_prompt, prompt_to_reference)
        print(f"ht1: {ht1:.4f} ht10: {ht10:.4f} nd10: {nd10:.4f}")
        sys.exit()

if __name__ == '__main__':
    
    torch.multiprocessing.set_start_method('spawn')
    
    parser = ArgumentParser()
    parser.add_argument('--mode', default="train", choices=['train', 'test'], type=str)
    parser.add_argument('--template_type', default='video_game', choices=['movie', 'video_game', 'sport_and_out'], type=str, required=False)
    parser.add_argument('--user_data_path', default="mazon_reviews_Video_Games", type=str)
    parser.add_argument('--item_data_path', default="Amazon_meta_Video_Games", type=str)
    parser.add_argument('--interaction_history_out_path', default="ETRLRec/Stage1_Temporal_Reasoning_Construction/Reasoning_Construction_Guidance_Prompt/interaction_history.txt", type=str)
    parser.add_argument('--item_out_path', default="ETRLRec/Stage1_Temporal_Reasoning_Construction/Reasoning_Construction_Guidance_Prompt/item_info.txt", type=str)
    parser.add_argument('--template_path', default="ETRLRec/Stage1_Temporal_Reasoning_Construction/Reasoning_Construction_Guidance_Prompt/instruction_template.txt", type=str)
    parser.add_argument('--guidance_prompt_out_path', default="ETRLRec/Stage1_Temporal_Reasoning_Construction/Reasoning_Construction_Guidance_Prompt/construction_guidance_prompt.csv", type=str)
    parser.add_argument('--candidate_num', default=25, type=int)
    parser.add_argument('--thread_num', default=1, type=int)
    parser.add_argument('--temperature', default=1.3, type=float)
    parser.add_argument('--LLM_key', default="", type=str)
    parser.add_argument('--LLM_url', default="", type=str)
    parser.add_argument('--LLM_version', default="", type=str)
    parser.add_argument('--constructed_temporal_reasoning_path', default="ETRLRec/Stage1_Temporal_Reasoning_Construction/Constructed_Temporal_Reasoning.csv", type=str)
    
    
    parser.add_argument('--smaller_llm_int8_threshold', default=6.0, type=float)
    parser.add_argument('--candidate_num', default=25, type=int)
    parser.add_argument('--use_int8', default=True, type=bool)
    parser.add_argument('--device', default='auto', type=str)
    parser.add_argument('--smaller_llm_ori_path', default='your_smaller_llm_ori_path', type=str)
    parser.add_argument('--padding_side', default='left', type=str)
    parser.add_argument('--saved_distilled_smaller_llm_path', default='your_distilled_smaller_llm_path', type=str)
    parser.add_argument('--saved_best_distilled_smaller_llm_path', default="best_llm", type=str)
    parser.add_argument('--prompt_max_length', default=1300, type=int)
    parser.add_argument('--output_token_max_length', default=800, type=int)
    parser.add_argument('--if_output_do_sample', default=True, type=bool)
    parser.add_argument('--smaller_llm_temperature', default=0.5, type=float)
    parser.add_argument('--smaller_llm_top_p', default=0.8, type=float)
    parser.add_argument('--per_device_eval_distill_batch_size', default=10, type=int)
    parser.add_argument('--per_device_distill_batch_size', default=30, type=int)#args.
    parser.add_argument('--distill_gradient_accumulation_steps', default=30, type=int)
    parser.add_argument('--distill_learning_rate', default=1e-4, type=float)
    parser.add_argument('--distill_weight_decay', default=1e-2, type=float)
    parser.add_argument('--distill_num_epochs', default=100, type=int) #
    parser.add_argument('--distill_gradient_checkpointing', default=True, type=bool)
    parser.add_argument('--distill_logging_steps', default=1, type=int)
    parser.add_argument('--distill_save_strategy', default="no", type=str)
    parser.add_argument('--distill_save_steps', default=15, type=int)
    parser.add_argument('--distill_save_total_limit', default=6, type=int)
    parser.add_argument('--distill_if_use_bf16', default=True, type=bool)
    parser.add_argument('--distill_ddp_find_unused_parameters', default=True, type=bool)
    parser.add_argument('--distill_remove_unused_columns', default=False, type=bool)
    parser.add_argument('--distill_optimizer', default="adamw_torch_fused", type=str)
    parser.add_argument('--distill_eval_steps', default=60, type=int)
    parser.add_argument('--distill_neftune_noise_alpha', default=5.0, type=float)
    parser.add_argument('--distill_include_tokens_per_second', default=True, type=bool)
    parser.add_argument('--distilled_smaller_llm_path', default="./distilled_smaller_scale_llm_path", type=str)

    parser.add_argument('--rf_smaller_llm_path', default="./rl_smaller_scale_llm_path", type=str)
    parser.add_argument('--trained_CRM_path', default="", type=str)
    parser.add_argument('--CRM_batch_size', default=256, type=int)
    parser.add_argument('--CRM_device', default="cuda", type=str)
    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--beta2', default=0.98, type=float)
    parser.add_argument('--maxlen', default=30, type=int)
    parser.add_argument('--train_dir', default="", type=str)
    parser.add_argument('--hidden_units', default=120, type=int)
    parser.add_argument('--norm_first', action='store_true', default=True)
    parser.add_argument('--inference_only', default=True, type=str2bool)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_heads', default=2, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--dropout_rate', default=0.3, type=float)
    parser.add_argument('--num_epochs', default=1000, type=int)
    parser.add_argument('--temperature', default=0.3, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--scaling_K', default=1000, type=int)
    parser.add_argument('--reasoning_score_template_path', default="/hy-tmp/ETRLRec/Stage3_Reinforcement_Learning_Enhancement/scoring_template.txt", type=str)
    parser.add_argument('--warm_up_stage', default=50, type=int)
    parser.add_argument('--smaller_llm_temperature_rl', default=0.7, type=float)
    parser.add_argument('--top_p_rl', default=0.85, type=float)
    parser.add_argument('--rl_learning_rate', default=5e-6, type=float)
    parser.add_argument('--rl_weight_decay', default=1e-4, type=float)
    parser.add_argument('--rl_per_device_train_batch_size', default=30, type=int)
    parser.add_argument('--rl_gradient_accumulation_steps', default=30, type=int)
    parser.add_argument('--rl_smaller_llm_out_path', default="", type=str)
    parser.add_argument('--rl_logging_steps', default=1, type=int)
    parser.add_argument('--num_generations', default=32, type=int)
    parser.add_argument('--kl_weight', default=4e-2, type=float)
    parser.add_argument('--cliprange', default=0.2, type=float)
    parser.add_argument('--ds3_gather_for_generation', default=False, type=bool)
    parser.add_argument('--gradient_checkpointing', default=True, type=bool)
    parser.add_argument('--remove_unused_columns', default=False, type=bool)
    parser.add_argument('--use_bf16', default=True, type=bool)
    parser.add_argument('--rl_optim', default="adamw_torch_fused", type=str)
    parser.add_argument('--use_vllm', default=True, type=bool)
    parser.add_argument('--vllm_mode', default="colocate", type=str)
    parser.add_argument('--vllm_gpu_memory_utilization', default=0.6, type=float)
    parser.add_argument('--vllm_tensor_parallel_size', default=2, type=int)
    parser.add_argument('--save_total_limit', default=10, type=int)
    parser.add_argument('--ddp_find_unused_parameters', default=False, type=bool)
    parser.add_argument('--group_by_length', default=True, type=bool)
    parser.add_argument('--rl_save_strategy', default="steps", type=str)
    parser.add_argument('--per_device_test_size', default=1000, type=int)
    
    args = parser.parse_args()
    main(args)