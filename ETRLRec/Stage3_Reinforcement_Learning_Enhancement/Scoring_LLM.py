from ..utils import LLMClient

def _load_template(template_path, template_type):
    """
    Load the specified type of template from scoring_template.txt
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

def Scoring_LLM(args, interaction_history, candidate_item_set, actual_next_item, output_temporal_reasoning, prediction_next_item, SR_Logits):
    
    Scoring_llm = LLMClient(api_key=args.LLM_key, base_url=args.LLM_url,model_version=args.LLM_version,temperature=args.temperature)
    
    reasoning_score_prompt = str(_load_template(args.reasoning_score_template_path, args.template_type))
    
    input_text = reasoning_score_prompt.format(
        interaction_history = interaction_history,
        candidate_item_set = candidate_item_set,
        actual_next_item = actual_next_item,
        output_temporal_reasoning = output_temporal_reasoning,
        prediction_next_item = prediction_next_item,
        SR_Logits = SR_Logits,



    )
    response = Scoring_llm(input_text)
    
    return response.choices[0].message.content
    