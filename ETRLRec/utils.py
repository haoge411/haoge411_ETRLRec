import openai
from openai import OpenAI
import json

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