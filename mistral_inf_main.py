import os  
import json 
from tqdm import tqdm

from KGP.LLMs.Mistral.utils import load_lora_model, inference
from KGP.LLMs.Mistral.quantize_mistral_mlx import load_config


if __name__ == '__main__':
    args = load_config('./configs/ft_mistral.yml')
    data = json.load(open('DATA/Mistral/mistral/test.jsonl', 'r'))
    
    model, tokenizer = load_lora_model(
        model=args['model']['quantized_model_path'],
        adapter_file=args['checkpoint']['resume_adapter_file'],
        lora_rank=args['model']['lora_rank'],
        lora_layer=args['model']['lora_layers'])
    
    limit = 100
    limit = min(limit, len(data))
    
    resps = []
    for d in tqdm(data[:limit], total=limit):
        text = d['text']
        try:
            q, e, f = text.split('\n')
            prompt = q + ' ' + e
            pred = inference(model, prompt, tokenizer, temp=args['model']['temperature'], 
                    max_token_len=args['model']['max_token_len'], parse_template=True)
            
            resps.append(
                {
                    'question': q,
                    'evidence': e,
                    'follow-up': f,
                    'response': pred
                }
            )
        except:
            continue 
    
    with open(os.path.join(args['root_dir'], 'DATA/Mistral/test_responses.jsonl'), 'w') as f:
        json.dump(resps, f, indent=4)