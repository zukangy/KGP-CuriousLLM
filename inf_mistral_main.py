import os  
import json 
from tqdm import tqdm

from KGP.LLMs.Mistral.utils import load_lora_model, load, inference
from KGP.LLMs.Mistral.quantize_mistral_mlx import load_config


if __name__ == '__main__':
    args = load_config('./configs/mistral/ft_mistral.yml')
    data = json.load(open('DATA/Mistral/mistral/test.jsonl', 'r'))
    
    checkpoint = args['checkpoint']['resume_adapter_file']  
    if checkpoint:
        model, tokenizer = load_lora_model(
            model=args['model']['quantized_model_path'],
            adapter_file=checkpoint,
            lora_rank=args['model']['lora_rank'],
            lora_layer=args['model']['lora_layers'])
    else:
        # Load the quantized model without lora => raw model 
        model, tokenizer, _ = load(args['model']['quantized_model_path'])
        tokenizer.model_max_length = 2048
    
    limit = 10000
    limit = min(limit, len(data))
    
    resps = []
    for d in tqdm(data[:limit], total=limit):
        text = d['text']
        try:
            q, e, f = text.split('\n')
            prompt = q + ' ' + e
            
            temperature = args['model']['temperature']
            top_p = args['model']['top_p']
            pred = inference(model, prompt, tokenizer, temp=temperature, top_p=top_p,
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
    
        save_path = os.path.join(args['root_dir'], 'DATA/Mistral')
        os.makedirs(save_path, exist_ok=True)
        
        with open(os.path.join(save_path, 'test_responses.json'), 'w') as f:
            json.dump(resps, f, indent=4)