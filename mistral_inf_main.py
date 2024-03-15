import os  
import json 

from KGP.LLMs.Mistral.utils import load_lora_model, generate
from KGP.LLMs.Mistral.quantize_mistral_mlx import load_config


if __name__ == '__main__':
    args = load_config('./configs/ft_mistral.yml')
    data = json.load(open('DATA/T5_traversal_agent/mistral/test.jsonl', 'r'))
    
    model, tokenizer = load_lora_model(
        model=args['model']['quantized_model_path'],
        adapter_file=args['checkpoint']['resume_adapter_file'],
        lora_rank=args['model']['lora_rank'],
        lora_layer=args['model']['lora_layers'])
    
    # prompt = "Question: What role could Christian Daniel Claus have had during the Revolution? \n Evidence: Christian Daniel Claus (1727\u20131787) was a Commissioner of Indian Affairs and a prominent Loyalist during the American Revolution. \n Follow-up Question: "
    limit = 10
    for i, d in enumerate(data[:limit]):
        text = d['text']
        try:
            q, e, f = text.split('\n')
            prompt = q + ' ' + e
            print(f"Prompt {i}:" )
            print(f"Question: {q}" )
            print(f"Evidence: {e}" )
            print(f"Follow-up Question {i}: {f}" )
            pred = generate(model, prompt, tokenizer, temp=args['model']['temperature'], 
                    max_token_len=args['model']['max_token_len'], parse_template=True)
            print("=========>")
            print(f"Prediction {i}: {pred}" )
            print("=========>")
        except:
            continue