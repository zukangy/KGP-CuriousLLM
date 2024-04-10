import os
import json
from tqdm import tqdm

from KGP.LLMs.Mistral.utils import load_lora_model, load, inference
from KGP.LLMs.Mistral.quantize_mistral_mlx import load_config

def process_data(model, tokenizer, data, temp, top_p, max_token_len, output_dir, model_name):
    responses = []

    for d in tqdm(data, desc=f"Temp: {temp}, Top P: {top_p}, Max Tokens: {max_token_len}", total=len(data)):
        text = d['text']
        try:
            question, evidence, follow_up = text.split('\n')
            prompt = f"{question} {evidence}"

            prediction = inference(model, prompt, tokenizer, temp=temp, top_p=top_p,
                                   max_token_len=max_token_len, parse_template=True)

            responses.append({
                'question': question,
                'evidence': evidence,
                'follow-up': follow_up,
                'response': prediction
            })
        except ValueError:  # Skip entries that don't have the correct format
            continue

    save_name = f"{model_name}_temp_{temp}_top_p_{top_p}_max_token_len_{max_token_len}.json"
    save_path = os.path.join(output_dir, save_name)
    with open(save_path, 'w') as f:
        json.dump(responses, f, indent=4)

if __name__ == '__main__':
    args = load_config('./configs/ft_mistral.yml')
    data = json.load(open('DATA/Mistral/mistral/test.jsonl', 'r'))
    data_dir = os.path.join(args['root_dir'], 'DATA/Mistral/test_followups')
    os.makedirs(data_dir, exist_ok=True)

    checkpoint = args['checkpoint']['resume_adapter_file']
    if checkpoint:
        model, tokenizer = load_lora_model(
            model=args['model']['quantized_model_path'],
            adapter_file=checkpoint,
            lora_rank=args['model']['lora_rank'],
            lora_layer=args['model']['lora_layers'])
    else:
        model, tokenizer, _ = load(args['model']['quantized_model_path'])
        tokenizer.model_max_length = 2048

    limit = min(100, len(data))
    data = data[:limit]

    model_name = "model_0_epoch"
    temps = [0.3, 0.6, 0.9]
    top_ps = [0.85, 0.9, 0.95, 1.0]
    max_token_lens = [50, 100]

    for temp in temps:
        for top_p in top_ps:
            for max_token_len in max_token_lens:
                process_data(model, tokenizer, data, temp, top_p, max_token_len, data_dir, model_name)
