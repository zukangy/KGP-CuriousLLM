import os 
import yaml 
from mlx_lm.utils import convert


def load_config(config_path):
    return yaml.safe_load(open(config_path, "r"))


def convert_(args):
    model = args['model_name']
    root_dir = args['root_dir']
    save_path = args['save_path']
    
    if args['quantize']:
        quantize = args['quantize']
        q_group_size = args['q_group_size']
        q_bits = args['q_bits']
        dtype = args['dtype']
        
        save_dir = os.path.join(root_dir, save_path)
        save_path = os.path.join(save_dir, f'mistral_quantized_{q_bits}_bit')
        os.makedirs(save_dir, exist_ok=True)
        
        convert(model, save_path, quantize, q_group_size, q_bits, dtype)
    else:
        print('No quantization chosen...')


if __name__ == "__main__":
    args = load_config('./configs/quantized_mistral_config.yml')
    
    convert_(args)
    

