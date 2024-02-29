from KGP.Traversal_agents.mistral_7b.quantize_mistral_mlx import load_config, convert_


if __name__ == "__main__":
    args = load_config('./configs/quantized_mistral_config.yml')
    
    convert_(args)