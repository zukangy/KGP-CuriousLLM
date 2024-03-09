# Quantize the Mistral model using the quantization config file

from KGP.LLMs.Mistral.quantize_mistral_mlx import load_config, convert_


if __name__ == "__main__":
    args = load_config('./configs/quantize_mistral.yml')
    
    convert_(args)