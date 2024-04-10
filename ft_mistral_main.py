import os  
import math
from pathlib import Path
import yaml
from datetime import datetime
import numpy as np
import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from KGP.LLMs.Mistral.mistral import mistral_loss
from KGP.LLMs.Mistral.quantize_mistral_mlx import load_config
from KGP.LLMs.Mistral.utils import load_dataset, evaluate, load_lora_model
from KGP.LLMs.Mistral.train import train


if __name__ == "__main__":
    args = load_config('./configs/mistral/ft_mistral.yml')
    
    # Create model folder if it doesn't exist
    os.makedirs(os.path.join(args['root_dir'], args['models_dir']), exist_ok=True)

    np.random.seed(args['model']['seed'])

    print("Loading pretrained model")
    
    resume_checkpoint = args['checkpoint']['resume']
    model, tokenizer = load_lora_model(model=args['model']['quantized_model_path'],
                                        adapter_file=args['checkpoint']['resume_adapter_file'] \
                                            if resume_checkpoint else None,
                                        lora_rank=args['model']['lora_rank'],
                                        lora_layer=args['model']['lora_layers'],
                                        verbose=True)

    print("Loading datasets")
    train_set, valid_set, test_set = load_dataset(args)
        
    if args['model']['train']:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args['id'] = timestamp
        save_model_path = os.path.join(args['root_dir'], args['models_dir'], f"ft_{args['model']['model_name']}_{args['id']}")
        os.makedirs(save_model_path, exist_ok=True)
        
        # Save the config file
        with open(os.path.join(save_model_path, 'config.yml'), 'w') as f:
            yaml.dump(args, f, sort_keys=False)
        
        print("Start training...")
        opt = optim.Adam(learning_rate=args['model']['learning_rate'])

        # Train model
        train(model, train_set, valid_set, opt, mistral_loss, tokenizer, args)

        # Save adapter weights
        mx.savez(os.path.join(save_model_path, 'ft_adapters.npz'), **dict(tree_flatten(model.trainable_parameters())))
        print(f"Training finished！ Saved adapter weights to {save_model_path}/ft_adapters.npz.")
        
    if args['model']['test']:
        # Load the LoRA adapter weights which we assume should exist by this point
        if not Path(args['test_adapter_file']).is_file():
            raise ValueError(
                f"Adapter file {args['test_adapter_file']} missing. "
                "Use --train to learn and save the adapters.npz."
            )
            
        model.load_weights(args['test_adapter_file'], strict=False)

        print("Testing")
        model.eval()
        test_loss = evaluate(
            model,
            test_set,
            mistral_loss,
            tokenizer,
            args['model']['batch_size'],
        )
        test_ppl = math.exp(test_loss)

        print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")
