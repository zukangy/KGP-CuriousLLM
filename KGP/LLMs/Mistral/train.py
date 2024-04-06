import os 
import time
import json 
import numpy as np 
from datetime import datetime
from tqdm import tqdm
import mlx.nn as nn
import mlx.core as mx
from mlx.utils import tree_flatten

from KGP.LLMs.Mistral.utils import iterate_batches, evaluate


def train(model, train_set, val_set, optimizer, loss, tokenizer, args):
    # Load meta data if resuming training
    metadata = {"train_loss": [], "val_loss": [], "epoch": 0}
    if args['checkpoint']['resume']:
        checkpint_dir = '/'.join(args['checkpoint']['resume_adapter_file'].split('/')[:-1])
        try:
            metadata = json.load(open(os.path.join(checkpint_dir, 'metadata.json')))
            print(f"Resuming training from epoch {metadata['epoch']}")
            print(f"train_loss: {metadata['train_loss'][-1]}") 
            try:
                print(f"val_loss: {metadata['val_loss'][-1]}")
            except IndexError:
                print("No val_loss found.")
        except FileNotFoundError:
            print(f"Metadata file not found in {checkpint_dir}. Save metadata from scratch.")
    
    # Create value and grad function for loss
    loss_value_and_grad = nn.value_and_grad(model, loss)

    losses = []
    n_tokens = 0

    print(f"Starting training from epoch {metadata['epoch']}")
    
    # Main training loop
    start = time.perf_counter()
    with tqdm(zip(
        range(args['model']['epochs']), iterate_batches(train_set, tokenizer, args['model']['batch_size'], train=True),
    ), total=args['model']['epochs'], miniters=1, mininterval=0.1) as pbar:
        for epoch, batch in pbar:
            if epoch < metadata['epoch']:
                continue
            # Forward and backward pass
            (lvalue, toks), grad = loss_value_and_grad(model, *batch)

            # Model update
            optimizer.update(model, grad)
            mx.eval(model.parameters(), optimizer.state, lvalue)

            # Record loss
            losses.append(lvalue.item())
            n_tokens += toks.item()

            # Report training loss if needed
            if (epoch + 1) % args['model']['steps_per_report'] == 0:
                train_loss = np.mean(losses)

                stop = time.perf_counter()
                step_sec = args['model']['steps_per_report'] / (stop - start)
                tokens_sec = float(n_tokens) / (stop - start)
                print( f"Step {epoch + 1}: Train loss {train_loss:.4f} | Step/sec {step_sec:.3f} | Tokens/sec {tokens_sec:.3f}")
                metadata['train_loss'].append(train_loss)
                losses = []
                n_tokens = 0
                start = time.perf_counter()
                
            # Report validation loss if needed
            if epoch != 0 and (epoch + 1) % args['model']['steps_per_eval'] == 0:
                print('Starting validation')
                stop = time.perf_counter()
                val_loss = evaluate(
                    model, val_set, loss, tokenizer, args['model']['batch_size']
                )
                print(f"Step {epoch + 1} | Val Loss {val_loss:.3f} | Val took {(time.perf_counter() - stop):.3f}s")

                start = time.perf_counter()
                
                metadata['val_loss'].append(val_loss)
                
            # Save adapter weights if needed
            if (epoch + 1) % args['model']['save_every'] == 0:
                cp_save_dir = os.path.join(args['root_dir'], args['models_dir'], f"ft_{args['model']['model_name']}_{args['id']}",
                                        f"cp_{args['model']['model_name']}")
                os.makedirs(cp_save_dir, exist_ok=True)
                mx.savez(
                    os.path.join(cp_save_dir, "adapters.npz"), 
                    **dict(tree_flatten(model.trainable_parameters()))
                )
                print(f"Iter {epoch + 1}: Saved adapter weights to {cp_save_dir}/adapters.npz.")
                
                metadata['epoch'] += args['model']['save_every']
                
                with open(os.path.join(cp_save_dir, 'metadata.json'), 'w') as f:
                    json.dump(metadata, f, sort_keys=False, indent=4)