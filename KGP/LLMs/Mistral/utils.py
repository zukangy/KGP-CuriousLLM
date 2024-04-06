# Copyright © 2023 Apple Inc.
# Taken and modified from mlx_example/lora/*

import glob
import json
import logging
from pathlib import Path
import numpy as np
from typing import Generator
from tqdm import tqdm

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import transformers
from huggingface_hub import snapshot_download

import KGP.LLMs.Mistral.mistral as mistral
from KGP.LLMs.Mistral.lora import LoRALinear
from KGP.LLMs.Mistral.dataset import Dataset


# Constants
MODEL_MAPPING = {
    "llama": mistral,
    "mistral": mistral,  # mistral is compatible with llama
}


def _get_classes(config: dict):
    """
    Retrieve the model and model args classes based on the configuration.

    Args:
        config (dict): The model configuration.

    Returns:
        A tuple containing the Model class and the ModelArgs class.
    """
    model_type = config["model_type"]
    if model_type not in MODEL_MAPPING:
        msg = f"Model type {model_type} not supported."
        logging.error(msg)
        raise ValueError(msg)

    arch = MODEL_MAPPING[model_type]
    return arch.Model, arch.ModelArgs


def load(path_or_hf_repo: str):
    # If the path exists, it will try to load model form it
    # otherwise download and cache from the hf_repo and cache
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        model_path = Path(
            snapshot_download(
                repo_id=path_or_hf_repo,
                allow_patterns=["*.json", "*.safetensors", "tokenizer.model"],
            )
        )

    with open(model_path / "config.json", "r") as f:
        config = json.loads(f.read())
        quantization = config.get("quantization", None)

    weight_files = glob.glob(str(model_path / "*.safetensors"))
    if len(weight_files) == 0:
        raise FileNotFoundError("No safetensors found in {}".format(model_path))

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf).items())

    model_class, model_args_class = _get_classes(config=config)
    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)
    if quantization is not None:
        nn.QuantizedLinear.quantize_module(
            model,
            **quantization,
            linear_class_predicate=lambda m: isinstance(m, nn.Linear)
            and m.weight.shape[0] != 8,
        )

    model.load_weights(list(weights.items()))

    mx.eval(model.parameters())
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer, config


def load_lora_model(model: str, adapter_file: str = None, lora_rank: int = 8, 
                    lora_layer: int = 16, verbose: bool = False):
    model, tokenizer, _ = load(model)
    
    tokenizer.model_max_length = 2048
    
    # Freeze all layers other than LORA linears
    model.freeze()
    for l in model.model.layers[len(model.model.layers) - lora_layer :]:
        l.self_attn.q_proj = LoRALinear.from_linear(l.self_attn.q_proj, rank=lora_rank)
        l.self_attn.v_proj = LoRALinear.from_linear(l.self_attn.v_proj, rank=lora_rank)
        if hasattr(l, "block_sparse_moe"):
            l.block_sparse_moe.gate = LoRALinear.from_linear(l.block_sparse_moe.gate, rank=lora_rank)
    
    if verbose:
        p = sum(v.size for _, v in tree_flatten(model.parameters())) / 10**6
        print(f"Total parameters {p:.3f}M")
        p = sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 10**6
        print(f"Trainable parameters {p:.3f}M")
    
    if adapter_file is not None:
        print(f"Loading pretrained adapters from {adapter_file}")
        model.load_weights(adapter_file, strict=False)
        print("Weights loaded successfully！")   
        
    return model, tokenizer


def generate1(
    prompt: mx.array, model: nn.Module, temp: float = 0.0
) -> Generator[mx.array, None, None]:
    """
    Generate text based on the given prompt and model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        temp (float): The temperature for sampling. If temp is 0, use max sampling.

    Yields:
        mx.array: The generated text.
    """

    def sample(logits: mx.array) -> mx.array:
        return (
            mx.argmax(logits, axis=-1)
            if temp == 0
            else mx.random.categorical(logits * (1 / temp))
        )

    y = prompt
    cache = None
    while True:
        logits, cache = model(y[None], cache=cache)
        logits = logits[:, -1, :]
        y = sample(logits)
        yield y
        
        
def generate2(
    prompt: mx.array, model: nn.Module, temp: float = 0.0,
    top_p: float = 1.0
) -> Generator[mx.array, None, None]:
    """
    Generate text based on the given prompt and model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        temp (float): The temperature for sampling. If temp is 0, use max sampling.
        top_p (float): The nucleus sampling parameter.
    Yields:
        mx.array: The generated text.
    """

    def sample(logits: mx.array) -> mx.array:
        softmax_logits = mx.softmax(logits)
        
        if temp == 0:
            token = mx.argmax(softmax_logits, axis=-1)
        else: 
            if top_p > 0 and top_p < 1.0:
                if (
                    logits.dtype == mx.bfloat16
                ):  # workdaround for unable to load kernel contiguous_scan_inclusive_sum_bfloat16_bfloat16
                    logits = logits.astype(mx.float32)
                probs = mx.softmax(logits / temp, axis=-1)

                sorted_probs = mx.sort(probs)[::-1]
                sorted_indices = mx.argsort(probs)[::-1]
                cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

                top_probs = mx.where(
                    cumulative_probs > 1 - top_p,
                    sorted_probs,
                    mx.zeros_like(sorted_probs),
                )
                sorted_token = mx.random.categorical(mx.log(top_probs))
                token = sorted_indices.squeeze(0)[sorted_token]
            else:
                token = mx.random.categorical(logits * (1 / temp))
            
            return token

    y = prompt
    cache = None
    while True:
        logits, cache = model(y[None], cache=cache)
        logits = logits[:, -1, :]
        y = sample(logits)
        yield y
        
        
def inference(model, prompt, tokenizer, temp: float = 0.3, top_p: float = 1.0, max_token_len: int = 100, 
              parse_template: bool = True, verbose: bool = 0):
    
    if parse_template:
        instruction = """You are a critical thinker and like to ask questions. 
        Please provide only the follow-up question without additional information.
        Please think carefully and provide a follow-up question that is relevant to the previous conversation."""
        prompt = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False)
    
    # print(prompt, end="", flush=True)
    prompt = mx.array(tokenizer.encode(prompt))

    tokens = []
    skip = 0
    pred_token_seq = generate2(prompt, model, temp=temp, top_p=top_p)
    for token, n in zip(pred_token_seq, range(max_token_len)):
        if token == tokenizer.eos_token_id:
            break

        tokens.append(token.item())
        s = tokenizer.decode(tokens)
        if len(s) - skip > 1:
            if verbose:
                print(s[skip:-1], end="", flush=True)
            skip = len(s) - 1
            
    if verbose:
        print(tokenizer.decode(tokens)[skip:], flush=True)
        print("=" * 10)
        
    if len(tokens) == 0:
        print("No tokens generated for this prompt")
        return ""
    return s
    

def load_dataset(args):
    def load_and_check(name):
        dataset_path = Path(args['dataset']) / f"{name}.jsonl"
        try:
            return Dataset(dataset_path)
        except Exception as e:
            print(f"Unable to build dataset {dataset_path} ({e})")
            raise

    names = ("train", "valid", "test")
    train, valid, test = (load_and_check(n) for n in names)

    if args['model']['train'] and len(train) == 0:
        raise ValueError(
            "Training set not found or empty. Must provide training set for fine-tuning."
        )
    if args['model']['train'] and len(valid) == 0:
        raise ValueError(
            "Validation set not found or empty. Must provide validation set for fine-tuning."
        )
    if args['model']['test'] and len(test) == 0:
        raise ValueError(
            "Test set not found or empty. Must provide test set for evaluation."
        )
    return train, valid, test


def iterate_batches(dset, tokenizer, batch_size, train=False):
    # Shuffle indices
    while True:
        indices = np.arange(len(dset))
        if train:
            indices = np.random.permutation(indices)

        # Collect batches from dataset
        for i in range(0, len(indices) - batch_size + 1, batch_size):
            # Encode batch
            batch = []
            for j in range(batch_size):
                prompt = dset[indices[i + j]]
                messages = [
                {"role": "user", "content": "You are a critical thinker and like to ask questions."},
                {"role": "assistant", "content": prompt},
                ]
                text = tokenizer.apply_chat_template(messages, tokenize=False)
                batch.append(tokenizer.encode(text))
            lengths = [len(x) for x in batch]

            # Pad to the max length
            batch_arr = np.zeros((batch_size, max(lengths)), np.int32)

            for j in range(batch_size):
                batch_arr[j, : lengths[j]] = batch[j]
            batch = mx.array(batch_arr)
            yield batch[:, :-1], batch[:, 1:], mx.array(lengths)

        if not train:
            break
        
        
def evaluate(model, dataset, loss, tokenizer, batch_size):
    all_losses = []
    ntokens = 0
    for batch in tqdm(iterate_batches(dataset, tokenizer, batch_size)):
        losses, toks = loss(model, *batch)
        all_losses.append((losses * toks).item())
        ntokens += toks.item()

    return np.sum(all_losses) / ntokens
