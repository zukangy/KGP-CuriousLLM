# This script is used to generate the embeddings for the documents in test_doc.json using the MDR model.

import os 
import torch 
import json 

from KGP.MDR.tokenizer import load_tokenizer
from KGP.KG.mdr_encoder import Retriever_inf
from KGP.KG.train import run 
from KGP.LLMs.Mistral.quantize_mistral_mlx import load_config



if __name__ == "__main__":
    args = load_config('./configs/mdr_embedding/mdr_2wiki_embedding.yml')
    
    raw_documents_data = json.load(open(os.path.join(args['root_dir'], args['dataset']), 'r'))
    
    tokenizer, config = load_tokenizer(model_name=args['model']['base_model'])
    
    model = Retriever_inf(config, base_model=args['model']['base_model'])
    
    model_checkpoint = torch.load(os.path.join(args['root_dir'], args['model']['from_checkpoint']))
    print("Loading model from checkpoint...")
    model.load_state_dict(model_checkpoint['model_state_dict'])
    print("Model loaded successfully...")
        
    run(raw_documents_data, model, tokenizer, args)