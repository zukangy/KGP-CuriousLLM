import os  
import yaml 
import torch

from KGP.Traversal_agents.retriever import TF_IDF_retriever
from KGP.LLMs.Mistral.utils import load_lora_model
from KGP.Traversal_agents.KGP import get_response
from KGP.Traversal_agents.utils import (get_docs_from_kg, 
                                        Mistral_Inference, 
                                        load_graph)
from KGP.MDR.tokenizer import load_tokenizer
from KGP.KG.mdr_encoder import Retriever_inf
from KGP.LLMs.Mistral.quantize_mistral_mlx import load_config


if __name__ == '__main__':
    args = load_config('./configs/kgp.yml')
    
    # Get knowledge graph
    G = load_graph(args['KG'])
    
    # Get documents from the knowledge graph
    documents = get_docs_from_kg(G)

    # Retriever for identifying seeding documents
    retriever  = TF_IDF_retriever(args['init_retriever']['topk'], documents, 
                                  model_params=args['init_retriever']['params'])
    
    # Initialize Mistral Inference
    model_path = os.path.join(args['root_dir'], args['retriever']['model'])
    adapter_path = os.path.join(args['root_dir'], args['retriever']['adapter'])
    lora_rank = args['retriever']['model_params']['lora_rank']
    lora_layer = args['retriever']['model_params']['lora_layers']
    model, tokenizer = load_lora_model(model=model_path, adapter_file=adapter_path, 
                                       lora_rank=lora_rank, lora_layer=lora_layer)
    temp = args['retriever']['inference_params']['temp']
    max_token_len = args['retriever']['inference_params']['max_token_len']
    
    mistral_inference = Mistral_Inference(model, tokenizer, temp=temp, 
                                          max_token_len=max_token_len, 
                                          parse_template=True)
    
    # Load embedding model
    tokenizer, config = load_tokenizer(args['emb_model']['base_model'])
    emb = Retriever_inf(config, base_model=args['emb_model']['base_model'])
    model_checkpoint = torch.load(os.path.join(args['root_dir'], args['emb_model']['from_checkpoint']))
    emb.load_state_dict(model_checkpoint['model_state_dict'])
    emb_device = args['emb_model']['device']
    emb = emb.to(emb_device)
    emb.eval()
    
    #==============================================================
    prompt = "Anthony Avent played basketball fo a High School that is located in a city approcimately 8 mi west of where?"
    # mistral_inference = None 
    get_response(prompt, G, retriever, mistral_inference, emb, tokenizer, args)
