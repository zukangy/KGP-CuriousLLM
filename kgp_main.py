import os  
import json 
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from KGP.Traversal_agents.retriever import TF_IDF_retriever
from KGP.LLMs.Mistral.utils import load_lora_model
from KGP.Traversal_agents.KGP import get_supporting_evidence
from KGP.Traversal_agents.utils import (get_docs_from_kg, 
                                        Mistral_Inference, 
                                        load_graph)
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
    
    mistral_inference = Mistral_Inference(model, 
                                          tokenizer, 
                                          temp=temp, 
                                          max_token_len=max_token_len, 
                                          parse_template=True)
    
    # Load sentence transformer
    emb = SentenceTransformer(args['emb_model']['model'])
    
    #==============================================================
    question_dataset = "DATA/HotpotQA/MDR/test_docs.json"
    with open(question_dataset, 'r') as f:
        questions = json.load(f)

    cp = []
    if args['checkpoint']['resume']:
        print("Resuming from the checkpoint...")
        load_path = os.path.join(args['root_dir'], args['checkpoint']['checkpoint_path'])
        cp = json.load(open(load_path, 'r'))
    cp_questions = [q['question'] for q in cp]

    qp = []
    for i, q in tqdm(enumerate(questions), total=len(questions)):
        prompt = q['question']
        
        if prompt in cp_questions:
            continue
        
        evidence = get_supporting_evidence(prompt, G, retriever, mistral_inference, emb, args)
        if evidence:
            out = {
                "question": prompt,
                "found_evidence": evidence,
                "answer": q['answer'],
                "supports": q['supports'],
            }
            qp.append(out)
            
            if i != 0 and (i+1) % args['checkpoint']['save_every'] == 0:
                os.makedirs(os.path.join(args['root_dir'], args['checkpoint']['save_dir']), exist_ok=True)
                with open(os.path.join(args['root_dir'], args['checkpoint']['save_dir'], 'cp_evidence.json'), 'w') as f:
                    json.dump(qp, f, indent=4)
        else:
            print(f"No evidence found for the question: {prompt}")
        
    # Save the evidence
    os.makedirs(os.path.join(args['root_dir'], 'DATA/KG/evidence'), exist_ok=True)
    with open(os.path.join(args['root_dir'], 'DATA/KG/evidence', 'evidence.json'), 'w') as f:
        json.dump(qp, f, indent=4)
