import os  
import json 
import yaml 
from tqdm import tqdm
import random 

from KGP.Traversal_agents.KGP import get_supporting_evidence
from KGP.Traversal_agents.utils import (get_docs_from_kg,
                                        get_titled_docs_from_kg, 
                                        load_graph,
                                        get_seeding_retriever)
from KGP.Traversal_agents.agents import (Mistral_Agent, 
                                         TF_IDF_Agent, 
                                         BM25_Agent, 
                                         MDR_Agent,
                                         T5_Agent)
from KGP.LLMs.Mistral.quantize_mistral_mlx import load_config


if __name__ == '__main__':
    # Load config
    args = load_config('./configs/kgp.yml')
    
    # Index match the position of the question_dataset dataset
    bridge = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 22, 23, 24, 25, 26, 28]
    comparison = [13, 18, 21, 27, 31, 32, 33, 34, 43, 44, 46, 59, 62, 75, 76, 96, 117, 118, 124, 128, 162, 174, 178, 191, 197]
    
    # Load questions
    question_dataset = args['question_dataset']
    with open(question_dataset, 'r') as f:
        dataset = json.load(f)
        
    question_indices = bridge + comparison
    random.seed(args['seed'])
    random.shuffle(question_indices)
    questions = [(i, dataset[i]) for i in question_indices]
    
    # Get knowledge graph
    G = load_graph(args['KG'])
    
    # Get documents from the knowledge graph
    if args['titled']:
        documents = get_titled_docs_from_kg(G)
    else:
        documents = get_docs_from_kg(G)

    # Initialize the seed retriever
    retriever_name = args['init_retriever']['name']
    init_retriever = get_seeding_retriever(retriever_name)
    if retriever_name in ['gold', 'none']:
        init_retriever = init_retriever(args['init_retriever']['topk'], dataset)
    else:
        init_retriever = init_retriever(args['init_retriever']['topk'], documents, 
                            model_params=args['init_retriever']['params'])
    
    # Initialize the traversal agent
    if args['retriever']['name'] == 'mistral':
        traversal_agent = Mistral_Agent(args)
        
    elif args['retriever']['name'] == 'tfidf':
        traversal_agent = TF_IDF_Agent(args)
        
    elif args['retriever']['name'] == 'bm25':
        traversal_agent = BM25_Agent(args)
        
    elif args['retriever']['name'] == 'mdr':
        traversal_agent = MDR_Agent(args)
        
    elif args['retriever']['name'] == 't5':
        traversal_agent = T5_Agent(args)
        
    else:
        raise ValueError("Invalid retriever name")
    
    #==============================================================
    cp = [] # Checkpoint
    cp_questions = [] # Checkpoint questions
    if args['checkpoint']['resume']:
        print("Resuming from the checkpoint...")
        load_path = os.path.join(args['root_dir'], args['checkpoint']['checkpoint_path'])
        with open(load_path, 'r') as f:
            cp = json.load(f)
            cp_questions = [q['question'] for q in cp]

    save_path = os.path.join(args['root_dir'], args['checkpoint']['save_dir'], f"checkpoint_{args['checkpoint']['id']}")
    os.makedirs(save_path, exist_ok=True)

    save_config = False
    
    qp = [] # Questions and their evidence
    for i, q in tqdm(enumerate(questions), total=len(questions)):
        idx = q[0]
        q = q[1]
        prompt = q['question']
        
        if prompt in cp_questions:
            continue
        
        if retriever_name in ['gold', 'none']:
            evidence = init_retriever.retrieve(prompt)
        else:
            evidence = get_supporting_evidence(prompt, G, init_retriever, traversal_agent, args)
            
        if evidence:
            if idx in bridge:
                q_type = 'bridge'
            elif idx in comparison:
                q_type = 'comparison'
            else:
                q_type = 'unknown'
            out = {
                "type": q_type,
                "question": prompt,
                "found_evidence": evidence,
                "answer": q['answer'],
                "supports": q['supports'],
            }
            qp.append(out)
        else:
            print(f"No evidence found for the question: {prompt}")
        
        # Checkpoint  
        if i != 0 and (i+1) % args['checkpoint']['save_every'] == 0:
            with open(os.path.join(save_path, 'cp_evidence.json'), 'w') as f:
                json.dump(cp + qp, f, indent=4)
        
        if not save_config:     
            # Save the config file
            with open(os.path.join(save_path, 'config.yml'), 'w') as f:
                yaml.dump(args, f)
            
            save_config = True
        
    # Save the evidence
    with open(os.path.join(save_path, 'evidence.json'), 'w') as f:
        json.dump(cp + qp, f, indent=4)
