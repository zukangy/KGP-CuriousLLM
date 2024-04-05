import os  
import json 
import yaml 
from tqdm import tqdm
import random 

from KGP.Traversal_agents.KGP import get_supporting_evidence
from KGP.Traversal_agents.utils import load_graph, generate_evidence_record, parse_evidence_string
from KGP.Traversal_agents.seed_retriever import get_seeding_retriever
from KGP.Traversal_agents.agents import get_traversal_agent


if __name__ == '__main__':
    # Load config
    args = yaml.safe_load(open('./configs/kgp.yml', 'r'))
    
    # Index match the position of the question_dataset dataset: indexed_test_docs.json
    if args['dataset'] == 'hotpot':
        bridge = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 22, 23, 24, 25, 26, 28,
                29, 35, 36, 37, 38, 39, 47, 54, 55, 57, 58, 60, 64, 65, 66, 68, 70, 72, 73, 79, 80, 81, 82, 84, 88]
        
        comparison = [13, 18, 21, 27, 31, 32, 33, 34, 42, 43, 44, 46, 50, 59, 62, 75, 76, 96, 117, 118, 124, 128, 
                    162, 174, 178, 191, 197, 227, 237, 244, 253, 261, 262, 272, 273, 283, 285, 296, 299, 307, 317, 318, 319, 338,
                    342, 348, 353, 354, 356, 357]
    elif args['dataset'] == '2wiki':
        bridge = [0, 1, 2, 11, 12, 13, 14, 18, 22, 28, 29, 31, 32, 33, 35, 38, 39, 41, 42, 43, 45, 46, 51, 53, 54, 55, 56, 57, 58, 64, 
                  65, 67, 68, 69, 73, 74, 77, 79, 80, 82, 83, 86, 87, 91, 92, 97, 98, 100, 104, 106]
        
        comparison = [3, 4, 5, 6, 7, 8, 10, 15, 16, 17, 19, 24, 25, 26, 30, 34, 37, 47, 48, 50, 59, 60, 61, 62, 63, 70, 71, 72,
                      76, 78, 84, 85, 89, 90, 93, 94, 95, 96, 99, 107, 114, 115, 116, 117, 119, 128, 139, 141, 145, 150]
    
    question_indices = bridge + comparison
    random.seed(args['seed'])
    random.shuffle(question_indices)
    
    # Load question set. 
    question_dataset = args['question_dataset']
    with open(question_dataset, 'r') as f:
        dataset = json.load(f)
        
    questions = [(idx, dataset[idx]) for idx in question_indices]
    
    # Get constructed knowledge graph
    G = load_graph(args['KG'])

    # Initialize the seed retriever
    retriever_name = args['init_retriever']['name']
    init_retriever = get_seeding_retriever(retriever_name, topk=args['init_retriever']['topk'], G=G, dataset=dataset)
    
    # Initialize the traversal agent if using it
    if not args['init_retriever']['no_traversal']:
        traversal_agent = get_traversal_agent(args)
    
    #==============================================================
    
    cp = [] # Checkpoint
    cp_questions = [] # Checkpoint questions
    if args['checkpoint']['resume']:
        print("Resuming from the checkpoint...")
        load_path = os.path.join(args['root_dir'], args['checkpoint']['checkpoint_path'])
        with open(load_path, 'r') as f:
            cp = json.load(f)
            cp_questions = [q['question'] for q in cp]
        print(f"Resumed from the checkpoint with {len(cp)} questions.")

    save_path = os.path.join(args['root_dir'], args['checkpoint']['save_dir'], f"{args['checkpoint']['id']}")
    os.makedirs(save_path, exist_ok=True)

    # Save the config file once
    save_config = False
    
    qp = [] # Questions and their evidence
    for i, q in tqdm(enumerate(questions), total=len(questions)):
        # questions[0] = (idx, question)
        idx = q[0]
        q = q[1]
        user_query = q['question']
        
        if user_query in cp_questions:
            # Only happen when resuming from the checkpoint
            continue
        
        if retriever_name in ['gold', 'none']:
            # No traversal needed 
            evidence = init_retriever.retrieve(user_query)
        else:
            if args['init_retriever']['no_traversal']:
                # Use the retriever only
                init_retriever.topk = args['init_retriever']['no_traversal_topk'] # Update the topk
                evidence_indices = init_retriever.retrieve(user_query) 
                evidence = [parse_evidence_string(init_retriever.all_documents[i]) for i in evidence_indices]
            else:
                evidence = get_supporting_evidence(user_query, G, init_retriever, traversal_agent, args)
            
        if evidence or retriever_name == 'none':
            if idx in bridge:
                q_type = 'bridge'
            elif idx in comparison:
                q_type = 'comparison'
            else:
                q_type = 'unknown'
                    
            record = generate_evidence_record(q_type, user_query, evidence, q['answer'], q['supports'])
            qp.append(record)
            
        else:
            print(f"No evidence found for the question: {user_query}")
        
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
        
    if os.path.exists(os.path.join(save_path, 'cp_evidence.json')):
        os.remove(os.path.join(save_path, 'cp_evidence.json'))
