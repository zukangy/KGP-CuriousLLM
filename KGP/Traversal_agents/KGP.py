from time import time
from collections import deque
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
     
     
def get_response(prompt, G, init_retriever, retriever, emb, tokenizer, args):
    seeds = init_retriever.retrieve(prompt)
    
    # Initialize the queue with seed nodes and their depth
    queue = deque([([seed], 0) for seed in seeds])  # Each item is a tuple (node, depth)
    
    visited = set()
    
    print("Start Traversal to collect supporting evidence...")
    
    start = time()
    while queue:
        curr_node, depth = queue.popleft()
        
        # Add the visited node to the set
        visited.add(curr_node[-1])
        
        if depth < args['traversal_params']['n_hop']:
            input_prompt = "Question: " + prompt + " Evidence: " + ' '.join([G.nodes[node]['passage'] for node in curr_node])
            question = retriever.get_question(input_prompt)
            
            # Tokenize the question
            question = tokenizer(question, max_length=args['retriever']['inference_params']['max_token_len'],
                                 return_tensors='pt', padding=True, 
                                 truncation=True)
            
            for key in question:
                question[key] = question[key].to(args['emb_model']['device'])
            question_emb = emb(question['input_ids'], question['attention_mask'])
            
            neighbors = list(G.neighbors(curr_node[-1]))
            neighbors_emb = [G.nodes[neighbor]['emb'].reshape(1, -1) for neighbor in neighbors]
            neighbors_emb = np.concatenate(neighbors_emb, axis=0)
            
            # Rank the neighbors based on similarity
            # sim = cosine_similarity(question_emb.cpu().detach().numpy(), neighbors_emb).flatten()
            question_emb_reshaped = question_emb.cpu().detach().numpy().reshape(-1)
            sim = np.dot(neighbors_emb, question_emb_reshaped)
            # Get the top k neighbors
            n_neighbors = args['traversal_params']['n_neighbors']
            top_neighbors = sim.argsort()[-n_neighbors:].tolist()
            
            for neighbor in top_neighbors:
                new_path = curr_node + [neighbor]
                # Append neighbors with incremented depth
                queue.append((new_path, depth + 1))
    
    end = time()
    print(f"Traversal completed in {end - start} seconds")
    
    print("For question prompt: ", prompt)
    print("The supporting evidence is: ")
    for node in visited:
        print(G.nodes[node]['passage'])
        print()