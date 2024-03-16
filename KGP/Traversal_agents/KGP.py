from time import time
from collections import deque
from sentence_transformers import util
     
     
def get_supporting_evidence(prompt, G, init_retriever, retriever, sentence_emb, args, verbose=0):
    
    emb = sentence_emb
    seeds = init_retriever.retrieve(prompt)
    
    # Initialize the queue with seed nodes and their depth
    queue = deque([([seed], 0) for seed in seeds])  # Each item is a tuple (node, depth)
    visited = set()
    if verbose == 1:
        print("Start Traversal to collect supporting evidence...")
    
    start = time()
    
    check = 0
    while queue:
        curr_node, depth = queue.popleft()
        
        # Add the visited node to the set
        visited.add(curr_node[-1])
        
        if depth < args['traversal_params']['n_hop']:
            input_prompt = "Question: " + prompt + " Evidence: " + ' '.join([G.nodes[node]['passage'] for node in curr_node])
            question = retriever.get_question(input_prompt)
            
            # Encode the question
            question_emb = emb.encode(question, device=args['emb_model']['device'])
            
            neighbors = list(G.neighbors(curr_node[-1]))
            neighbors_passages = [G.nodes[neighbor]['passage'] for neighbor in neighbors]
            
            neighbors_emb = emb.encode(neighbors_passages, device=args['emb_model']['device'])
            
            sim_scores = util.dot_score(question_emb, neighbors_emb).cpu().numpy().flatten()
            
            # Get the top k neighbors
            n_neighbors = args['traversal_params']['n_neighbors']
            top_neighbors_indices = sim_scores.argsort()[-n_neighbors:].tolist()
            top_neighbors = [neighbors[i] for i in top_neighbors_indices]
            
            for neighbor in top_neighbors:
                new_path = curr_node + [neighbor]
                # Append neighbors with incremented depth
                queue.append((new_path, depth + 1))
                
            check += 1
    
    end = time()
    if verbose == 1:
        print(f"Traversal completed in {end - start} seconds")
        print("Number of nodes visited: ", len(visited))
    
    if visited:
        return [G.nodes[n]['passage'] for n in visited]
    return None
    