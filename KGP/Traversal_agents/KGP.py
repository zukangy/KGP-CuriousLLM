from time import time
from collections import deque
     
     
def get_supporting_evidence(prompt, G, init_retriever, retriever, args, verbose=0):
    
    seeds = init_retriever.retrieve(prompt)
    
    # Initialize the queue with seed nodes and their depth
    queue = deque([([seed], 0) for seed in seeds])  # Each item is a tuple (node, depth)
    visited = set()
    
    if verbose == 1:
        print("Start Traversal to collect supporting evidence...")
    
    start = time()
    
    check = 0
    while queue:
        curr_path, depth = queue.popleft()
        
        # Add the visited node to the set
        visited.add(curr_path[-1])
        
        if depth < args['retriever']['traversal_params']['n_hop']:
            top_neighbors = retriever.get_top_k_neighbors(G, curr_path, prompt)
            
            for neighbor in top_neighbors:
                new_path = curr_path + [neighbor]
                # Append neighbors with incremented depth
                queue.append((new_path, depth + 1))
                
            check += 1
    
    end = time()
    
    if verbose == 1:
        print(f"Traversal completed in {end - start} seconds")
        print("Number of nodes visited: ", len(visited))
    
    if visited:
        return [f"Title: {G.nodes[n]['title']}. Evidence: {G.nodes[n]['passage']}" for n in visited]
    return
    