# This script is used to construct the knowledge graph from the passage embeddings. 
# Nodes are connected based on the similarity of their embeddings.
# The graph consists nodes as their passage id, and each node has a feature vector of the passage embedding:
# title, passage, question_id, and emb.
# Note that question_id isn't necessary for it's useful for testing. 

import os 
import json 
import pickle
import numpy as np
import yaml 

from KGP.KG.neighbor_kg_construct import get_kg_graph_gpu, add_node_features
from KGP.LLMs.Mistral.quantize_mistral_mlx import load_config


if __name__=="__main__":
    args = load_config('./configs/kg_construct.yml')
    
    # Load the embeddings, the position in the array should match the node index as well as the passage_id
    embs = np.load(os.path.join(args['root_dir'], args['emb_file']))
    
    G = get_kg_graph_gpu(embs=embs, **args['algo_params'])
    
    # Specified run_id to avoid overwriting the previous graph
    os.makedirs(os.path.join(args['root_dir'], f"DATA/KG/graph_{args['run_id']}"), exist_ok=True)
    
    passages_data = json.load(open(os.path.join(args['root_dir'], args['passages_file'])))
    
    updated_G = add_node_features(G, passages_data, embs)
    
    # Save the updated graph into pickle file
    with open(os.path.join(args['root_dir'], f"DATA/KG/graph_{args['run_id']}", "graph.gpickle"), 'wb') as f:
        pickle.dump(updated_G, f, pickle.HIGHEST_PROTOCOL)
    
    # Save the config file    
    with open(os.path.join(args['root_dir'], f"DATA/KG/graph_{args['run_id']}", 'config.yml'), 'w') as f:
        yaml.dump(args, f, sort_keys=False)