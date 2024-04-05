from tqdm import tqdm
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import torch 
from torch.utils.data import dataloader

from KGP.KG.dataset import EmbeddingDataset


# Below implementation is very slow for large embeddings, should consider using batch KNN or approximate KNN.
def get_kg_graph_cpu(embs, **algo_params):
    """
    Construct a graph from the embeddings of the entities in the knowledge graph.
    Args:
        embs: numpy array of shape (num_entities, emb_dim)
        algo_params: dict of algorithm parameters
            params: n_neighbors, radius, algorithm, leaf_size, metric, p, n_jobs
            n_neighbors: int, optional (default = 5)
                Number of neighbors to use by default for kneighbors queries.
            radius: float, optional (default = 1.0)
                Range of parameter space to use by default for radius neighbors queries.
            metric: string or callable, default 'minkowski', ['minkowski', 'cosine', 'euclidean']
                the distance metric to use for the tree
    Returns:
        kg_graph: networkx graph
    """
    # Check if the algorithm parameters are valid
    valid_params = ['n_neighbors', 'radius', 'algorithm', 'leaf_size', 'metric', 'p', 'n_jobs']
    for param in algo_params:
        assert param in valid_params, f'Invalid parameter: {param}'
        
    nn_model = NearestNeighbors(n_jobs=-1, **algo_params)
    nn_model.fit(embs)
    
    # Get the nearest neighbors
    print('Finding nearest neighbors')
    n_neighbors = algo_params['n_neighbors']
    _, indices = nn_model.kneighbors(embs, n_neighbors)
    print('Nearest neighbors found')
    
    # Create a graph from the nearest neighbors
    kg_graph = nx.Graph()
    for i in tqdm(range(len(embs)), total=len(embs), desc='Adding edges to the graph'):
        for j in indices[i]:
            if i != j:
                kg_graph.add_edge(i, j)
    return kg_graph 


def normalize_embeddings(embeddings):
    norms = torch.linalg.vector_norm(embeddings, dim=1, keepdim=True)
    return embeddings / norms
    
    
def get_kg_graph_gpu(embs, n_neighbors=20, metric='cosine', batch_size=100,
                     device='cuda:0'):
    """ Pytorch implementation to speed up similarity computation 
    metric: ['cosine', 'euclidean']
    """
    embeddings = torch.tensor(embs).float().to(device)

    if metric == 'cosine':
        embeddings = normalize_embeddings(embeddings)
        
    if metric == 'euclidean':
        # Pre-compute squared norms of embeddings to use in Euclidean distance computation
        sq_norms = torch.sum(embeddings ** 2, axis=1, keepdim=True)
    
    dataset = EmbeddingDataset(embeddings=embeddings)
    data_loader = dataloader.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    G = nx.Graph()
    
    for batch in tqdm(data_loader, total=len(data_loader)):
        batch_indices, batch_embeddings = batch
        batch_embeddings = batch_embeddings.to(device)
        
        if metric == 'cosine':
            similarity_matrix = torch.mm(batch_embeddings, embeddings.t())
        elif metric == 'euclidean':
            # Compute squared Euclidean distances efficiently
            batch_sq_norms = torch.sum(batch_embeddings ** 2, axis=1, keepdim=True)
            distance_matrix = sq_norms.t() - 2 * torch.mm(batch_embeddings, embeddings.t()) + batch_sq_norms
            similarity_matrix = -distance_matrix  # Negative distances since torch.topk finds the largest values
        
        topk_indices = torch.topk(similarity_matrix, k=n_neighbors+1, largest=True).indices

        for idx, i in enumerate(batch_indices):
            for j in topk_indices[idx]:
                if i.item() != j.item():
                    G.add_edge(i.item(), j.item())
    return G


def add_node_features(G, passages_data, embs):
    for node in tqdm(G.nodes, total=len(G.nodes), desc='Adding node features'):
        passage_dict = passages_data[node]
        if passage_dict['passage_id'] == node:
            node_features = {
                'title': passage_dict['title'],
                'passage': passage_dict['passage'],
                'emb': embs[node]
            }
            G.nodes[node].update(node_features)
        else:
            print(f"Node {node} not found in passages data")
    return G 