import pickle 
from KGP.LLMs.Mistral.utils import inference
from KGP.Traversal_agents.seed_retriever import (TF_IDF_Retriever, 
                                                 BM25_Retriever, 
                                                 GoldStandard_Retriever,
                                                 No_Retriever)


def get_docs_from_kg(G, doc_field='passage'):
    nodes = list(G.nodes)    
    nodes.sort()
    
    documents = []
    for node in nodes:
        documents.append(G.nodes[node][doc_field])
        
    return documents


def get_titled_docs_from_kg(G, doc_field='passage', title_field='title'):
    nodes = list(G.nodes)    
    nodes.sort()
    
    documents = []
    for node in nodes:
        passage = G.nodes[node][doc_field]
        title = G.nodes[node][title_field]
        combined_doc = "Title: " + title + "." + " " + passage
        documents.append(combined_doc)
        
    return documents 


class Mistral_Inference:
    def __init__(self, model, tokenizer, temp, max_token_len, parse_template=True):
        self.model = model
        self.tokenizer = tokenizer
        self.temp = temp
        self.max_token_len = max_token_len
        self.parse_template = parse_template
    
    def get_question(self, prompt, verbose=0):
        return inference(self.model, 
                         prompt=prompt, 
                         tokenizer=self.tokenizer, 
                         temp=self.temp, 
                         max_token_len=self.max_token_len, 
                         parse_template=self.parse_template,
                         verbose=verbose)
    
    
def load_graph(graph_path):
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    return graph


def get_seeding_retriever(name='tfidf'):
    if name == 'tfidf':
        return TF_IDF_Retriever 
    elif name == 'bm25':
        return BM25_Retriever
    elif name == 'gold':
        return GoldStandard_Retriever
    elif name == 'none':
        return No_Retriever
    else:
        raise ValueError(f"Retriever {name} not implemented")