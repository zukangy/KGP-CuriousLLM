import pickle 
from KGP.LLMs.Mistral.utils import inference
from KGP.KG.mdr_encoder import Retriever_inf


def get_docs_from_kg(G, doc_field='passage'):
    
    nodes = list(G.nodes)    
    nodes.sort()
    
    documents = []
    for node in nodes:
        documents.append(G.nodes[node][doc_field])
        
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