import pickle 
from KGP.LLMs.Mistral.utils import inference


def get_titled_docs_from_kg(G, doc_field='passage', title_field='title'):
    nodes = list(G.nodes)    
    nodes.sort()
    
    documents = []
    for node in nodes:
        passage = G.nodes[node][doc_field]
        title = G.nodes[node][title_field]
        combined_doc = "TITLE: " + title + "." + " " + passage
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


def generate_evidence_record(type, question, evidence, answer, supports):
    record = {
        "type": type,
        "question": question,
        "evidence": evidence,
        "answer": answer,
        "supports": supports
    }
    return record


def parse_evidence_string(evidence):
    return ''.join(evidence.split('.')[1:]).strip() + '.'