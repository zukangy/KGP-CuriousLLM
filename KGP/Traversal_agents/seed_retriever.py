from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
import torch 

from KGP.KG.mdr_encoder import Retriever_inf
from KGP.MDR.tokenizer import load_tokenizer
from KGP.Traversal_agents.utils import get_titled_docs_from_kg


class Base_retriever(ABC):
    def __init__(self, topk, G):
        self.topk = topk 
        self.G = G
        
    @abstractmethod
    def retrieve(self, query):
        """Return list- or 1d array-like object of indices of the top-k documents
        # e.g array([99381, 99367, 99340, 99379, 99321])
        """
        raise NotImplementedError


class TF_IDF_Retriever(Base_retriever):
    def __init__(self, topk, G):
        super().__init__(topk, G)
        self.all_documents = get_titled_docs_from_kg(self.G)
        self.vectorizer, self.tfidf_matrix = self.init_model(self.all_documents)
        
    def retrieve(self, query):
        query_emb = self.vectorizer.transform([query])
        cosine_sim = cosine_similarity(query_emb, self.tfidf_matrix).flatten()
        return cosine_sim.argsort()[-self.topk:][::-1]
    
    @staticmethod    
    def init_model(all_documents):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_documents)
        return vectorizer, tfidf_matrix
        

class BM25_Retriever(Base_retriever):
    def __init__(self, topk, G):
        super().__init__(topk, G)
        self.all_documents = get_titled_docs_from_kg(G)
        self.tokenizer = word_tokenize
        self.bm25 = self.init_model(G)
    
    def init_model(self, G):
        corpus = [word_tokenize(doc) for doc in self.all_documents]
        bm25 = BM25Okapi(corpus)
        return bm25
        
    def retrieve(self, query):
        scores = self.bm25.get_scores(self.tokenizer(query))
        return scores.argsort()[-self.topk:][::-1]
        
        
class GoldStandard_Retriever(Base_retriever):
    def __init__(self, topk, G):
        super().__init__(topk, G)
        self.gold_standard = self.init_model(self.G)
    
    def retrieve(self, query):
        supports = self.gold_standard[query]
        return [i[1] for i in supports]
    
    @staticmethod
    def init_model(dataset):
        gold_standard = {i['question']: i['supports'] for i in dataset}
        return gold_standard
    
    
class No_Retriever(Base_retriever):
    def __init__(self, topk, G):
        super().__init__(topk, G)
    
    def retrieve(self, query):
        return []
    

class MDR_Retriever(Base_retriever):
    """This retriever simulates a two-hop retrieval process using the MDR model"""
    def __init__(self, topk, G, device='mps'):
        super().__init__(topk, G)
        self.embs = self.init_emb_from_graph(self.G).to(device)
        self.all_documents = get_titled_docs_from_kg(self.G)
        self.model, self.tokenizer = self.init_model()
        self.model = self.model.to(device)
        self.device = device
    
    @staticmethod
    def init_model():
        # Please manually change below in your workspace
        print("Initializing MDR...")
        tokenizer, config = load_tokenizer(model_name='deepset/tinyroberta-squad2')
        model = Retriever_inf(config, base_model='deepset/tinyroberta-squad2')
        model_checkpoint = torch.load("./models/checkpoint_tinyRo/MDR/HotpotQA/1/2024-03-15_00-39-31/best_model.pt")
        model.load_state_dict(model_checkpoint['model_state_dict'])
        return model, tokenizer
        
    @staticmethod
    def init_emb_from_graph(G):
        print("Initializing embeddings from the graph...")
        nodes = list(G.nodes)
        nodes.sort()
        embs = [torch.tensor(G.nodes[n]['emb']) for n in nodes]
        return torch.stack(embs) 
    
    def retrieve(self, query):
        encoded_query = self.tokenizer(text=query, max_length=200, return_tensors='pt', 
                                       padding=True, truncation=True).to(self.device)
        query_emb = self.model(encoded_query['input_ids'], encoded_query['attention_mask']) # Output is a in-device tensor
        scores = torch.matmul(query_emb, self.embs.transpose(0, 1))
        scores = scores.squeeze(0).detach().cpu().numpy()
        topk = scores.argsort()[-self.topk:][::-1] # topk in the first hop
        
        two_hop_top_ks = []
        two_hop_top_ks.extend(topk.tolist())
        
        # second hop
        for idx in topk:
            encoded_query2 = self.tokenizer(text=query, text_pair=self.all_documents[idx], max_length=200, return_tensors='pt', 
                                            padding=True, truncation=True).to(self.device)
            query2_emb = self.model(encoded_query2['input_ids'], encoded_query2['attention_mask'])
            scores2 = torch.matmul(query2_emb, self.embs.transpose(0, 1))
            scores2 = scores2.squeeze(0).detach().cpu().numpy()
            # Select topk from the second hop
            topk2 = scores2.argsort()[-self.topk:][::-1]
            two_hop_top_ks.extend(topk2.tolist())
        
        return list(set(two_hop_top_ks))
            
        
def get_seeding_retriever(name, topk, G=None, dataset=None):
    if name == 'tfidf':
        return TF_IDF_Retriever(topk=topk, G=G) 
    elif name == 'bm25':
        return BM25_Retriever(topk=topk, G=G)
    elif name == 'gold':
        #E Provide the dataset with the gold standard supports
        return GoldStandard_Retriever(topk=topk, G=dataset)
    elif name == 'mdr':
        return MDR_Retriever(topk=topk, G=G)
    elif name == 'none':
        return No_Retriever(topk=topk, G=G)
    else:
        raise ValueError(f"Retriever {name} not implemented")