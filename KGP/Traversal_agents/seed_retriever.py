from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi


class Base_retriever(ABC):
    def __init__(self, topk):
        self.topk = topk
        
    @abstractmethod
    def init_model(self, all_documents, model_params=None):
        raise NotImplementedError
        
    @abstractmethod
    def retrieve(self, query):
        """Return list- or array-like object of indices of the top-k documents
        # e.g array([99381, 99367, 99340, 99379, 99321])
        """
        raise NotImplementedError


class TF_IDF_Retriever(Base_retriever):
    def __init__(self, topk, all_documents, model_params=None):
        super().__init__(topk)
        self.init_model(all_documents, model_params)
        
    def init_model(self, all_documents, model_params=None):
        if model_params is None:
            model_params = {}
        self.vectorizer = TfidfVectorizer(**model_params)
        tfidf_matrix = self.vectorizer.fit_transform(all_documents)
        self.tfidf_matrix = tfidf_matrix
        
    def retrieve(self, query):
        query_emb = self.vectorizer.transform([query])
        cosine_sim = cosine_similarity(query_emb, self.tfidf_matrix).flatten()
        return cosine_sim.argsort()[-self.topk:][::-1]
        

class BM25_Retriever(Base_retriever):
    def __init__(self, topk, all_documents, tokenizer=None, model_params=None):
        super().__init__(topk)
        if tokenizer is None:
            self.tokenizer = word_tokenize
        else:
            self.tokenizer = tokenizer
        corpus = [self.tokenizer(doc) for doc in all_documents]
        self.init_model(corpus, model_params)
        
    def init_model(self, all_documents, model_params=None):
        if model_params is None:
            model_params = {}
        self.bm25 = BM25Okapi(all_documents, **model_params)
        
    def retrieve(self, query):
        scores = self.bm25.get_scores(self.tokenizer(query))
        return scores.argsort()[-self.topk:][::-1]
        
        
class GoldStandard_Retriever(Base_retriever):
    def __init__(self, topk, all_documents, model_params=None):
        super().__init__(topk)
        self.init_model(all_documents, model_params)
        
    def init_model(self, all_documents, model_params=None):
        self.gold_standard = {i['question']: i['supports'] for i in all_documents}
    
    def retrieve(self, query):
        supports = self.gold_standard[query]
        return [i[1] for i in supports]
    
    
class No_Retriever(Base_retriever):
    def __init__(self, topk, all_documents, model_params=None):
        super().__init__(topk)
        
    def init_model(self, all_documents, model_params=None):
        pass
    
    def retrieve(self, query):
        return []
    

# TODO Finish this 
class MDR_retriever(Base_retriever):
    def __init__(self, corpus):
        self.corpus = corpus
    
    def retrieve(self, data, i):
        return self.corpus[i]