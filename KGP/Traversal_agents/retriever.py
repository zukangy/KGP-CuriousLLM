from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TF_IDF_retriever:
    def __init__(self, topk, all_documents, model_params=None):
        self.topk = topk
        self.init_model(all_documents, model_params)
        
    def retrieve(self, query):
        query_emb = self.vectorizer.transform([query])
        cosine_sim = cosine_similarity(query_emb, self.tfidf_matrix).flatten()
        return cosine_sim.argsort()[-self.topk:][::-1]
        
    def init_model(self, all_documents, model_params=None):
        if model_params is None:
            model_params = {}
        self.vectorizer = TfidfVectorizer(**model_params)
        tfidf_matrix = self.vectorizer.fit_transform(all_documents)
        self.tfidf_matrix = tfidf_matrix
        
        