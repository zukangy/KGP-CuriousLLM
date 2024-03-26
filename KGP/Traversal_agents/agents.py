import os 
from abc import ABC, abstractmethod
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from sentence_transformers import util, SentenceTransformer
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch 

from KGP.KG.mdr_encoder import Retriever_inf
from KGP.MDR.tokenizer import load_tokenizer
from KGP.Traversal_agents.utils import Mistral_Inference
from KGP.LLMs.Mistral.utils import load_lora_model


class Base_Agent(ABC):
    def __init__(self, args):
        self.args = args
        
    @abstractmethod
    def get_top_k_neighbors(self, G, curr_path, prompt):
        raise NotImplementedError
    

class Mistral_Agent(Base_Agent):
    def __init__(self, args):
        super().__init__(args)
        self.retriever = self.init_mistral_inference(self.args)
        self.emb = self.init_sentence_transformer(self.args)
    
    def get_top_k_neighbors(self, G, curr_path, prompt):
        question = self.retriever.get_question(self.get_prompt(prompt, curr_path, G))
        question_emb = self.emb.encode(question, device=self.args['device'])
        
        neighbors = list(G.neighbors(curr_path[-1]))
        
        if self.args['titled']:
            neighbors_passages = [f"Title: {G.nodes[neighbor]['title']}. {G.nodes[neighbor]['passage']}" for neighbor in neighbors]
        else:
            neighbors_passages = [G.nodes[neighbor]['passage'] for neighbor in neighbors]
            
        neighbors_emb = self.emb.encode(neighbors_passages, device=self.args['device'])
        
        sim_scores = self.get_sim_scores(question_emb, neighbors_emb)
        
        top_neighbors_indices = sim_scores.argsort()[-self.args['retriever']['traversal_params']['n_neighbors'] : ].tolist()
        top_neighbors = [neighbors[i] for i in top_neighbors_indices]
        
        return top_neighbors
    
    @staticmethod
    def get_prompt(prompt, curr_path,  G):
        return "Question: " + prompt + " Evidence: " + ' '.join([G.nodes[node]['passage'] for node in curr_path])
    
    @staticmethod
    def get_sim_scores(question_emb, neighbors_emb):
        return util.dot_score(question_emb, neighbors_emb).cpu().numpy().flatten()
    
    @staticmethod
    def init_mistral_inference(args, parse_template=True):
        print("Initializing Mistral Inference...")
        # Initialize Mistral Inference
        model_path = os.path.join(args['root_dir'], args['retriever']['model'])
        adapter_path = os.path.join(args['root_dir'], args['retriever']['adapter'])
        lora_rank = args['retriever']['model_params']['lora_rank']
        lora_layer = args['retriever']['model_params']['lora_layers']
        model, tokenizer = load_lora_model(model=model_path, adapter_file=adapter_path, 
                                            lora_rank=lora_rank, lora_layer=lora_layer)
        temp = args['retriever']['inference_params']['temp']
        max_token_len = args['retriever']['inference_params']['max_token_len']
        return Mistral_Inference(model, tokenizer, temp, max_token_len, parse_template)
    
    @staticmethod
    def init_sentence_transformer(args):
        print("Initializing Sentence Transformer...")
        return SentenceTransformer(args['emb_model']['model'])
    
    
class TF_IDF_Agent(Base_Agent):
    def __init__(self, args):
        super().__init__(args)
        
    def get_top_k_neighbors(self, G, curr_path, prompt):
        neighbors = list(G.neighbors(curr_path[-1]))
        neighbors_passages = [f"Title: {G.nodes[neighbor]['title']}. {G.nodes[neighbor]['passage']}" 
                              for neighbor in neighbors]  
        
        vectorizer = TfidfVectorizer() 
        tfidf_matrix = vectorizer.fit_transform(neighbors_passages)
        query_emb = vectorizer.transform([prompt])
        cosine_sim = cosine_similarity(query_emb, tfidf_matrix).flatten()
        topk = cosine_sim.argsort()[-self.args['retriever']['traversal_params']['n_neighbors']:][::-1]
        return [neighbors[i] for i in topk]

        
class BM25_Agent(Base_Agent):
    def __init__(self, args):
        super().__init__(args)
        self.tokenizer = word_tokenize
        
    def get_top_k_neighbors(self, G, curr_path, prompt):
        neighbors = list(G.neighbors(curr_path[-1]))
        neighbors_passages = [f"Title: {G.nodes[neighbor]['title']}. {G.nodes[neighbor]['passage']}" for neighbor in neighbors] 
        
        corpus = [self.tokenizer(doc) for doc in neighbors_passages]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(self.tokenizer(prompt))
        topk = scores.argsort()[-self.args['retriever']['traversal_params']['n_neighbors']:][::-1]
        return [neighbors[i] for i in topk]
    

class MDR_Agent(Base_Agent):
    def __init__(self, args):
        super().__init__(args)
        self.model, self.tokenizer = self.init_mdr()
        
    def get_top_k_neighbors(self, G, curr_path, prompt):
        neighbors = list(G.neighbors(curr_path[-1]))
        self.model = self.model.to(self.args['device'])
        try: 
            neighbors_passage_embs = [G.nodes[neighbor]['emb'] for neighbor in neighbors]
            neighbors_passage_embs = torch.tensor(np.vstack(neighbors_passage_embs)).to(self.args['device'])
            
        except:
            neighbors_titles = [G.nodes[neighbor]['title'] for neighbor in neighbors]
            neighbors_passages = [G.nodes[neighbor]['passage'] for neighbor in neighbors]    
        
            encoded_pair = self.tokenizer(text=neighbors_titles, text_pair=neighbors_passages, max_length=200, 
                                          return_tensors='pt', padding=True, truncation=True).to(self.args['device'])
            
            neighbors_passage_embs = self.model(encoded_pair['input_ids'], encoded_pair['attention_mask'])
        
        concated_passage = self.concat_title_passage(G.nodes[curr_path[-1]]['title'], G.nodes[curr_path[-1]]['passage'])
        curr_query_encoded = self.tokenizer(text=prompt, text_pair=concated_passage, 
                                            max_length=200, return_tensors='pt', padding=True, truncation=True).to(self.args['device'])
        curr_query_emb = self.model(curr_query_encoded['input_ids'], curr_query_encoded['attention_mask'])
        scores = torch.matmul(curr_query_emb, neighbors_passage_embs.transpose(0, 1))
        scores = scores.squeeze(0).detach().cpu().numpy()
        topk = scores.argsort()[-self.args['retriever']['traversal_params']['n_neighbors']:][::-1]
        return [neighbors[i] for i in topk]
        
    @staticmethod
    def concat_title_passage(title, passage):
        return f"Title: {title}. {passage}"
    
    @staticmethod
    def init_mdr():
        print("Initializing MDR...")
        tokenizer, config = load_tokenizer(model_name='deepset/tinyroberta-squad2')
        model = Retriever_inf(config, base_model='deepset/tinyroberta-squad2')
        model_checkpoint = torch.load("./models/checkpoint_tinyRo/MDR/HotpotQA/1/2024-03-15_00-39-31/best_model.pt")
        model.load_state_dict(model_checkpoint['model_state_dict'])
        return model, tokenizer
    

class T5_Agent(Base_Agent):
    def __init__(self, args):
        super().__init__(args)
        self.model, self.tokenizer = self.init_t5(self.args)
        self.emb = self.init_sentence_transformer(self.args)
    
    def get_top_k_neighbors(self, G, curr_path, prompt):
        input_text = self.get_prompt(prompt, curr_path, G)
        
        source = self.tokenizer.batch_encode_plus(
            [input_text],
            max_length=self.args['retriever']['T5_params']['max_source_length'],
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        sources_ids = source["input_ids"].to(dtype=torch.long)
        sources_ids = sources_ids.to(self.args['device'])
        
        self.model = self.model.to(self.args['device'])
        self.model.eval()
        
        outputs = self.model.generate(input_ids=sources_ids, 
                                      max_length=self.args['retriever']['T5_params']['max_target_length'])
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        output_emb = self.emb.encode(output, device=self.args['device']) 
        
        neighbors = list(G.neighbors(curr_path[-1]))
        
        neighbors_passages = [G.nodes[neighbor]['passage'] for neighbor in neighbors]
        
        neighbors_emb = self.emb.encode(neighbors_passages, device=self.args['device'])
        
        sim_scores = self.get_sim_scores(output_emb, neighbors_emb)
        
        top_neighbors_indices = sim_scores.argsort()[-self.args['retriever']['traversal_params']['n_neighbors'] : ].tolist()
        top_neighbors = [neighbors[i] for i in top_neighbors_indices]
        
        return top_neighbors
        
    
    @staticmethod
    def get_prompt(prompt, curr_path, G):
        input_text = prompt + '\n'.join([G.nodes[node]['passage'] for node in curr_path])
        input_text = " ".join(input_text.split())
        return input_text
    
    @staticmethod    
    def init_t5(args):
        print("Initializing T5...")
        model_path = os.path.join(args['root_dir'], args['retriever']['T5_params']['model_path'])
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer
    
    @staticmethod
    def init_sentence_transformer(args):
        print("Initializing Sentence Transformer...")
        return SentenceTransformer(args['emb_model']['model'])
    
    @staticmethod
    def get_sim_scores(source_emb, neighbors_emb):
        return util.dot_score(source_emb, neighbors_emb).cpu().numpy().flatten()
    

def get_traversal_agent(args):
    if args['retriever']['name'] == 'mistral':
        return Mistral_Agent(args)
        
    elif args['retriever']['name'] == 'tfidf':
        return TF_IDF_Agent(args)
        
    elif args['retriever']['name'] == 'bm25':
        return BM25_Agent(args)
        
    elif args['retriever']['name'] == 'mdr':
        return MDR_Agent(args)
        
    elif args['retriever']['name'] == 't5':
        return T5_Agent(args)
        
    else:
        raise ValueError("Invalid retriever name")