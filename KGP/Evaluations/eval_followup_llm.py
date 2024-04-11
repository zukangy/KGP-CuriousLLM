import json
import pandas as pd
from tqdm import tqdm
from rouge import Rouge
from sentence_transformers import SentenceTransformer, util
import os

# Assuming all files are in the 'DATA/Mistral/' directory
# test_followups/ is the output from grid_search_mistral_main.py
data_dir = './DATA/Mistral/test_followups'
files = [f for f in os.listdir(data_dir) if f.endswith('.json')]

# Initialize the ROUGE and sentence transformer
rouge = Rouge()
model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')

def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as input_file:
        data = json.load(input_file)

    rouge_scores = []
    co_sim_scores = []

    for record in data:
        reference = record['follow-up']
        candidate = record['response']
        rouge_score = rouge.get_scores(candidate, reference)[0]
        # Only store the F1 score
        rouge_scores.append({
            'rouge1': rouge_score['rouge-1']['f'],
            'rougeL': rouge_score['rouge-l']['f']
        })
        
        reference_emb = model.encode(reference, convert_to_tensor=True)
        candidate_emb = model.encode(candidate, convert_to_tensor=True)
        co_sim_score = util.cos_sim(reference_emb, candidate_emb)[0].cpu().item()
        co_sim_scores.append(co_sim_score)

    avg_rouge1 = sum([score['rouge1'] for score in rouge_scores])/len(rouge_scores)
    avg_rougeL = sum([score['rougeL'] for score in rouge_scores])/len(rouge_scores)
    avg_co_sim = sum(co_sim_scores)/len(co_sim_scores)
    
    return avg_rouge1, avg_rougeL, avg_co_sim

results = []

for file in tqdm(files, desc="Processing files", total=len(files)):
    file_path = os.path.join(data_dir, file)
    rouge1, rougeL, cos_sim = process_file(file_path)
    results.append({
        'file': file,
        'rouge-1': rouge1,
        'rouge-L': rougeL,
        'cosine_similarity': cos_sim
    })

df = pd.DataFrame(results)

# Visualization or analysis of df
print(df)

# You could save this DataFrame to a CSV for further analysis
df.to_csv('evaluation_results.csv', index=False)
