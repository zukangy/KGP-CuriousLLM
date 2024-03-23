import json 
from tqdm import tqdm 
from sentence_transformers import util
from sentence_transformers import SentenceTransformer


DEVICE = "mps"

if __name__=="__main__":
    with open('./DATA/KG/evidence/checkpoint_mdr_agent/evidence.json', 'r') as f:
        data = json.load(f)
        
    # Load sentence transformer
    emb = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
    total_supports = 0
    correct_supports = 0
    total_comparison_support = 0
    total_bridge_support = 0
    correct_comparison = 0
    correct_bridge = 0
    for record in tqdm(data, total=len(data)):
        type = record['type']
        question = record['question']
        found_evidence = record['found_evidence']
        evidence_emb = emb.encode(found_evidence, device=DEVICE)
        supports = record['supports']
        for s in supports:
            total_supports += 1
            support_emb = emb.encode(s[1], device=DEVICE)
            sim_score = util.dot_score(support_emb, evidence_emb).cpu().numpy().flatten() 
            
            if type == 'comparison':
                total_comparison_support += 1
                
            elif type == 'bridge':
                total_bridge_support += 1
            
            if max(sim_score) > 0.8:
                correct_supports += 1
                
                if type == 'comparison':
                    correct_comparison += 1
                elif type == 'bridge':
                    correct_bridge += 1
            
    print(f"Total supports: {total_supports} | Correct supports: {correct_supports} | EM: {(correct_supports / total_supports):.4f}")
    print(f"Total comparison supports: {total_comparison_support} | Correct comparison supports: {correct_comparison} | EM: {(correct_comparison / total_comparison_support):.4f}")
    print(f"Total bridge supports: {total_bridge_support} | Correct bridge supports: {correct_bridge} | EM: {(correct_bridge / total_bridge_support):.4f}")
    