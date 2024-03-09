import os 
from tqdm import tqdm 
import numpy as np 
import json 
import torch 
from torch.utils.data import DataLoader

from KGP.KG.dataset import DocumentsDataset, parse_data


@torch.no_grad()
def run(data, model, tokenizer, args):
    data = parse_data(data)
    
    # Save a copy of the raw data
    # Define run_id in case of overwriting
    os.makedirs(os.path.join(args['root_dir'], 'DATA/KG', f"emb_{args['model']['run_id']}"), exist_ok=True)
    with open(os.path.join(args['root_dir'], 'DATA/KG', f"emb_{args['model']['run_id']}", 'raw_data.json'), 'w') as f:
        json.dump(data, f, indent=4)
    print('Raw data saved...')    
    
    # Make sure the position of the data matches the passage_id
    dataset = DocumentsDataset(data)
    data_loader = DataLoader(dataset, batch_size=args['model']['batch_size'], shuffle=False)
    
    model = model.to(args['model']['device'])
    model.eval()
    
    cp_count = 0
    cp_embs = None 
    if args['emb_checkpoint']:
        cp_embs = np.load(args['emb_checkpoint'])
        cp_count = len(cp_embs)
        print(f"Checkpoint found at {args['emb_checkpoint']}. Resuming from {cp_count}...")
    else:
        print("No checkpoint found. Starting from scratch...")
    
    embs = []
    for id, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        passage_id, title, passage = batch # passage_id should be the same as id

        if id != passage_id:
            raise ValueError(f"Position mismatch at id: {id} and passage_id: {passage_id}.")
        
        # Skip the checkpointed embeddings if exists
        if id < cp_count:
            continue 
        
        encoded_pair = tokenizer(text=title, text_pair=passage, max_length=args['model']['max_token_len'], 
                                 return_tensors='pt', padding=True, truncation=True)
        
        for key in encoded_pair:
            encoded_pair[key] = encoded_pair[key].to(args['model']['device'])
            
        passage_emb = model(encoded_pair['input_ids'], encoded_pair['attention_mask']).detach().cpu().numpy()
        embs.append(passage_emb)
        
        # Checkpoint the embeddings
        if id != 0 and (id + 1) % args['model']['save_every'] == 0:
            temp_embs = np.concatenate(embs, axis=0)
            if cp_embs is not None:
                temp_embs = np.concatenate([cp_embs, temp_embs], axis=0)
            np.save(os.path.join(args['root_dir'], 'DATA/KG', f"emb_{args['model']['run_id']}", f'passage_embs_{len(temp_embs)}.npy'), temp_embs)
            print(f"Checkpoint saved at {len(temp_embs)}...")
        
    final_embs = np.concatenate(embs, axis=0)
    if cp_embs is not None:
        final_embs = np.concatenate([cp_embs, final_embs], axis=0)
        
    print(f"{len(final_embs)} batches processed. (Total: {len(data_loader)})")
    # Save the embeddings
    np.save(os.path.join(args['root_dir'], 'DATA/KG', f"emb_{args['model']['run_id']}", 'passage_embs.npy'), final_embs)
    return