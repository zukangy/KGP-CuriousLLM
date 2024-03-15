import os 
import math 
import yaml 
from datetime import datetime 
from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import get_linear_schedule_with_warmup

from KGP.MDR.dataset import HotpotQANeg
from KGP.MDR.utils import move_to_gpu


def train(train_data, val_data, model, tokenizer, collate, args, checkpoint_dir=None):
    if args['do_train']:
        train_dataset = HotpotQANeg(train_data, tokenizer, args, train=True)

        train_dataloader = DataLoader(train_dataset, batch_size=args['train_bsz'], pin_memory=True, 
                                      collate_fn=collate, num_workers=args['num_workers'], shuffle=True)

        val_dataset = HotpotQANeg(val_data, tokenizer, args, train=False)

        val_dataloader = DataLoader(val_dataset, batch_size=args['eval_bsz'], pin_memory=True, 
                                    collate_fn=collate, num_workers=args['num_workers'], shuffle=False)
        
        t_total = len(train_dataloader) * args['epochs']
        warmup_steps = math.ceil(t_total * args['warm_ratio'])

        print("Start Training")
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Checkpoint the model at the end of each epoch
        os.makedirs(f"./models/{args['checkpoint_root_dir']}/MDR/HotpotQA/{args['checkpoint_id']}/{timestamp}", exist_ok=True)
        
        # Save the config info 
        with open(f"./models/{args['checkpoint_root_dir']}/MDR/HotpotQA/{args['checkpoint_id']}/{timestamp}/mdr_config.yml", 'w') as file:
            yaml.dump(args, file, sort_keys=False)
        
        epoch = 0
        best_mrr = 0
        batch_num = 0
        hist_losses = []
        
        no_decay = ['bias', 'LayerNorm.weight']
        if checkpoint_dir is not None:
            print('Resuming training from checkpoint...')
            checkpoint = torch.load(checkpoint_dir)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(args['gpus'])
            
            # Load optimizer
            optimizer_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(
                    nd in n for nd in no_decay)], 'weight_decay': args['weight_decay']},
                {'params': [p for n, p in model.named_parameters() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = Adam(optimizer_parameters, lr=float(args['lr']), eps=float(args['adam_epsilon']))
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, 
                                                    num_training_steps=t_total)
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            epoch = checkpoint['epoch']
            batch_num = checkpoint['batch_num']
            curr_loss = checkpoint['smoothed_loss']
            hist_losses = checkpoint['losses']
            curr_mrr = checkpoint['curr_mrr']
            best_mrr = checkpoint['best_mrr']
            
            print("Loaded checkpoint from epoch {} batch {}".format(epoch, batch_num))
            print("Loss: {}, Curr MRR: {}, Best MRR: {}".format(curr_loss, curr_mrr, best_mrr))
            
        else:
            print('Starting training from scratch...')
            model = model.to(args['gpus'])
            optimizer_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(
                    nd in n for nd in no_decay)], 'weight_decay': args['weight_decay']},
                {'params': [p for n, p in model.named_parameters() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = Adam(optimizer_parameters, lr=float(args['lr']), eps=float(args['adam_epsilon']))
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, 
                                                        num_training_steps=t_total)
            
        if epoch >= args['epochs']:
            print("Epoch {} has already been trained".format(epoch))
            return 
        
        model = model.to(args['gpus'])
        
        for epoch in range(epoch, args['epochs']):
            model.train()
            progress = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch}')
            for idx, samples in progress:
                if batch_num != 0 and idx <= batch_num:
                    # Skip the batches that have already been trained
                    continue
                
                batch_num = 0
                
                samples = move_to_gpu(samples, device=args['gpus'])
                loss = mp_loss(model, samples)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                
                if idx % 100 == 0:
                    hist_losses.append(loss.item())
                    print("Epoch: {}, Batch: {}, Loss: {}".format(epoch, idx, np.mean(hist_losses)))
                    
                if idx % args['checkpoint_step'] == 0 and idx != 0:
                    print(f"Start Evaluation after {args['checkpoint_step']} steps...")
                    mrr_1, mrr_2 = eval(model, val_dataloader, args)
                    mrr_avg = (mrr_1 + mrr_2) / 2
                    
                    print("Epoch: {}, Loss: {}, MRR_1: {}, MRR_2: {}, Ave_MRR: {}, Best_MRR: {}".format(epoch, np.mean(hist_losses), mrr_1, mrr_2, 
                                                                                                        mrr_avg, best_mrr))
                    
                    model_paths = [f"./models/{args['checkpoint_root_dir']}/MDR/HotpotQA/{args['checkpoint_id']}/{timestamp}/model.pt"]
                    
                    if mrr_avg > best_mrr:
                        best_mrr = mrr_avg
                        
                        print('Saving best model...')
                        model_paths.append(f"./models/{args['checkpoint_root_dir']}/MDR/HotpotQA/{args['checkpoint_id']}/{timestamp}/best_model.pt")
                    
                    for path in model_paths:
                        torch.save({
                            'epoch': epoch,
                            'batch_num': idx,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'smoothed_loss': np.mean(hist_losses),
                            'losses': hist_losses,
                            'curr_mrr': mrr_avg,
                            'best_mrr': best_mrr}, path)      
                
    else:
        val_dataset = HotpotQANeg(val_data, tokenizer, args, train = False)
        val_dataloader = DataLoader(val_dataset, batch_size=args.eval_bsz, pin_memory=True, 
                                    collate_fn=collate, num_workers=args['num_workers'], shuffle=False)
        
        mrr_1, mrr_2 = eval(model, val_dataloader, args)
        mrr_avg = (mrr_1 + mrr_2) / 2

        print("MRR_1: {}, MRR_2: {}, Ave_MRR: {}".format(mrr_1, mrr_2, (mrr_1 + mrr_2) / 2))


def mp_loss(model, batch):
    embs = model(batch)
    loss_fct = CrossEntropyLoss(ignore_index = -1)

    c_embs = torch.cat([embs["c1_emb"], embs["c2_emb"]], dim = 0) # 2B x d
    n_embs = torch.cat([embs["n1_emb"].unsqueeze(1), embs["n2_emb"].unsqueeze(1)], dim = 1) # B*2*M*h

    scores_1 = torch.mm(embs["q_emb"], c_embs.t()) # B x 2B
    n_scores_1 = torch.bmm(embs["q_emb"].unsqueeze(1), n_embs.permute(0, 2, 1)).squeeze(1) # B x 2B
    scores_2 = torch.mm(embs["q_c1_emb"], c_embs.t()) # B x 2B
    n_scores_2 = torch.bmm(embs["q_c1_emb"].unsqueeze(1), n_embs.permute(0, 2, 1)).squeeze(1) # B x 2B

    # mask the 1st hop
    bsize = embs["q_emb"].size(0)
    scores_1_mask = torch.cat([torch.zeros(bsize, bsize), torch.eye(bsize)], dim=1).to(embs["q_emb"].device)
    scores_1 = scores_1.float().masked_fill(scores_1_mask.bool(), float('-inf')).type_as(scores_1)
    scores_1 = torch.cat([scores_1, n_scores_1], dim=1)
    scores_2 = torch.cat([scores_2, n_scores_2], dim=1)

    target_1 = torch.arange(embs["q_emb"].size(0)).to(embs["q_emb"].device)
    target_2 = torch.arange(embs["q_emb"].size(0)).to(embs["q_emb"].device) + embs["q_emb"].size(0)

    loss = loss_fct(scores_1, target_1) + loss_fct(scores_2, target_2)

    return loss


@torch.no_grad()
def eval(model, dataloader, args):
    model.eval()

    rrs_1, rrs_2 = [], []
    for batch in tqdm(dataloader):
        batch = move_to_gpu(batch, device=args['gpus'])

        embs = model(batch)
        eval_results = mhop_eval(embs)

        _rrs_1, _rrs_2 = eval_results['rrs_1'], eval_results['rrs_2']
        rrs_1 += _rrs_1
        rrs_2 += _rrs_2
    
    return np.mean(rrs_1), np.mean(rrs_2)


def mhop_eval(embs):
    c_embs = torch.cat([embs['c1_emb'], embs['c2_emb']], dim=0) # (2B) * D
    n_embs = torch.cat([embs["n1_emb"].unsqueeze(1), embs["n2_emb"].unsqueeze(1)], dim=1) # B * 2 * D


    scores_1 = torch.mm(embs["q_emb"], c_embs.t()) #B * 2B
    n_scores_1 = torch.bmm(embs["q_emb"].unsqueeze(1), n_embs.permute(0, 2, 1)).squeeze(1) # B * 2
    scores_2 = torch.mm(embs["q_c1_emb"], c_embs.t()) #B * 2B
    n_scores_2 = torch.bmm(embs["q_emb"].unsqueeze(1), n_embs.permute(0, 2, 1)).squeeze(1) # B * 2


    bsize = embs["q_emb"].size(0)
    scores_1_mask = torch.cat([torch.zeros(bsize, bsize), torch.eye(bsize)], dim=1).to(embs["q_emb"].device)
    scores_1 = scores_1.float().masked_fill(scores_1_mask.bool(), float('-inf')).type_as(scores_1)
    scores_1 = torch.cat([scores_1, n_scores_1], dim=1)
    scores_2 = torch.cat([scores_2, n_scores_2], dim=1)
    target_1 = torch.arange(embs["q_emb"].size(0)).to(embs["q_emb"].device)
    target_2 = torch.arange(embs["q_emb"].size(0)).to(embs["q_emb"].device) + embs["q_emb"].size(0)

    ranked_1_hop = scores_1.argsort(dim=1, descending=True)
    ranked_2_hop = scores_2.argsort(dim=1, descending=True)
    idx2ranked_1 = ranked_1_hop.argsort(dim=1)
    idx2ranked_2 = ranked_2_hop.argsort(dim=1)
    
    rrs_1, rrs_2 = [], []
    for t, idx2ranked in zip(target_1, idx2ranked_1):
        rrs_1.append(1 / (idx2ranked[t].item() + 1))

    for t, idx2ranked in zip(target_2, idx2ranked_2):
        rrs_2.append(1 / (idx2ranked[t].item() + 1))
    
    return {"rrs_1": rrs_1, "rrs_2": rrs_2}
