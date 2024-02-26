import os 
import yaml 

import math
import numpy as np
from tqdm import tqdm
from datetime import datetime
import torch 
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import get_linear_schedule_with_warmup

from KGP.MDR.dataset import load_dataset
from KGP.MDR.tokenizer import load_tokenizer
from KG_LLM_MDQA.MDR.utils import seed_everything
from KG_LLM_MDQA.MDR.learn import train, eval, mp_loss
from KG_LLM_MDQA.MDR.utils import move_to_cuda
from KG_LLM_MDQA.MDR.model import Retriever
from KG_LLM_MDQA.MDR.loader import Dataset_process, Dataset_collate, Dataset_process2


os.environ["TOKENIZERS_PARALLELISM"] = 'false'

# T0D0: Add logging
def run(train_data, val_data, model, tokenizer, collate, args, checkpoint_dir=None):
    if args['do_train']:
        if args['dataset'] == 'HotpotQA':
            train_dataset = Dataset_process(train_data, tokenizer, args, train=True)
        elif args['dataset'] == 'MuSiQue':
            train_dataset = Dataset_process2(train_data, tokenizer, args, train=True)

        train_dataloader = DataLoader(train_dataset, batch_size=args['train_bsz'], pin_memory=True, 
                                      collate_fn=collate, num_workers=args['num_workers'], shuffle=True)

        if args['dataset'] == 'HotpotQA':
            val_dataset = Dataset_process(val_data, tokenizer, args, train=False)
        elif args['dataset'] == 'MuSiQue':
            val_dataset = Dataset_process2(val_data, tokenizer, args, train=False)

        val_dataloader = DataLoader(val_dataset, batch_size=args['eval_bsz'], pin_memory=True, 
                                    collate_fn=collate, num_workers=args['num_workers'], shuffle=False)
        
        t_total = len(train_dataloader) * args['epochs']
        warmup_steps = math.ceil(t_total * args['warm_ratio'])

        print("Start Training")
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Checkpoint the model at the end of each epoch
        os.makedirs(f"./{args['checkpoint_root_dir']}", exist_ok=True)
        os.makedirs(f"./{args['checkpoint_root_dir']}/MDR", exist_ok=True)
        os.makedirs(f"./{args['checkpoint_root_dir']}/MDR/{args['dataset']}", exist_ok=True)
        os.makedirs(f"./{args['checkpoint_root_dir']}/MDR/{args['dataset']}/{args['checkpoint_id']}", exist_ok=True)
        os.makedirs(f"./{args['checkpoint_root_dir']}/MDR/{args['dataset']}/{args['checkpoint_id']}/{timestamp}", exist_ok=True)
        
        # Save the config info 
        with open(f"./{args['checkpoint_root_dir']}/MDR/{args['dataset']}/{args['checkpoint_id']}/{timestamp}/mdr_config.yml", 'w') as file:
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
                
                samples = move_to_cuda(samples, device=args['gpus'])
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
                    
                    model_paths = [f"./{args['checkpoint_root_dir']}/MDR/{args['dataset']}/{args['checkpoint_id']}/{timestamp}/model.pt"]
                    
                    if mrr_avg > best_mrr:
                        best_mrr = mrr_avg
                        
                        print('Saving best model...')
                        model_paths.append(f"./{args['checkpoint_root_dir']}/MDR/{args['dataset']}/{args['checkpoint_id']}/{timestamp}/best_model.pt")
                    
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
        if args.dataset == 'HotpotQA':
            val_dataset = Dataset_process(val_data, tokenizer, args, train = False)
        elif args.dataset == 'MuSiQue':
            val_dataset = Dataset_process2(val_data, tokenizer, args, train = False)
        val_dataloader = DataLoader(val_dataset, batch_size=args.eval_bsz, pin_memory=True, 
                                    collate_fn=collate, num_workers=args['num_workers'], shuffle=False)
        
        mrr_1, mrr_2 = eval(model, val_dataloader, args)
        mrr_avg = (mrr_1 + mrr_2) / 2

        print("MRR_1: {}, MRR_2: {}, Ave_MRR: {}".format(mrr_1, mrr_2, (mrr_1 + mrr_2) / 2))


if __name__=="__main__":
    args = yaml.safe_load(open('mdr_tinyroberta-squad2.yml', 'r'))
    
    train_data, val_data = load_dataset(dataset=args['dataset'], train_percent=args['train_percent'], 
                                        seed=args['seed'])
    
    seed_everything(args['seed'])
    
    tokenizer, config = load_tokenizer(args['model_name'])
    
    checkpoint_dir = args['from_checkpoint']

    model = Retriever(config, args)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The model has {num_params} parameters.")
    
    run(train_data, val_data, model, tokenizer, Dataset_collate, args, checkpoint_dir)