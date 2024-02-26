import json 
import random 


def load_dataset(dataset='HotpotQA', train_percent=1.0, seed=42):
    if dataset == 'HotpotQA':
        with open('./DATA/HotpotQA/MDR/train_with_neg_v0.json', 'r') as file:
            train_data = [json.loads(line) for line in file if len(json.loads(line)['neg_paras']) >= 2]
            
        with open('./DATA/HotpotQA/MDR/val_with_neg_v0.json', 'r') as file:
            val_data = [json.loads(line) for line in file]
        
    # TODO: add MuSiQue
    elif dataset == 'MuSiQue':
        pass 
    
    print('Loaded {} train and {} val data'.format(len(train_data), len(val_data)))
    
    random.seed(seed)
    if train_percent < 1.0:
        random.shuffle(train_data)
        train_data = random.sample(train_data, int(train_percent * len(train_data)))
        print('Sampled {} train data'.format(len(train_data)))
        
    if train_percent >= 1.0:
        print("Using all the training data")
    return train_data, val_data