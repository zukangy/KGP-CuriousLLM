import os 

from KGP.MDR.tokenizer import load_tokenizer
from KGP.MDR.utils import seed_everything
from KGP.MDR.model import Retriever
from KGP.MDR.train import train
from KGP.MDR.dataset import load_dataset, Dataset_collate
from KGP.LLMs.Mistral.quantize_mistral_mlx import load_config


os.environ["TOKENIZERS_PARALLELISM"] = 'false'


if __name__=="__main__":
    args = load_config('configs/MDR.yml')
    
    train_data, val_data = load_dataset(train_percent=args['train_percent'], seed=args['seed'])
    
    seed_everything(args['seed'])
    
    tokenizer, config = load_tokenizer(args['model_name'])
    
    checkpoint_dir = args['from_checkpoint']

    model = Retriever(config, args)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The model has {num_params} parameters.")
    
    train(train_data, val_data, model, tokenizer, Dataset_collate, args, checkpoint_dir)