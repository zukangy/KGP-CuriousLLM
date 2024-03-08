import os 
import yaml 
import torch 
from transformers import Trainer
from peft import PeftModel

from KGP.Traversal_agents.T5.dataset import load_dataset, EvidencePromptDataset
from KGP.Traversal_agents.T5.utils import seed_everything, print_number_of_trainable_model_parameters
from KGP.Traversal_agents.T5.flan_t5 import create_flan_t5_base_model
from KGP.Traversal_agents.T5.lora import create_peft_model, create_training_args


def run(train_data, val_data, args):
    # model and tokenizer
    model_name = args['model']['model_name']
    base_model, tokenizer = create_flan_t5_base_model(model_name)
    
    if args['model']['from_local_checkpoint'] is None:            
        # Initialize the peft model
        print('Creating new model from scratch... ')
        peft_model = create_peft_model(orig_model=base_model, lora_config=args['lora']['lora_params'])
    else:
        try:
            print('Loading from local checkpoint... ')
            peft_model = PeftModel.from_pretrained(model=base_model,
                                                model_id=args['model']['from_local_checkpoint'],
                                                torch_dtype=torch.float32,
                                                is_trainable=True)
        except ValueError:
            raise ValueError("Provided checkpoint does not exist...")
        
    print_number_of_trainable_model_parameters(peft_model)
    
    # Create the datasets
    train_set = EvidencePromptDataset(
        train_data,
        tokenizer,
        args['dataset']['max_source_text_len'],
        args['dataset']['max_target_text_len'],
    )

    val_set = EvidencePromptDataset(
        val_data,
        tokenizer,
        args['dataset']['max_source_text_len'],
        args['dataset']['max_target_text_len'],
    )
    
    checkpoint_dir = os.path.join(args['root_dir'], args['checkpoint_dir'])
    peft_model_path = os.path.join(checkpoint_dir, args['lora']['save_model_path'])
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(peft_model_path, exist_ok=True)
    
    peft_training_args = create_training_args(args['lora']['lora_training_params'])
    peft_training_args.output_dir = os.path.join(checkpoint_dir, peft_training_args.output_dir)

    peft_trainer = Trainer(
        model=peft_model,
        args=peft_training_args,
        train_dataset=train_set,
        eval_dataset=val_set)
    
    if args['model']['from_local_checkpoint'] is None:
        peft_trainer.train()
    else:
        print('Resuming training from local checkpoint... ')
        peft_trainer.train(resume_from_checkpoint=args['model']['from_local_checkpoint'])
    
    peft_trainer.model.save_pretrained(peft_model_path)
    tokenizer.save_pretrained(peft_model_path)

if __name__ == "__main__":
    args = yaml.safe_load(open('./t5_agent_config.yml', 'r'))
    
    train_data, val_data = load_dataset(args)
    
    seed_everything(args['random_seed'])
    
    run(train_data, val_data, args)