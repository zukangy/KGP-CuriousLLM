from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments


def generate_lora_config(config_args):
    return LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, 
                      bias="none",
                      **config_args)
    
    
def create_peft_model(orig_model, lora_config):
    lora_config = generate_lora_config(config_args=lora_config)
    return get_peft_model(model=orig_model, 
                          peft_config=lora_config)
    

def create_training_args(training_args):
    return TrainingArguments(auto_find_batch_size=True,
                             logging_first_step=True,
                             use_cpu=False,
                             **training_args)
                   