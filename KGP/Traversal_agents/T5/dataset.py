import pandas as pd
from torch.utils.data import Dataset
import torch
import json

def load_dataset(args):
    data = json.load(open(f"{args['root_dir']}/{args['dataset']['data_dir']}", 'r'))
    df = pd.DataFrame.from_records(data)

    # df = df[['input', 'output']]

    train_size = args['dataset']['train_size']
    seed = args['random_seed']
    
    if train_size > 1.:
        raise ValueError("train_size should be less than 1")
    
    train_data = df.sample(frac=train_size, random_state=seed)
    val_data = df.drop(train_data.index).reset_index(drop=True)
    train_data = train_data.reset_index(drop=True)
    
    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")

    return train_data, val_data


class EvidencePromptDataset(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, dataframe, tokenizer, source_len, target_len):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = target_len
        self.start_prompt = self.data['instruction']
        self.end_prompt = '\n\nEvidence: '
        self.target_text = self.data['input']
        self.source_text = self.data['output']
        
    def tokenize_func(self, start_prompt, input, output):
        prompt = [start_prompt + input + self.end_prompt]
        input_ids = self.tokenizer(prompt, 
                                   max_length=self.source_len,
                                   padding="max_length", 
                                   truncation=True, 
                                   return_tensors="pt").input_ids
        
        labels = self.tokenizer(output,
                                max_length=self.summ_len,
                                padding="max_length", 
                                truncation=True, 
                                return_tensors="pt").input_ids
        
        input_ids = input_ids.squeeze(0)
        labels = labels.squeeze(0)
        
        return {
            "input_ids": input_ids.to(dtype=torch.long),
            "labels": labels.to(dtype=torch.long)
        }

    def __len__(self):
        """returns the length of dataframe"""
        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""
        
        start_prompt = str(self.start_prompt[index])
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        start_prompt = " ".join(start_prompt.split())
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())
        
        start_prompt = start_prompt + '\n\n'

        output = self.tokenize_func(start_prompt=start_prompt, 
                                  input=source_text, 
                                  output=target_text)
        # input_shape = output['input_ids'].size()
        return output




class EvidencePromptDatasetInf(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(self, source_text, tokenizer, source_len):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.source_len = source_len
        self.source_text = source_text

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.source_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        
        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            'source_text': source_text
        }