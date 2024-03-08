# Copyright © 2023 Apple Inc.

from pathlib import Path
import os 
import json 
from sklearn.model_selection import train_test_split


class Dataset:
    """
    Light-weight wrapper to hold lines from a jsonl file
    """

    def __init__(self, path: Path, key: str = "text"):
        if not path.exists():
            self._data = None
        else:
            with open(path, "r") as f:
                self._data = json.load(f)
        self._key = key

    def __getitem__(self, idx: int):
        return self._data[idx][self._key]

    def __len__(self):
        return len(self._data)
    
    
def extract_records(data):
    records = []
    for record in data:
        question = record['question'].strip()
        evidence = record['evidence'].strip()
        output = record['follow_up_question'].strip()
        
        input_string = f"Question: {question} \n Evidence: {evidence} \n Follow-up Question: '{output}'"
        records.append({'text': input_string})
    return records
    
    
def split_raw_dataset(raw_data_path, root_dir='.', valid_perc=0.1, test_perc=0.2, seed=1028):
    with open(raw_data_path, 'r') as f:
        raw_dataset = json.load(f)
    
    train, test = train_test_split(raw_dataset, test_size=valid_perc + test_perc, random_state=seed)
    valid, test = train_test_split(test, test_size=test_perc / (valid_perc + test_perc), random_state=seed)
    
    os.makedirs(f'{root_dir}/mistral', exist_ok=True)
    
    for name, data in zip(['train', 'valid', 'test'], [train, valid, test]):
        records = extract_records(data)
        with open(f'{root_dir}/mistral/{name}.jsonl', 'w') as f:
            json.dump(records, f, indent=4)
        
        
if __name__ == '__main__':
    raw_data_path = './cp_gpt_question_instruction.json'
    split_raw_dataset(raw_data_path)