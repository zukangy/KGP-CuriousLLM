import os 
import json 
from tqdm import tqdm


def read_json(file_path):
    with open (file_path, 'r') as f:
        data = json.load(f)
    return data


def parse_json_record(record):
    try:
        _, input, output = record['instruction'], record['input'], record['output']
    except KeyError:
        print(f"Record {record} is missing a required field")
        return
    
    question, evidence = input.split('\n')[0], input.split('\n')[1]
    return question, evidence, output

        
def process_data(data_path, output_root_dir='.', output_file='question_instruction.json'):
    data = read_json(data_path)
    record_collection = []
    for i, record in tqdm(enumerate(data)):
        question, evidence, output = parse_json_record(record)
        record_collection.append({'id': i, 'question': question, 'evidence': evidence, 'output': output})
    
    if record_collection:
        save_dir = os.path.join(output_root_dir, 'DATA/T5_traversal_agent')
        save_path = os.path.join(save_dir, output_file)
        os.makedirs(save_dir, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(record_collection, f, indent=4)
    
    print(f"Processed {len(record_collection)} records")
    return 
    

if __name__ == "__main__":
    data_path = "./DATA/T5_traversal_agent/reason_instruction.json"
    output_root_dir = "."
    output_file = "parsed_reason_instruction.json"
    process_data(data_path, output_root_dir, output_file)


