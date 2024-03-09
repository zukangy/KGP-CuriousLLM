import os 
from tqdm import tqdm 
import json
from openai import OpenAI
import time


client=OpenAI()


def read_json(file_path: str):
  with open (file_path, 'r') as f:
      data = json.load(f)
  return data


def parse_json_record(record: dict):
  """ Parse a single record from the JSON file.
  record: dict, a single record from the JSON file with keys: 'instruction', 'input', 'output'.
  
  return tuple: (question, evidence, output)
  """
  try:
      _, input, output = record['instruction'], record['input'], record['output']
  except KeyError:
      print(f"Record {record} is missing a required field")
      return
  
  question, evidence = input.split('\n')[0], input.split('\n')[1]
  return question, evidence, output
  
  
def parse_data(data_path):
  """ Parse the JSON file and return a list of records. Each record is indexed by its position in the list.
  data_path: str, path to the JSON file.
  
  return list: a list of records, each record is a dict with keys: 'id', 'question', 'evidence', 'output'.
  """
  data = read_json(data_path)
  record_collection = []
  print("Starting to parse data")
  for i, record in tqdm(enumerate(data)):
      question, evidence, output = parse_json_record(record)
      record_collection.append({'id': i, 'question': question, 'evidence': evidence, 'output': output})
  print(f"Parsed {len(record_collection)} records")
  return record_collection


def generate_question(primary_question, evidence):
  """Generate a follow-up question based on the primary question and evidence.
  primary_question: str, the primary question.
  evidence: str, the evidence for the primary question.
  
  return str: the follow-up question.
  """
  with open('./KGP/LLMs/GPT/instruction_prompt.txt', 'r') as f:
    prompt = f.read()
    
  prompt = prompt.format(primary_question=primary_question, evidence=evidence)
  
  completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": "You are a critical thinker."},
      {"role": "user", "content": prompt}
    ]
  )
  return completion.choices[0].message.content


def pipeline(root_dir='.', data_file='reason_instruction.json', limit=10000, 
             create_new=True, checkpoint=None, cp_every=50):
  """Pipeline to generate follow-up questions for a given dataset.

  Args:
      root_dir (str, optional): main directory. Defaults to '.'.
      data_file (str, optional): path to data. Defaults to 'reason_instruction.json'.
      limit (int, optional): number of records to generate follow-up questions, from 0th to (limit - 1)-th position. Defaults to 10000.
      create_new (bool, optional): If True, generating questions from scratch. Defaults to True.
      checkpoint (_type_, optional): If specified, create_new has to set to True. Defaults to None.
      cp_every (int, optional): Number of records generated before a checkpoint. Defaults to 50.
  """
  # Limit: Int, each 10k inferences translate to $2.5 using GPT3.5-turbo.
  
  # Default dataset is reason_instruction.json
  data = parse_data(os.path.join(root_dir, data_file))

  os.makedirs(os.path.join(root_dir, 'DATA/T5_traversal_agent'), exist_ok=True)
  
  if create_new:
    dataset = []
  else:
    if checkpoint is None:
      raise ValueError('Checkpoint path is required to load existing dataset.')
    else:
      with open(os.path.join(checkpoint), 'r') as f:
        dataset = json.load(f)
      print(f"Loaded {len(dataset)} records from checkpoint")
    
  existing_ids = [record['id'] for record in dataset]

  for record in tqdm(data[:limit], total=limit):
    # Skip if the record already exists in the dataset
    if record['id'] in existing_ids:
      continue
    
    question = generate_question(record['question'], record['evidence'])
    new_record = {'id': record['id'], 
                  'question': record['question'], 
                  'evidence': record['evidence'], 
                  'output': record['output'], 
                  'follow_up_question': question}
    dataset.append(new_record)
    time.sleep(.1)
    
    # Save checkpoint every cp_every records
    if record['id'] % cp_every == 0:
        with open(os.path.join(root_dir, 'DATA/T5_traversal_agent', 'cp_gpt_question_instruction.json'), 'w') as f:
            json.dump(dataset, f, indent=4)
        
        time.sleep(1)
                
  # Save the dataset once all records are processed
  with open(os.path.join(root_dir, 'DATA/T5_traversal_agent', 'gpt_question_instruction.json'), 'w') as f:
      json.dump(dataset, f, indent=4)