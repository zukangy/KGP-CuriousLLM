import os 
from tqdm import tqdm 
import json
from openai import OpenAI


client = OpenAI()

def generate_reasoning_question(primary_question, evidence):
  """Generate a follow-up question based on the primary question and evidence."""
  
  with open('./KGP/Traversal_agents/mistral_7b/instruction_prompt.txt', 'r') as f:
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


def pipeline(root_dir='.', data_dir='DATA/T5_traversal_agent/', limit=10000, 
             create_new=True, checkpoint=None):
  """Limit: Int, each 10k inferences translate to $2.5 using GPT3.5-turbo."""
  with open(os.path.join(root_dir, data_dir, 'parsed_reason_instruction.json'), 'r') as f:
    data = json.load(f)
    
  os.makedirs(os.path.join(root_dir, data_dir), exist_ok=True)
  
  if create_new:
    dataset = []

    for record in tqdm(data[:limit]):
        question = generate_reasoning_question(record['question'], record['evidence'])
        new_record = {'id': record['id'], 
                      'question': record['question'], 
                      'evidence': record['evidence'], 
                      'output': record['output'], 
                      'follow_up_question': question}
        dataset.append(new_record)
        
        # Save checkpoint every 50 records
        if record['id'] % 50 == 0:
            with open(os.path.join(root_dir, data_dir, 'cp_gpt_question_instruction_dataset.json'), 'w') as f:
                json.dump(dataset, f, indent=4)
  else:
    if checkpoint is None:
      raise ValueError('Checkpoint path is required to load existing dataset.')
    else:
      with open(os.path.join(root_dir, checkpoint), 'r') as f:
        dataset = json.load(f)
      
      existing_ids = [record['id'] for record in dataset]
      
      for record in tqdm(data[:limit]):
        if record['id'] in existing_ids:
            continue
        else:
            question = generate_reasoning_question(record['question'], record['evidence'])
            new_record = {'id': record['id'], 
                          'question': record['question'], 
                          'evidence': record['evidence'], 
                          'output': record['output'], 
                          'follow_up_question': question}
            dataset.append(new_record)
            
            # Save checkpoint every 50 records
            if record['id'] % 50 == 0:
                with open(os.path.join(root_dir, checkpoint), 'w') as f:
                    json.dump(dataset, f, indent=4)
  
  # Save the dataset
  with open(os.path.join(root_dir, data_dir, 'gpt_question_instruction.json'), 'w') as f:
      json.dump(dataset, f, indent=4)