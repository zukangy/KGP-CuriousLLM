import json 
import random 


question_dataset_path = "DATA/2WikiMQA/test_docs.json"

with open(question_dataset_path, 'r') as f:
    dataset = json.load(f)

random.seed(1028)
random.shuffle(dataset)   
    
questions = [{i: d['question']} for i, d in enumerate(dataset)]

with open("./indexed_test_docs.json", 'w') as f:
    json.dump(dataset, f, indent=4)
    
with open("./indexed_2wiki_test_questions.json", 'w') as f:
    json.dump(questions, f, indent=4)