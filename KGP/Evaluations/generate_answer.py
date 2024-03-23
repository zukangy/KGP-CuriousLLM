import json 
from tqdm import tqdm 
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_json(file_path: str):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def pipeline(data, model, tokenizer, device="cpu"):
    prompt = """Given the following question and contexts, create a final answer to the question. 
    QUESTION: {question}
    CONTEXT: {context}
    ANSWER: Please provide only the answer and keep the answer less than 6 words. 
    """
    responses = []
    
    for record in tqdm(data, total=len(data)):
        type = record['type']
        question = record['question']
        contexts = record['found_evidence']
        gt = record['answer']
        
        contexts = '\n'.join(f'{i}: {c}' for i, c in enumerate(contexts, start=1))
        
        input_prompt = prompt.format(question=question, context=contexts)
        
        messages = [
            {"role": "user", "content": input_prompt},
            {"role": "assistant", "content": "ANSWER: Please keep the answer less than 6 words. "}
        ]
        
        model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
        
        generated_ids = model.generate(model_inputs, max_new_tokens=20, do_sample=True)
        decoded = tokenizer.batch_decode(generated_ids)
        
        resp = decoded[0].split("[/INST]")[-1].strip()
        
        response = {
            "type": type,
            "question": question,
            "gt": gt,
            "response": resp
        }
        responses.append(response)
    return responses
    

if __name__ == '__main__':    
    device = "mps" 
    
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    
    data_path = "./DATA/KG/evidence/checkpoint_mdr_agent/evidence.json"
    data = load_json(data_path)
    
    model = model.to(device)
    
    responses = pipeline(data, model, tokenizer, device=device)
    
    with open("./DATA/KG/answers/mdr_responses.json", 'w') as f:
        json.dump(responses, f, indent=4)
