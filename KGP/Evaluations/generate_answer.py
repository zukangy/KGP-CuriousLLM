import json 
from tqdm import tqdm 
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_json(file_path: str):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def pipeline(data, model, tokenizer, device="cpu"):
    prompt = """Given the following question and contexts, create a final answer in English to the question. 
    QUESTION: {question}
    CONTEXT: {context}
    ANSWER: [Please provide only the answer and keep the answer less than 6 words.]
    """
    
    none_prompt = """Given the following question and contexts, create a final answer in English to the question.
    QUESTION: {question}
    ANSWER: [Please provide only the answer and keep the answer less than 6 words.]
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
            
    responses = []
    
    for record in tqdm(data, total=len(data)):
        type = record['type']
        question = record['question']
        contexts = record['evidence']
        gt = record['answer']
        
        if contexts:
            contexts = '\n'.join(f'{i}: {c}' for i, c in enumerate(contexts, start=1))
            input_prompt = prompt.format(question=question, context=contexts)
        else:
            input_prompt = none_prompt.format(question=question)
        
        messages = [
            {"role": "user", "content": input_prompt},
            {"role": "assistant", "content": "ANSWER: "}
        ]
        
        model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
        
        generated_ids = model.generate(model_inputs, pad_token_id=tokenizer.pad_token_id,
                                       max_new_tokens=25, do_sample=True, temperature=0.5)
        decoded = tokenizer.batch_decode(generated_ids)
        
        resp = decoded[0].split("[/INST]")[-1].strip()
        if '</s>' in resp:
            resp = ''.join(resp.split('</s>')[1:]).replace('</s>', '').strip()
            
        print(resp)
        
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
    
    data_path = "./DATA/KG/evidence_100/mistral_agent/evidence.json"
    data = load_json(data_path)
    
    model = model.to(device)
    
    responses = pipeline(data, model, tokenizer, device=device)
    
    with open("./DATA/KG/answers/mistral_agent_responses.json", 'w') as f:
        json.dump(responses, f, indent=4)
