import json 
from tqdm import tqdm 
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_json(file_path: str):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def pipeline(data, model, tokenizer, save_path, device="cpu"):
    # prompt = """Given the following question and contexts, create a final answer in English to the question. 
    # QUESTION: {question}
    # CONTEXT: {context}
    # ANSWER: [Please provide only the answer and keep the answer less than 6 words.]
    # """
    
    prompt = """
    Given the question and its associated contexts below, please generate a concise, precise answer in English. The answer must strictly adhere to the following guidelines:

    - The answer should be directly relevant to the question.
    - Provide the answer in a clear, straightforward format.
    - Limit your answer to no more than 6 words, focusing on the essential information requested.
    - If the provided contexts do not contain enough information to answer the question, respond with "Information not available".
    - Do not include any additional tokens, explanations, or information beyond the direct answer.

    QUESTION: {question}
    CONTEXT: {context}
    ANSWER: [Your concise answer here or "Information not available" if the answer cannot be determined from the contexts.]

    """
    
    none_prompt = """Given the following question, create a final answer in English to the question.
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
                                       max_new_tokens=30, do_sample=True, temperature=0.1)
        decoded = tokenizer.batch_decode(generated_ids)
        
        resp = decoded[0].split("[/INST]")[-1].strip()
        if '</s>' in resp:
            resp = ''.join(resp.split('</s>')[1:]).replace('</s>', '').strip()
        
        response = {
            "type": type,
            "question": question,
            "gt": gt,
            "response": resp
        }
        responses.append(response)
        
        with open(save_path, 'w') as f:
            json.dump(responses, f, indent=4)
            
    return
    

if __name__ == '__main__':    
    device = "mps" 
    
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    
    data = "./DATA/KG/evidence/wiki_evidence_100/mdr_agent/evidence.json"
    save_path = "./DATA/KG/answers/wiki_answers_mistral/mdr_agent_responses.json"
    data = load_json(data)
    
    model = model.to(device)
    
    responses = pipeline(data, model, tokenizer, save_path=save_path, device=device)
