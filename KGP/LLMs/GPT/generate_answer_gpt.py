import json 
from tqdm import tqdm 

from openai import OpenAI


client = OpenAI()


def load_json(file_path: str):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def pipeline(data, save_path):
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
    - If the provided contexts do not contain enough information to answer the question, respond with "Information not available" or "Cannot determine from provided context."
    - Do not include any additional tokens, explanations, or information beyond the direct answer.
    - Carefully reason through the question and contexts if the question involes time. 

    QUESTION: {question}
    CONTEXT: {context}
    ANSWER: [Your concise answer here or "Information not available" if the answer cannot be determined from the contexts.]

    """
    
    none_prompt = """Given the following question and contexts, create a final answer in English to the question.
    QUESTION: {question}
    ANSWER: [Please provide only the answer and keep the answer less than 6 words.]
    """
    
            
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
        
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
            {"role": "system", "content": "You are a QA generation assistant."},
            {"role": "user", "content": input_prompt}
            ]
        )
        
        resp = completion.choices[0].message.content            
        
        response = {
            "type": type,
            "question": question,
            "gt": gt,
            "response": resp
        }
        responses.append(response)
        
        with open(save_path, 'w') as f:
            json.dump(responses, f, indent=4)
        
    return responses
    

if __name__ == '__main__':    
    data_path = "./DATA/KG/evidence/wiki_evidence_100/mdr_title_agent/evidence.json"
    save_path = "./DATA/KG/answers/wiki_answers_gpt/mdr_title_agent_responses.json"
    data = load_json(data_path)
    
    responses = pipeline(data, save_path)
