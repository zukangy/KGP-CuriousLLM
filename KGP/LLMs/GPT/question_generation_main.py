# This script is used to generate follow-up questions for a given dataset.
# This new dataset is used to fine-tune the mistral 7B model as a traversal agent.

from KGP.LLMs.GPT.gpt_get_question import pipeline as gpt_pipeline


if __name__ == "__main__":
    gpt_pipeline(data_file='./DATA/T5_traversal_agent/reason_instruction.json', 
                 limit=50000, 
                 create_new=False,
                 checkpoint=None,
                 cp_every=20)