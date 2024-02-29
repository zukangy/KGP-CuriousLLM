from KGP.Traversal_agents.GPT.gpt_get_question import pipeline as gpt_pipeline


if __name__ == "__main__":
    gpt_pipeline(data_dir='DATA/T5_traversal_agent/', 
                 limit=20000, 
                 create_new=False,
                 checkpoint='DATA/T5_traversal_agent/cp_gpt_question_instruction.json')