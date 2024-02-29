from transformers import T5ForConditionalGeneration, AutoTokenizer


def create_flan_t5_base_model(model_name):
    try:
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        return model, tokenizer
    except OSError:
        raise OSError("Model not found. Please check the model name and try again.")