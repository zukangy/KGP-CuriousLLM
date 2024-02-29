from transformers import AutoConfig, AutoTokenizer


def load_tokenizer(model_name='bert-base-uncased'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    return tokenizer, config