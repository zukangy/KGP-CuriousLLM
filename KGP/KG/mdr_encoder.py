
import torch.nn as nn
from transformers import AutoModel
import torch


class Retriever_inf(nn.Module):
    def __init__(self, config, base_model="deepset/tinyroberta-squad2"):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(base_model)
        self.project = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), 
                                     nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps))
    
    def encode_seq(self, input_ids, mask):
        cls_rep = self.encoder(input_ids, mask)[0][:, 0, :]
        vector = self.project(cls_rep)
        return vector

    def forward(self, input_ids, attention_mask):
        emb = self.encode_seq(input_ids, attention_mask)
        return emb


@torch.no_grad()
def emb_pipeline(d, model, tokenizer, args):
    model.eval()
    
    titles = [title for title, _ in d['title_chunks']]
    chunks = [chunk for _, chunk in d['title_chunks']]

    chunks_encode = tokenizer(text = titles, text_pair = chunks, max_length = args.max_len, return_tensors = 'pt', padding=True, truncation=True)

    for key in chunks_encode:
        chunks_encode[key] = chunks_encode[key].to(args.device)

    c_emb = model(chunks_encode['input_ids'], chunks_encode['attention_mask'])

    return c_emb.cpu().numpy()