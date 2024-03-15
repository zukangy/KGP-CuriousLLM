import torch.nn as nn
from transformers import AutoModel
import torch


class Retriever(nn.Module):
    def __init__(self, config, args):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(args['model_name'])
        self.args = args 
        
        # Freeze some layers if specified
        self.freeze_encoder()
        
        self.project = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), 
                                     nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps))

    def freeze_encoder(self):
        # Freeze some layers if specified
        if self.args['freeze_layers'] > 0:  
            model_config = self.encoder.config
            if self.args['freeze_layers'] >= model_config.num_hidden_layers:
                # Freeze all layers
                for param in self.encoder.parameters():
                    param.requires_grad = False      
            else:
                # Freeze the parameters of the top 'freeze_layers' layers
                for name, param in self.encoder.named_parameters():
                    # Layers to be frozen are prefixed with 'encoder.layer' followed by the layer index
                    layer_index = name.split('.')[2]
                    if name.startswith('encoder.layer') and int(layer_index) < self.args['freeze_layers']:
                        param.requires_grad = False

    def encode_seq(self, input_ids, mask):
        cls_rep = self.encoder(input_ids, mask)[0][:, 0, :] # Extract the CLS token representation
        vector = self.project(cls_rep)
        return vector

    def forward(self, batch):
        q_emb = self.encode_seq(batch['q_enc_btz'], batch['q_mask'])
        q_c1_emb = self.encode_seq(batch['q_c1_enc_btz'], batch['q_c1_mask'])

        c1_emb = self.encode_seq(batch['c1_enc_btz'], batch['c1_mask'])
        c2_emb = self.encode_seq(batch['c2_enc_btz'], batch['c2_mask'])

        n1_emb = self.encode_seq(batch['n1_enc_btz'], batch['n1_mask'])
        n2_emb = self.encode_seq(batch['n2_enc_btz'], batch['n2_mask'])

        return {'q_emb': q_emb, 'q_c1_emb': q_c1_emb, \
                "c1_emb": c1_emb, 'c2_emb': c2_emb, \
                "n1_emb": n1_emb, 'n2_emb': n2_emb}


class Retriever_inf(nn.Module):
    def __init__(self, config, args):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(args['model_name'])
        self.project = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), 
                                     nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps))
    
    def encode_seq(self, input_ids, mask):
        cls_rep = self.encoder(input_ids, mask)[0][:, 0, :]
        vector = self.project(cls_rep)
        return vector

    def forward(self, input_ids, attention_mask):
        #split batch into two pieces
        emb_1 = self.encode_seq(input_ids[:input_ids.shape[0]//4], attention_mask[:input_ids.shape[0]//4])
        emb_2 = self.encode_seq(input_ids[input_ids.shape[0]//4:input_ids.shape[0]//2], attention_mask[input_ids.shape[0]//4:input_ids.shape[0]//2])
        emb_3 = self.encode_seq(input_ids[input_ids.shape[0]//2:input_ids.shape[0]//4*3], attention_mask[input_ids.shape[0]//2:input_ids.shape[0]//4*3])
        emb_4 = self.encode_seq(input_ids[input_ids.shape[0]//4*3:], attention_mask[input_ids.shape[0]//4*3:])

        return torch.cat([emb_1, emb_2, emb_3, emb_4], axis = 0)