import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class SimplE(torch.nn.Module):
    def __init__(self, params):
        super(SimplE, self).__init__()
        self.hidden_dim = params.hidden_dim
        self.device = params.device
        self.n_ent = params.contents.num_entities
        
        self.ent_embs_h = nn.Embedding(self.n_ent, self.hidden_dim)
        self.ent_embs_t = nn.Embedding(self.n_ent, self.hidden_dim)
        self.rel_embs_f = nn.Embedding(self.n_ent, self.hidden_dim)
        self.rel_embs_i = nn.Embedding(self.n_ent, self.hidden_dim)
        
        nn.init.xavier_uniform_(self.ent_embs_h.weight)
        nn.init.xavier_uniform_(self.ent_embs_t.weight)
        nn.init.xavier_uniform_(self.rel_embs_f.weight)
        nn.init.xavier_uniform_(self.rel_embs_i.weight)
    

    def getEmbeddings(self, X):
        heads = torch.from_numpy(X.src_idx).to(self.device)
        rels = torch.from_numpy(X.rel_idx).to(self.device)
        tails = torch.arange(self.n_ent, device=self.device)

        h_embs1 = self.ent_embs_h(heads).unsqueeze(1)
        r_embs1 = self.rel_embs_f(rels).unsqueeze(1)
        t_embs1 = self.ent_embs_t(tails).unsqueeze(0)
        h_embs2 = self.ent_embs_h(tails).unsqueeze(0)
        r_embs2 = self.rel_embs_i(rels).unsqueeze(1)
        t_embs2 = self.ent_embs_t(heads).unsqueeze(1)
                
        return h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2
    
    def forward(self, X):
        batch_size = X.rel_idx.shape[0]
        h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2 = self.getEmbeddings(X)
        # print(h_embs1.shape, r_embs1.shape, t_embs1.shape)
        scores = ((h_embs1 * r_embs1) * t_embs1 + (h_embs2 * r_embs2) * t_embs2) / 2.0
        # scores = F.dropout(scores, p=self.params.dropout, training=self.training)
        scores = torch.sum(scores, dim=2)
        # print(scores.shape)
        attention_score = F.softmax(scores, dim=1).view(-1)
        cur_entity = torch.zeros(len(attention_score), 2, device=self.device)
        for i in range(batch_size):
            cur_entity[i*self.n_ent:(i+1)*self.n_ent, 0] = i
            cur_entity[i*self.n_ent:(i+1)*self.n_ent, 1] = torch.arange(self.n_ent, device=self.device) 
        return scores, (attention_score, cur_entity.cpu().numpy())
        
