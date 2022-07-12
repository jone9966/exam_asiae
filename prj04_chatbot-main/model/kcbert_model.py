import math

import torch
import torch.nn as nn
from transformers import BertModel
from sklearn.metrics.pairwise import cosine_similarity

class BERTClassifier(nn.Module):
    def __init__(self, hidden_size=1024, num_classes=2, dr_rate=0.5, device=torch.device("cpu")):
        super(BERTClassifier, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.bert = BertModel.from_pretrained("beomi/kcbert-large", return_dict=False)
        self.dropout = nn.Dropout(p=dr_rate)
        self.fuck_classifier = nn.Linear(hidden_size, 2)
        self.badword_classifier = nn.Linear(hidden_size, 3)

    def forward(self, input_ids, token_type_ids, attention_mask):
        input_ids = input_ids.long().to(self.device)
        token_type_ids = token_type_ids.long().to(self.device)
        attention_mask = attention_mask.long().to(self.device)
        _, pooler, hidden_states = self.bert(input_ids=input_ids, token_type_ids=token_type_ids,
                                             attention_mask=attention_mask, output_hidden_states=True)
        out = self.dropout(pooler)

        cls = pooler[0]
        word_embedding = hidden_states[0][0, 1:-1]

        attention_scores = torch.matmul(cls, word_embedding.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.hidden_size)
        attention_prob = nn.Softmax(dim=-1)(attention_scores)

        cls_pooler = cls.detach().cpu().numpy().reshape(1, -1)
        embedding = word_embedding.detach().cpu().numpy()
        cosine_sim = cosine_similarity(cls_pooler, embedding)

        return self.fuck_classifier(out), self.badword_classifier(out), (attention_prob, cosine_sim)