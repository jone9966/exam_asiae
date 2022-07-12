import re
import math
import torch
import torch.nn as nn
import numpy as np

from model.kcbert_model import KcBERTClassifier
from transformers import BertTokenizer


ctx = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(ctx)
tokenizer = BertTokenizer.from_pretrained("beomi/kcbert-large")
model = torch.load('../output/model/kcbert_multi.pt')
model.to(device)


def softmax(vals, idx):
    valscpu = vals.cpu().detach().squeeze(0)
    a = 0
    for i in valscpu:
        a += np.exp(i)
    return ((np.exp(valscpu[idx])) / a).item() * 100


def testModel(model, sentence):
    cate_fuck = ['비욕설', '욕설']
    cate_badword = ['일반_긍정', '섹슈얼', '혐오']
    sentence = re.sub(r"([!@#$%^&*()_+=,./?0-9])", r"", sentence)
    tokenized = tokenizer(sentence)
    tokens = tokenizer.convert_ids_to_tokens(tokenized['input_ids'])
    input_ids, token_type_ids, attention_mask = np.array(tokenized['input_ids']), np.array(tokenized['token_type_ids']), np.array(tokenized['attention_mask'])
    input_ids, token_type_ids, attention_mask = torch.LongTensor([input_ids]).to(device), torch.LongTensor([token_type_ids]).to(device), torch.LongTensor([attention_mask]).to(device)

    fuck, badword, (attention_prob, cosine_sim) = model(input_ids, token_type_ids, attention_mask)
    fuck_idx = fuck.argmax().cpu().item()
    badword_idx = badword.argmax().cpu().item()
    print("문장에는:", cate_fuck[fuck_idx], "{:.2f}%".format(softmax(fuck, fuck_idx)), cate_badword[badword_idx], "{:.2f}%".format(softmax(fuck, fuck_idx)))
    print(tokens[1:-1])
    print(attention_prob)
    print(cosine_sim)


while True:
    s = input('input: ')
    if s == 'quit':
        break
    testModel(model, s)
