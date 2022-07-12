import re
import numpy as np
import torch

from model.kcbert_model import BERTClassifier
from model.kogpt2_model import GPT2Chat
from transformers import PreTrainedTokenizerFast, BertTokenizer, BertModel

ctx = 'cuda:0'
device = torch.device(ctx)

badword_model = torch.load('./output/model/kcbert_multi.pt')
badword_model.to(device)
tokenizer = BertTokenizer.from_pretrained("beomi/kcbert-large")


chat_model = torch.load('./output/model/chatbot_big.pt')
chat_model.to(device)
koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                           bos_token='<s>', eos_token='</s>', unk_token='<unk>',
                                                           pad_token='<pad>', mask_token='<unused0>')


def totensor(i):
    return torch.LongTensor([np.array(i)]).to(device)


with torch.no_grad():
    while 1:
        q = input("user > ").strip()
        if q == "quit":
            break
        a = ""
        while 1:
            q = re.sub(r"([!@#$%^&*()_+=,./?0-9])", r"", q)
            tokenized = tokenizer(q)
            input_ids, token_type_ids, attention_mask = totensor(tokenized['input_ids']), totensor(tokenized['token_type_ids']), totensor(tokenized['attention_mask'])

            fuck, badword, (attention_prob, cosine_sim) = badword_model(input_ids, token_type_ids, attention_mask)
            fuck_idx = fuck.argmax().cpu().item()
            badword_idx = badword.argmax().cpu().item()
            if fuck_idx == 1:
                a = "비속어는 안돼요!"
                break

            input_ids = torch.LongTensor(
                koGPT2_TOKENIZER.encode("<usr>" + q + '<unused1>' + "<sys>" + a)).unsqueeze(dim=0)
            input_ids = input_ids.to(ctx)
            pred = chat_model(input_ids)
            pred = pred.logits
            pred = pred.cpu()
            gen = koGPT2_TOKENIZER.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]
            if gen == '</s>':
                break
            a += gen.replace("▁", " ")
        print("Chatbot > {}".format(a.strip()))