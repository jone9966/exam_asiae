import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import f1_score
from dataloader.dataloader import BERTDataset
from model.kobert_model import BERTClassifier


device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print(device)

# dataset_train = []
# file = open('../input/test.txt', 'r', encoding='utf-8')
# while True:
#     line = file.readline()
#     if not line:
#         break
#     dataset_train.append([line[:-3], line[-2]])
# file.close()
# dataset_train = dataset_train[:5000]
# print(dataset_train[:5])

label = ['비속어', '카테고리', 'text']
df = pd.read_csv("../input/Feb_22_bad_lang_ready_to_trainset.csv", sep='|')

dataset_train = []
for i in range(len(df)):
    label = 0 if df.iloc[i, 0] == '비욕설' else 1
    dataset_train.append([df.iloc[i, 2], label])


max_len = 64
batch_size = 4
warmup_ratio = 0.1
num_epochs = 15
max_grad_norm = 1
learning_rate = 5e-5
# add_token = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
#              'ㄲ', 'ㄸ', 'ㅃ', 'ㅆ', 'ㅉ', 'ㄳ', 'ㄵ', 'ㄶ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅄ']


train_dataset = BERTDataset(dataset_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
model = BERTClassifier(vocab_size=train_dataset.vocab_size)
model.to(device)


no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()


t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)


# continue_train(model, optimizer,start_epoch, '../output/model/cp_badword.pt')
losses = []
for epoch in range(num_epochs):
    model.train()
    for batch_id, (input_ids, token_type_ids, attention_mask, label) in enumerate(tqdm(train_dataloader)):
        input_ids = input_ids.long().to(device)
        token_type_ids = token_type_ids.long().to(device)
        attention_mask = attention_mask.long().to(device)
        label = label.long().to(device)

        optimizer.zero_grad()
        out = model(input_ids, token_type_ids, attention_mask)
        loss = loss_fn(out, label)
        loss.backward()
        losses.append(loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
    print(sum(losses)/len(losses))
    state = {'Epoch': epoch,
             'State_dict': model.state_dict(),
             'Optimizer': optimizer.state_dict()}
    # torch.save(state, '../output/model/cp_badword_cls.pt')

model.eval()
torch.save(model, '../output/model/kcbert_text.pt')


# def continue_train(model, optimizer, start_epoch, path):
#     checkpoint = torch.load(path)
#     start_epoch = checkpoint['Epoch']
#     model.load_state_dict(checkpoint['State_dict'])
#     optimizer.load_state_dict(checkpoint['Optimizer'])

