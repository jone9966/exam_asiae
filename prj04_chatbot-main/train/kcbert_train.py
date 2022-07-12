import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm
from dataloader.dataloader import KcBERTDataset
from model.kcbert_model import KcBERTClassifier
from train.EarlyStopping import EarlyStopping

device = torch.device("cuda:0")
# device = torch.device("cpu")

df = pd.read_csv("../input/final_train_test_dataset_normalize_0225.csv", sep='|')

dataset = []
for i in range(len(df)):
    fuck_label = 0 if df.iloc[i, 0] == '비욕설' else 1
    if df.iloc[i, 1] == '일반_긍정':
      badword_label = 0
    elif df.iloc[i, 1] == '섹슈얼':
      badword_label = 1
    else :
      badword_label = 2
    dataset.append([df.iloc[i, 2], fuck_label, badword_label])

max_len = 64
batch_size = 8
warmup_ratio = 0.1
n_epochs = 30
max_grad_norm = 1
learning_rate = 5e-5
early_stop = 7

train_size = int(0.8 * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

train_dataset = KcBERTDataset(train_dataset)
valid_dataset = KcBERTDataset(valid_dataset)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=0, shuffle=True)

model = KcBERTClassifier(device=device)
model.to(device)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

t_total = len(train_dataloader) * n_epochs
warmup_step = int(t_total * warmup_ratio)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)


avg_train_losses = []
avg_valid_losses = []
early_stopping = EarlyStopping(patience=early_stop, verbose=True)

for epoch in range(1, n_epochs+1):
    train_losses = []
    valid_losses = []

    model.train()
    for batch_id, (input_ids, token_type_ids, attention_mask, fuck_label, badword_label) in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()
        fuck_out, badword_out, _ = model(input_ids, token_type_ids, attention_mask)
        loss_fuck = loss_fn(fuck_out, fuck_label.long().to(device))
        loss_badword = loss_fn(badword_out, badword_label.long().to(device))
        loss = loss_fuck + loss_badword
        loss.backward()
        train_losses.append(loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

    model.eval()
    for input_ids, token_type_ids, attention_mask, fuck_label, badword_label in valid_dataloader :
        fuck_out, badword_out, _ = model(input_ids, token_type_ids, attention_mask)
        loss_fuck = loss_fn(fuck_out, fuck_label.long().to(device))
        loss_badword = loss_fn(badword_out, badword_label.long().to(device))
        loss = loss_fuck + loss_badword
        valid_losses.append(loss.item())

    train_loss = np.average(train_losses)
    valid_loss = np.average(valid_losses)
    avg_train_losses.append(train_loss)
    avg_valid_losses.append(valid_loss)

    epoch_len = len(str(n_epochs))
    print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                  f'train_loss: {train_loss:.5f} ' +
                  f'valid_loss: {valid_loss:.5f}')
    print(print_msg)

    early_stopping(valid_loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

model.load_state_dict(torch.load('../output/checkpoint.pt'))
torch.save(model, '../output/model/kcbert_multi.pt')