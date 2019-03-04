# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 16:48:41 2019

@author: Royed
"""

import torchtext
from torchtext.data import Field, BucketIterator, TabularDataset,Iterator
import time
import math
import torch
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd


######################## csv #######################

def create_csv(filename, labelled=True):
    lines = open(filename, encoding='utf-8').read().split('\n')
    if labelled:
        #lines[:-1] except last line: empty
        raw_data = {'text' : [line[1:] for line in lines[:-1]], 'label': [int(line[0]) for line in lines[:-1]]}
    else:
        raw_data = {'text' : [line for line in lines[:-1]], 'label': [2 for line in lines[:-1]]}
    df = pd.DataFrame(raw_data, columns=["text", "label"])
    return df
small=create_csv('data/train.txt')
small.to_csv("data/train.csv",index=False)
smallt = create_csv('data/test.txt')
smallt.to_csv("data/test.csv",index=False)
small_un = create_csv('data/unlabelled.txt', False)
small_un.to_csv("data/unlabelled.csv",index=False)


######################## classes #######################
class BatchGenerator:
    def __init__(self, dataloader, x_field, y_field):
        self.dataloader, self.x_field, self.y_field = dataloader, x_field, y_field
        
    def __len__(self):
        return len(self.dataloader)
    
    def __iter__(self):
        for batch in self.dataloader:
            X = getattr(batch, self.x_field)
            y = getattr(batch, self.y_field)
            yield (X,y)
            
class model_2(nn.Module):
    def __init__(self, vocab, embedding_dim, mode):
        super(model_2, self).__init__()
        self.vocab_size = len(vocab)
        self.embedding_dim = embedding_dim
        
        '''
        # 1 is the padding number 
        self.emb = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=1)
        #self.pool = nn.AvgPool1d(kernel_size=1, stride=None, padding=0, count_include_pad=False)
        '''
        #for pretrain: pad_index=1
        self.emb = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=1)
        if mode == "glove":
            self.emb.weight.data.copy_(vocab.vectors)
        #freeze embedding weight
        #self.emb.weight.requires_grad = False
        
        
        self.fc = nn.Sequential(
            nn.Linear(self.embedding_dim, 2),
            nn.Sigmoid()
        )
    
    def forward(self, seq, lengths):
        seq = seq.transpose(0,1)
        #print('seq_len: ',seq.shape) #3, 4: batch_size, input_dim(max sentence length in batch)
        embs = self.emb(seq) #3, 4, 5: batch_size, seq_len, hidden_dim
        
        embs = embs.transpose(1,2)
        #print('embedding after transpose: ', embs.shape) #want: 3,5,4
        #print(embs)
        avg_pool = torch.sum(embs, dim=2)/Variable(torch.LongTensor(lengths).view(-1,1)).float()
        #print(avg_pool.shape)
        out = self.fc(avg_pool)
        return out         


########################## preprocess ###############
#######!!!!!!! modify mode here !!!!!!!!#############
mode = "q2"
#######!!!!!!! modify mode here !!!!!!!!#############
#glove mode is for q3, pretrained. else it is for q2.

TEXT = Field(include_lengths=True)
LABEL = Field(sequential=False, use_vocab=False)

### 
train_ds, test_ds, un_ds = TabularDataset.splits(
        path='data/', train='train.csv',
        validation='test.csv', test='unlabelled.csv',
        format='csv', fields=[('text', TEXT), ('label', LABEL)], skip_header=True)

if mode == "glove":
    #use pretrained
    print("GLOVE!")
    TEXT.build_vocab(train_ds, vectors = 'glove.6B.100d')
else:
    TEXT.build_vocab(train_ds)


vocab = TEXT.vocab

##??????? val shuffle????
train_iter = BucketIterator(train_ds, batch_size=64, device=-1, # if you want to use the GPU, specify the GPU number here
 train=True, sort_key=lambda x: len(x.text), # the BucketIterator needs to be told what function it should use to group the data.
 sort_within_batch=False, repeat=False, # we pass repeat=False because we want to wrap this Iterator layer.
 shuffle=True, 
)
test_iter, un_iter = Iterator.splits((test_ds, un_ds), batch_sizes=(64, 10001), device=-1, 
                              sort=False, sort_within_batch=False, repeat=False, 
                              shuffle=False)

#train_iter = BucketIterator(train_ds, batch_size=3, shuffle=True,train=True)
#val_iter = BucketIterator(val_ds, batch_size=3, shuffle=False,train=False)

train_batch_dl = BatchGenerator(train_iter, 'text', 'label')
#val_batch_dl = BatchGenerator(val_iter, 'text', 'label')
test_batch_dl = BatchGenerator(test_iter, 'text', 'label')
un_batch_dl = BatchGenerator(un_iter, 'text', 'label')

######################## main ############################

learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn
#vocab_size = len(vocab)
embedding_dim = 100
n_out = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

net = model_2(vocab, embedding_dim, mode).to(device)
criterion = nn.CrossEntropyLoss()
#criterion = nn.NLLLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=learning_rate)

######################### train #######################
trainloader = train_batch_dl
for epoch in range(10):  # loop over the dataset multiple times
    start = time.time()
    running_loss = 0.0
    for i, ((texts, lengths ), labels) in enumerate(trainloader):
        #(texts, lengths ), labels = d
        #print(i, ((texts, lengths ), labels))
        texts = texts.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward pass
        outputs = net(texts, lengths)
        loss = criterion(outputs, labels)
        # backward pass
        loss.backward()
        # optimize the network
        optimizer.step()
        #print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            end = time.time()
            print('[epoch %d, iter %5d] loss: %.3f eplased time %.3f' %
                  (epoch + 1, i + 1, running_loss / 100, end-start))
            start = time.time()
            running_loss = 0.0
print('Finished Training')

######################### test #######################
testloader = test_batch_dl
correct = 0
total = 0
test_start = time.time()
with torch.no_grad():
    for (texts, lengths), labels in testloader:
        #print(texts,lengths, labels)
        texts = texts.to(device)
        labels = labels.to(device)
        outputs = net(texts, lengths)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the 10000 test images: %.3f %%' % (
    100 * correct / total))
print("testing used: ", time.time() - test_start)
######################### predict #######################
predict_loader = un_batch_dl
#very large batch size, so only 1 batch.
#all_predictions = []
pred_start = time.time()
with torch.no_grad():
    for (texts, lengths), _ in predict_loader:
        texts = texts.to(device)
        outputs = net(texts, lengths)
        _, predicted = torch.max(outputs.data, 1)
        print(predicted)
        #all_predictions.append(predicted)
print("prediction used: ", time.time() - pred_start)
with open('data/predictions'+mode+'.txt', 'w') as f:
    for i in predicted:
        f.write(str(int(i)))
        f.write('\n')