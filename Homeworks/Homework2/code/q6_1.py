# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 16:48:41 2019

@author: Royed
"""

import time
import math
import torch
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import torch.utils.data as data_utils


######################## classes #######################

            
class model_2(nn.Module):
    def __init__(self, vocab):
        super(model_2, self).__init__()
        self.vocab_size = len(vocab)     
        
        self.fc = nn.Sequential(
            nn.Linear(self.vocab_size, 2),
            nn.Sigmoid()
        )
    
    def forward(self, seq):
        out = self.fc(seq)
        return out         


def preprocess(filename, mode=""):
    #read in tokens
    vocalubary = set()
    with open(filename) as f:
        words_matrix = []
        lines = f.readlines()
        labels = []
        for l in lines:
            words = l.split()
            if mode != 'un':
                labels.append(int(words[0]))
                words_matrix.append(words[1:])
                for w in words[1:]:
                    vocalubary.add(w)
            else:
                labels.append(-1)
                words_matrix.append(words)
                for w in words:
                    vocalubary.add(w)

    dummy_train = pd.get_dummies(list(vocalubary))
    return dummy_train, vocalubary, words_matrix, labels

def encode(dummy_train, words_matrix, labels):
    bag_of_words_matrix = []
    for lidx, line in enumerate(words_matrix):
        if lidx % 500 == 0:
            print(lidx)
        dummy_test = pd.get_dummies(line)
        dummy_test = dummy_test.reindex(columns = dummy_train.columns, fill_value=0)
        bag_of_words_matrix.append(dummy_test.values.any(axis=0).astype('int'))
    ds = data_utils.TensorDataset(torch.FloatTensor(bag_of_words_matrix), torch.LongTensor(labels))
    return ds


dummy_train, vocalubary, words_matrix, labels=preprocess('data/train.txt')
train_ds = encode(dummy_train, words_matrix, labels)
trainloader = data_utils.DataLoader(train_ds, batch_size=50, shuffle=True)

_, _, words_matrix_t, labels_t=preprocess('data/test.txt')
test_ds = encode(dummy_train, words_matrix_t, labels_t)
testloader = data_utils.DataLoader(test_ds, batch_size=50, shuffle=True)

_, _, words_matrix_un, labels_un =preprocess('data/unlabelled.txt', 'un')
un_ds = encode(dummy_train, words_matrix_un, labels_un)
predict_loader = data_utils.DataLoader(un_ds, batch_size=10001, shuffle=True)



######################## main ############################

learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn
#vocab_size = len(vocab)
n_out = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

net = model_2(vocalubary).to(device)
criterion = nn.CrossEntropyLoss()
#criterion = nn.NLLLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=learning_rate)

######################### train #######################




for epoch in range(10):  # loop over the dataset multiple times
    start = time.time()
    running_loss = 0.0
    for i, (texts,  labels) in enumerate(trainloader):
        #(texts, lengths ), labels = d
        #print(i, ((texts, lengths ), labels))
        texts = texts.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward pass
        outputs = net(texts)
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
correct = 0
total = 0
test_start = time.time()
with torch.no_grad():
    for texts, labels in testloader:
        #print(texts,lengths, labels)
        texts = texts.to(device)
        labels = labels.to(device)
        outputs = net(texts)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the 10000 test images: %.3f %%' % (
    100 * correct / total))
print("testing used: ", time.time() - test_start)
######################### predict #######################
#very large batch size, so only 1 batch.
#all_predictions = []
pred_start = time.time()
with torch.no_grad():
    for texts, _ in predict_loader:
        texts = texts.to(device)
        outputs = net(texts)
        _, predicted = torch.max(outputs.data, 1)
        print(predicted)
        #all_predictions.append(predicted)
print("prediction used: ", time.time() - pred_start)
with open('data/predictions_q1.txt', 'w') as f:
    for i in predicted:
        f.write(str(int(i)))
        f.write('\n')