import torch.nn as nn
import torch
from torch.autograd import Variable


class CNN(nn.Module):
    def __init__(self, vocab, embedding_dim, n_feature, kernel_h, mode='average'):
        super(CNN, self).__init__()
        self.vocab_size = len(vocab)
        self.embedding_dim = embedding_dim
        self.mode = mode
        '''
        # 1 is the padding number 
        self.emb = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=1)
        #self.pool = nn.AvgPool1d(kernel_size=1, stride=None, padding=0, count_include_pad=False)
        '''
        # for pretrain: pad_index=1
        self.emb = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=1)
        self.emb.weight.data.copy_(vocab.vectors)
        # freeze embedding weight
        # self.emb.weight.requires_grad = False

        ## in_channels, out_channels, kernel_size
        ##input: N,C,H output: N,F,H''
        self.cnn = nn.Conv1d(self.embedding_dim, n_feature, kernel_h)
        print(kernel_h)
        self.fc = nn.Sequential(
            nn.Linear(n_feature, 2),
            nn.Sigmoid()
        )

    def forward(self, seq, lengths):

        #H is L: seq len
        N, H = seq.shape
        #print('seq: ',seq.shape) #3, 4: batch_size, input_dim(max sentence length in batch)
        embs = self.emb(seq)  # 3, 4, 5: N, H, C
        #print('embeddings:', embs.shape)
        embs = embs.transpose(2, 1)
        #print('embeddings:', embs.shape) #N, C, H

        #embs = pack_padded_sequence(embs, lengths)  # (4+4+2), 5: each length sum, hidden_dim
        # print("padded: ",embs.data.shape)
        cnn_out = self.cnn(embs)
        #print('cnn out',cnn_out.shape)
        if self.mode == 'average':
            #print(self.mode)
            pool = nn.functional.avg_pool1d(cnn_out, cnn_out.shape[2])
        else:
            #print(self.mode)
            pool = nn.functional.max_pool1d(cnn_out, cnn_out.shape[2])
        pool =pool.reshape(pool.shape[0],-1)
        # print("LSTM: ", self.hidden[0].shape,self.hidden[1].shape) # 1,64,150: seq_len, batch_size, n_hidden
        # output, lengths = pad_packed_sequence(lstm_out)  ## need to care for loss masking
        # print(output.view(output.shape[1],-1).shape)
        #print('pooled:',pool.shape)
        out = self.fc(pool)
        return out

