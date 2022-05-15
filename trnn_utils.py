import numpy as np
import pickle as pkl
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import *
import torch.optim as optim
torch.manual_seed(0)
c_len = 100
embed_d = 128

# data class
class input_data():
    def load_text_data(self, word_n = 100000):
        f = open('paper_abstract.pkl', 'rb')
        p_content_set = pkl.load(f)
        f.close()

        p_label = [0] * 21044
        label_f = open('paper_label.txt', 'r')
        for line in label_f:
            line = line.strip()
            label_s = re.split('\t',line)
            p_label[int(label_s[0])] = int(label_s[1])
        label_f.close()

        def remove_unk(x):
            return [[1 if w >= word_n else w for w in sen] for sen in x]

        p_content, p_content_id = p_content_set
        p_content = remove_unk(p_content)

        # padding with max len 
        for i in range(len(p_content)):
            if len(p_content[i]) > c_len:
                p_content[i] = p_content[i][:c_len]
            else:
                pad_len = c_len - len(p_content[i])
                p_content[i] = np.lib.pad(p_content[i], (0, pad_len), 'constant', constant_values=(0,0))

        p_id_train = []
        p_content_train = []
        p_label_train = []
        p_id_test = []
        p_content_test = []
        p_label_test = []
        for j in range(len(p_content)):
            if j % 10 in (3, 6, 9):
                p_id_test.append(p_content_id[j])
                #p_content_test.append(p_content[j])
                p_label_test.append(p_label[j])
            else:
                p_id_train.append(p_content_id[j])
                #p_content_train.append(p_content[j])
                p_label_train.append(p_label[j])

        p_train_set = (p_id_train, p_label_train)
        p_test_set = (p_id_test, p_label_test)

        return p_content, p_train_set, p_test_set


    def load_word_embed(self):
        #return word_embed
        word_embed = np.zeros((32784 + 2, 128))
        with open('word_embeddings.txt') as f:
            file = f.readlines()
            sentences = file[1:]
            for i, line in enumerate(sentences):
                line = line.strip().split(' ')
                word_embed[i,:] = np.array([float(x) for x in line[1:]])
        return word_embed

#text RNN Encoder
class Text_Encoder(nn.Module):
    def __init__(self, p_content, word_embed, learning_rate, device):
        # two input: p_content - abstract data of all papers, word_embed - pre-trained word embedding 
        super(Text_Encoder, self).__init__()

        self.word_embed = torch.Tensor(word_embed)
        self.p_content = torch.Tensor(p_content)
        self.learning_rate = learning_rate
        self.embed = nn.Embedding(32786, embed_d)
        self.device = device

        # LSTM layer
        self.lstm = nn.LSTM(128, 64)
        # fully connected layer
        self.decoder1 = nn.Linear(64, 32)
        self.decoder2 = nn.Linear(32, 5)
        # activation function
        self.sig = nn.Sigmoid()
        # optimizer
        self.opt = optim.Adam(self.parameters(), lr = self.learning_rate)

    def forward(self, id_batch):
        # id_batch: use id_batch (paper ids in this batch) to obtain paper conent of this batch 
        inputs = self.p_content[id_batch].long().to(self.device)
        x = self.embed(inputs)
        x = x.permute(1, 0, 2)
        x, (lstm_output, cn) = self.lstm(x)
        lstm_output = lstm_output.view(-1, 64)
        x = self.decoder1(lstm_output)
        x = F.relu(x)
        x = self.decoder2(x)
        x = self.sig(x)
        return x








