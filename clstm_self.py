#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: wang
# Time:2020/1/11 20:04

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
#基于下载的网络结构更改

class CLSTM(nn.Module):
    #这里个args需要给定7个参数
    def __init__(self):
        super(CLSTM, self).__init__()
        # self.hidden_dim = lstm_hidden_dim
        # self.num_layers = lstm_num_layers
        # V = embed_num
        # D = embed_dim
        # #C = class_num
        # Co = kernel_num
        # Ks = kernel_sizes
        # self.embed = nn.Embedding(V, D, padding_idx=args.paddingId)
        # # pretrained  embedding
        # if args.word_Embedding:
        #     self.embed.weight.data.copy_(args.pretrained_weight)

        # CNN
        layer1 = nn.Sequential()
        #x_size(1,4,1000)
        layer1.add_module('conv1', nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(4,25), stride=1))
        #x_size(6,4,976)
        layer1.add_module('relu', nn.ReLU(True))
        layer1.add_module('pool1', nn.MaxPool2d(2, 2))
        #x_size(6,2,488)
        self.layer1=layer1
        # KK = []
        # for K in Ks:
        #     KK.append(K + 1 if K % 2 == 0 else K)
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D), stride=1, padding=(K//2, 0)) for K in KK]
        #self.convs1 = [nn.Conv2d(Ci, D, (K, D), stride=1, padding=(K // 2, 0)) for K in KK]
        # for cnn cuda
        # if self.args.cuda is True:
        #     for conv in self.convs1:
        #         conv = conv.cuda()
        # LSTM
        layer2 = nn.LSTM(input_size=2,hidden_size=20,num_layers=2)
        self.layer2=layer2
        #self.lstm = nn.LSTM(D, self.hidden_dim, num_layers=self.num_layers, dropout=args.dropout)

        # linear
        layer3 = nn.Sequential()
        layer3.add_module('fc layer',nn.Linear(in_features=,out_features=125))
        layer3.add_module('ReLU',nn.ReLU(True))
        self.layer3=layer3
        # self.hidden2label1 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        # self.hidden2label2 = nn.Linear(self.hidden_dim // 2, C)
        # dropout
        #self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        # embed = self.embed(x)
        # # CNN
        # cnn_x = embed
        # cnn_x = self.dropout(cnn_x)
        # cnn_x = cnn_x.unsqueeze(1)
        # cnn_x = [F.relu(conv(cnn_x)).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)
        # cnn_x = torch.cat(cnn_x, 0)
        # cnn_x = torch.transpose(cnn_x, 1, 2)
        # LSTM
        lstm_out, _ = self.lstm(cnn_x)
        lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)
        lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2)
        # linear
        cnn_lstm_out = self.hidden2label1(F.tanh(lstm_out))
        cnn_lstm_out = self.hidden2label2(F.tanh(cnn_lstm_out))
        # output
        logit = cnn_lstm_out

        return logit




