#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: wang
# Time:2019/12/10 10:39
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable

def embedMethod(temp):
    w2id={}
    string="ACGT"
    for i,j in enumerate(string):
        w2id[j]=i
    #print(w2id)
    vectors=[]
    vectors.append([1,0,0,0])#A
    vectors.append([0,1,0,0])#C
    vectors.append([0,0,1,0])#G
    vectors.append([0,0,0,1])#T
    a=np.array(vectors)
    a=torch.tensor(a).float()
    #print(a.shape)
    embeds=nn.Embedding(4,4).from_pretrained(embeddings=a,freeze=False)
    #print(embeds.weight)
    # for key,value in dict.items():
    #     #print(key,value)
    #     A_idx=Variable(torch.LongTensor([dict[key]]))
    #     A_dx=embeds(A_idx)
    #     print(A_dx)
    str1=temp
    list=[]
    for i in str1:
        if i in w2id.keys():
            # print("ddd")
            list.append(w2id[i])

    #print(list)
    f=np.array(list)
    dd=torch.tensor(f).long()
    #print(dd)
    gg=embeds(dd)
    #print(gg)





