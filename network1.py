#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: wang
# Time:2019/12/8 17:13
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable

#读取特征文件
def read_feature(fname):
    fea=np.loadtxt(fname)
    return fea

#读取filter
def read_filter(filter):
    filter_file=np.load(filter)
    return filter_file

#定义网络
class net1(nn.Module):
    def __init__(self,num_input,num_hidden,num_output):
        super(net1,self).__init__()
        layer1=nn.Sequential()
        layer1.add_module('conv1',nn.Conv2d(1,5,(4,30),1))
        layer1.add_module('relu',nn.ReLU(True))
        layer1.add_module('pool1',nn.MaxPool2d(2,2))
        self.layer1=layer1

    def forward(self,x):
        x=self.layer1(x)
        return x






# class net1(nn.Module):
#     def __init__(self,num_input,num_hidden,num_output):
#         super(net1,self).__init__()
#         layer1=nn.Sequential()
#         layer1.add_module('conv1',nn.Conv2d(1,5,(4,30),1))
#         layer1.add_module('relu',nn.ReLU(True))
#         layer1.add_module('pool1',nn.MaxPool2d(2,2))
#         self.layer1=layer1
#
#     def forward(self,x):
#         x=self.layer1(x)
#         return x





if __name__=='__main__':
    feature=read_feature('features_1/one_hot')

