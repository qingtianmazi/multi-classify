#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: wang
# Time:2020/1/14 14:38

#对数据集进行划分,按照3:1:1划分训练集、验证集、测试集
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split

#读取文件夹下所有文件
def filename(string):
    for root,dirs,files in os.walk(string):
        return files

def read_file(namelist,label):
    f1 = open('train_valid_test/train/train_data.txt', 'w')
    f2 = open('train_valid_test/valid/valid_data.txt', 'w')
    f3 = open('train_valid_test/test/test_data.txt', 'w')
    for x_file,y_file in zip(namelist,label):
        x_f=open('data_after_phase2/'+x_file)
        y_f=open('data_after_label_phase2/'+y_file)
        x_data=x_f.readlines()
        y_data=y_f.readlines()
        x_train_valid,x_test,y_train_valid,y_test=train_test_split(x_data,y_data,test_size=0.2,random_state=10)
        x_train,x_valid,y_train,y_valid=train_test_split(x_train_valid,y_train_valid,test_size=0.25,random_state=10)
        for i,j in zip(x_train,y_train):
            # print(i.split('\n')[0]+' '+j.split('\n')[0])
            # print('\n')
            f1.write(i.split('\n')[0]+' '+j.split('\n')[0])
            f1.write('\n')
        for k,m in zip(x_valid,y_valid):
            f2.write(k.split('\n')[0]+' '+m.split('\n')[0])
            f2.write('\n')
        for a,b in zip(x_test,y_test):
            f3.write(a.split('\n')[0]+' '+b.split('\n')[0])
            f3.write('\n')
            print(a.split('\n')[0]+' '+b.split('\n')[0])

        x_f.close()
        y_f.close()
    f1.close()
    f2.close()
    f3.close()



#train_set,test_set= train_test_split()



if __name__=='__main__':
    namelist1=filename('data_after_phase2')
    label=filename('data_after_label_phase2')
    read_file(namelist1,label)


















