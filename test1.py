#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: wang
# Time:2020/1/15 10:10

#读取测试集样本,观察数据是否正确


def read_f(filename):
    f=open(filename)
    listfile=f.readlines()
    for item in listfile:
        fl=item.split(' ')
        #print(fl[0])
        print(fl[0],fl[-1].split('\n')[0])#(序列,class类别)
    f.close()




if __name__=='__main__':
    read_f('train_valid_test/test/test_data.txt')
