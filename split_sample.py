#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: wang
# Time:2019/12/4 11:03

#对所有样本按照类别进行划分
import pandas as pd
import numpy as np
from Bio import motifs
from Bio.Seq import Seq

#读取序列文件
def read_file(filename):
    f=open(filename)
    flist=f.readlines()
    seq=[]
    label=[]
    for s in flist:
        sl=s.strip()
        sl=sl.split(' ')
        #print(sl[0])
        #print(sl[-1])
        seq.append(sl[0])
        label.append(sl[-1])
    return seq,label


#创建聚类字典  (key,value)=(类别,序列list)
def create_dict(clustlist,seqlist):
    cluster_dict={}
    for item1,item2 in zip(clustlist,seqlist):
        index=item1.strip()
        print(index)
        if index not in cluster_dict.keys():#还没有
            cluster_dict[index]=[]
        cluster_dict[index].append(item2.strip())
    return cluster_dict

#转换成motif,用于生成PWM
def trans_motif(fasta):
    instances=[]
    for item in fasta:
        instances.append(Seq(item))
    return instances

#创建PWM字典  (key,value)=(类别,PWM)
def create_PWMdict(clusterdict):
    PWMdict={}
    for key,value in clusterdict.items():
        if key not in PWMdict.keys():#还没有
            PWMdict[key]=[]
        instances = trans_motif(value)
        m=motifs.create(instances)
        #PWMdict[key].append(m.counts.normalize(pseudocounts=0))
        PWMdict[key].append(m.counts.normalize(pseudocounts={'A':0.6, 'C': 0.4, 'G': 0.4, 'T': 0.6}))
    return PWMdict

#切分PWM矩阵为K段,得到K个filters
def cut_PWM(PWMdict,k):
    cutpwm={}
    for key,value in PWMdict.items():
        if key not in cutpwm.keys():
            cutpwm[key]=[]
        period=len(value[0]['A'])/k #period=25
        for i in range(k):
            matrix=[]
            for num in range(len(value[0])):
                #print(value[0])
                start=int(i*period)
                end=int(i*period+period)
                matrix.append(list(value[0][num])[start:end])
            cutpwm[key].append(np.array(matrix))
    return cutpwm



if __name__=='__main__':
    fasta_file,cluster_file=read_file('train_valid_test/train/train_data.txt')
    clusterdict=create_dict(cluster_file,fasta_file)
    PWMdict=create_PWMdict(clusterdict)
    np.save('features_1/my_clusterdict.npy', clusterdict)
    np.save('features_1/my_PWMdict1.npy',PWMdict)
    cutPWM=cut_PWM(PWMdict,6)









