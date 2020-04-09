#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: wang
# Time:2019/12/4 21:14

#测试生成PWM的正确性
from Bio import motifs
from Bio.Seq import Seq
import numpy as np

def read_fasta(file):
    f = open(file)
    docs = f.readlines()
    fasta = []
    for seq in docs:
        if seq.startswith(">"):
            continue
        else:
            fasta.append(seq.strip())
    return np.array(fasta)

def trans_motif(fasta):
    instances=[]
    for item in fasta:
        instances.append(Seq(item))
    return instances

if __name__=='__main__':
    fasta=read_fasta('features_1/test1.txt')
    instances=trans_motif(fasta)
    m=motifs.create(instances)
    print(m.counts)
    print(m.counts.normalize(pseudocounts={'A':0.6, 'C': 0.4, 'G': 0.4, 'T': 0.6}))
























'''instances=[Seq("TACAA"),
           Seq("TACGC"),
           Seq("TACAC"),
           Seq("TACCC"),
           Seq("AACCC"),
           Seq("AATGC"),
           Seq("AATGC"),
           ]
m=motifs.create(instances)
print(m)
print(m.counts)'''
