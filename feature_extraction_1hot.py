#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: wang
# Time:2019/11/6 15:50

import numpy as np

def read_fasta(file):
    f = open(file)
    docs = f.readlines()
    fasta = []
    for seq in docs:
        if seq.startswith(">"):
            continue
        else:
            fasta.append(seq)
    return np.array(fasta)
#one-hot   A:1,0,0,0  C:0,1,0,0  G:0,0,1,0  T:0,0,0,1
def binary(fastas, **kw):
    AA = 'ACGT'
    encodings = []
    for i in fastas:
        sequence = i.strip()
        code = []
        for aa in sequence:
            if aa == '-':#针对某些特殊格式
                code = code + [0, 0, 0, 0]
                continue
            for aa1 in AA:
                tag = 1 if aa == aa1 else 0
                code.append(tag)
        encodings.append(code)
    #np.savetxt("binary", encodings)
    np.savetxt('features_1/one_hot',encodings)
    return np.array(encodings)


if __name__=='__main__':
    fasta=read_fasta('data_after/wgEncodeAwgDnaseDuke8988tUniPk.txt')
    one_hot=binary(fasta)





