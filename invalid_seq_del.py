#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: wang
# Time:2020/1/14 11:00
#去除切割出的序列集中含有N的序列
import numpy as np
import os
import pandas as pd

#读取文件夹下所有文件
def filename(string):
    for root,dirs,files in os.walk(string):
        return files

def read_fasta(filelist):
    invalid=[]
    invalid_filename=[]
    valid=[]
    for index,file in enumerate(filelist):
        f = open('data_after/'+file)
        docs = f.readlines()
        output_file=open('data_after_phase2/'+file,'w')
        output_filter=open('data_after_150_phase2/'+file,'w')
        output_label=open('data_after_label_phase2/'+file,'w')
        for seq in docs:
            seq=seq.strip()
            if seq.startswith(">"):
                continue
            else:
                if seq.find('N')==-1 and len(seq)==1000:#没找到N
                    valid.append(seq)
                    output_file.write(seq+'\n')
                    output_filter.write(seq[425:425+150]+'\n')#取中间150bp
                    output_label.write(str(index)+'\n')
                else:
                    print(seq)
                    print(len(seq))
                    invalid_filename.append(file)
                    invalid.append(seq)
        output_label.close()
        output_file.close()
        output_filter.close()
        f.close()
    print(len(valid))
    print(len(invalid))




if __name__=='__main__':
    namelist=filename('data_after')
    read_fasta(namelist)







