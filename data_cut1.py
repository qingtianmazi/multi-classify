#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: wang
# Time:2019/11/1 10:48
#切割125种细胞类型,按照1000bp切取
import os
import pandas as pd
from pyfasta import Fasta

#读取指定目录下的所有文件
def filename(string):
    for root,dirs,files in os.walk(string):
        return files

#需要将每一个xlsx文件都写到单独的文件中去
def read_file(files,f):
    for file in files:
        print(file)
        namelists = file.split('.')  # 获取这一部分wgEncodeAwgDnaseDuke8988tUniPk文件名
        output_file=open('data_after/'+namelists[0]+'.txt','w')
        output_label=open('data_after_label/'+namelists[0]+'.txt','w')
        #output_file = open('data_after_150/' + namelists[0] + '.txt', 'w')
        df = pd.read_excel('Encode125_xlsx/'+file, header=None)
        # 存放chrosome的名称
        chr_num = df.iloc[:, 0]
        # 起始位点(include)
        start_pos = df.iloc[:, 1]-425
        #切段长为150bp用于进行聚类
        #start_pos = df.iloc[:, 1]
        # 终止位点(exclude)
        end_pos = df.iloc[:, 2]+425
        #end_pos = df.iloc[:, 2]
        y=df.iloc[:,6]
        for index,string in enumerate(chr_num):
            str1=f.sequence({'chr':string,'start':start_pos[index],'stop':end_pos[index]},one_based=False).upper()
            #res='>{0}\n{1}'.format(string+':'+str(start_pos[index])+':'+str(end_pos[index]),str1)
            res='>{0}|{1}\n{2}'.format(string,y[index],str1)
            output_file.write(res+'\n')
            label='{0}'.format(y[index])
            output_label.write(label+'\n')
        output_file.close()


if __name__=='__main__':
    namelist=filename('Encode125_xlsx')
    fa=Fasta('hg19.fa')
    read_file(namelist,fa)




