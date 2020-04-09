#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: wang
# Time:2019/10/11 11:17

#按照hg19-blacklist切割出来的411段序列

from pyfasta import Fasta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_excel('chr/hg19.xlsx',header=None)
#存放chrosome的名称
chr_num=df.iloc[:,0]
#起始位点
start_pos=df.iloc[:,1]
#终止位点
end_pos=df.iloc[:,2]
#切段长度
length=end_pos-start_pos
print(max(length),min(length))
x=np.arange(0,411)
plt.xlabel('number of parts')
plt.ylabel('length of parts')
plt.plot(x,length)
plt.show()

with open('sequence.txt','w') as fw:
    for index,string in enumerate(chr_num):
        f=Fasta('chr/'+string+'.fa')
        s=str(f[string])
        slic=s[start_pos[index]:end_pos[index]]
        fw.write(str(slic))
        fw.write('\n')


        #每一个数据文件对应一个DataFrame,创建一个空的DataFrame
        result = pd.DataFrame(columns=['string'])
        #读取对应的hg19文件
        for index, string in enumerate(chr_num):
            f = Fasta('chr/' + string + '.fa')
            s = str(f[string])
            slic = s[start_pos[index]:end_pos[index]]#注意一下是否包含末端的问题
            result=result.append(pd.DataFrame({'string':[slic]}),ignore_index=True)
        #result.to_csv('data_after/'+namelist[0]+'.csv',sep='\n',header=None,index=True)


