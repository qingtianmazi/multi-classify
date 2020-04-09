#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: wang
# Time:2019/12/3 17:14

#对所有样本进行聚类

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

fasta_file=pd.read_csv('features_1/train.csv',header=None)
kmeans = KMeans(n_clusters=50, random_state=0).fit(fasta_file.values)
labels=kmeans.labels_
#print(labels)
df=pd.DataFrame(np.array(labels))
df.to_csv('features_1/cluster_res.csv')
#cent=sample(list(fasta_file),50)#随机抽取50条样本作为聚类中心
'''myCentroids, clustAssing = KMeans(fasta_file.values, 50)
clustAssingDF=pd.DataFrame(clustAssing)
clustAssingDF.to_csv('features_1/cluster.csv')'''




