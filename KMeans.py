#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: wang
# Time:2019/11/14 9:08

#伪代码:
#创建k个点作为起始的质心(随机产生)
#当任意一个点的簇结果发生改变时
#   对数据集中每一个点
#       对每一个质心
#           计算点到质心的距离
#       将数据点分配到距离最短的簇
#   更新每个簇的质心(取每个簇的均值)
from second.smithwaterman import *
import numpy as np
from numpy import *
from pyforest import *
#load data

'''def read_fasta(file):
    f = open(file)
    docs = f.readlines()
    fasta = []
    for seq in docs:
        if seq.startswith(">"):
            continue
        else:
            fasta.append(seq.strip())
    return np.array(fasta)'''


'''def loadDataSet(filename):
    dataMat=[]
    fr=open(filename)
    for line in  fr.readlines():
        curLine=line.strip().split('\t')
        fltLine=list(map(float,curLine))
        dataMat.append(fltLine)
    return np.array(dataMat)'''
#计算所有样本点到中心点的距离
def distEclud(vecA,vecB):
    return np.sqrt(sum(np.power(vecA-vecB,2)))
#随机得到k个中心点
def randCent(dataSet,k):
    n=dataSet.shape[1]
    centroids=np.mat(np.zeros((k,n)))
    for j in range(n):
        minJ=min(dataSet[:,j])
        maxJ=max(dataSet[:,j])
        rangeJ=float(maxJ-minJ)
        centroids[:,j]=rangeJ*np.random.rand(k,1)+minJ
    return centroids

def kMeans(dataSet,k,distMeas=distEclud,createCent=randCent):
    m=np.shape(dataSet)[0]
    clusterAssment=np.mat(np.zeros((m,2)))#创建每个点的簇分配结果(index,dist)
    centroids=createCent(dataSet,k)
    clusterChanged=True
    while clusterChanged:
        clusterChanged=False
        for i in range(m):#遍历所有点,用于对所有点进行簇的分配
            minDist=np.inf;minIndex=-1
            for j in range(k):#遍历所有质心,便于将点划分到这个簇中
                distJI=distMeas(centroids[j,:],dataSet[i,:])
                #print(minDist,'\n',distJI)
                if distJI<minDist:#更新最小距离,把当前点所属的簇指出为minIndex
                    minDist=distJI;minIndex=j
            if clusterAssment[i,0]!=minIndex:#查看所属簇是否发生改变
                clusterChanged=True
            clusterAssment[i,:]=minIndex,minDist**2
        print(centroids)
        for cent in range(k):
            ptsInClust=dataSet[np.nonzero(clusterAssment[:,0].A==cent)[0]]#获得给定簇的所有点,dataSet可以按照行索引编号取元素(一个parameter表示行索引)
            centroids[cent,:]=np.mean(ptsInClust,axis=0)
    return centroids,clusterAssment

if __name__=='__main__':
    #dataMat = loadDataSet('testSet.txt')

    fasta_file=pd.read_csv('features_1/train.csv')

    #cent=sample(list(fasta_file),50)#随机抽取50条样本作为聚类中心
    myCentroids, clustAssing = kMeans(fasta_file.values, 50)
    clustAssingDF=pd.DataFrame(clustAssing)
    clustAssingDF.to_csv('features_1/cluster.csv')






