#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: wang
# Time:2019/11/8 10:31
import numpy as np
import pandas as pd
import re
import gensim
import word2vec

#对fasta文件进行分词操作

def save_wordfile(fastafile, wordfile, splite, kmer):
    f = open(fastafile)
    f1 = open(wordfile, "w")
    k = kmer - 1
    documents = f.readlines()
    string = ""
    flag = 0
    for document in documents:
        if document.startswith(">") and flag == 0:
            flag = 1
            continue
        elif document.startswith(">") and flag == 1:
            if splite == 0:#k-mer的切分方式,窗口滑动
                b = [string[i:i + kmer] for i in range(len(string)) if i < len(string) - k]
            else:#按照单词切分
                b = re.findall(r'.{kmer}', string)
            word = " ".join(b)
            f1.write(word)
            f1.write("\n")
            string = ""
        else:
            string += document
            string = string.strip()
    if splite == 0:
        b = [string[i:i + kmer] for i in range(len(string)) if i < len(string) - k]
    else:
        b = re.findall(r'.{kmer}', string)
    word = " ".join(b)
    f1.write(word)
    f1.write("\n")
    print("words have been saved in file {}！\n".format(wordfile))
    f1.close()
    f.close()

def save_csv(word_file, model, csv_file, b):
    wv = model.wv
    vocab_list = wv.index2word
    feature = []
    outputfile = csv_file
    with open(word_file) as f:
        # l = []
        words = f.readlines()
        for word in words:
            l = []
            cc = word.split(" ")
            for i in cc:
                i = i.rstrip()
                if i not in vocab_list:
                    flag = [b] * 100
                else:
                    flag = model[i]
                l.append(flag)
            word_vec = np.array(l)
            feature.append(np.mean(word_vec, axis=0))
        pd.DataFrame(feature).to_csv(outputfile, header=None, index=False)

    print("csv have been saved in file {}！\n".format(outputfile))


def tocsv(trainword_file, testword_file, sg, hs, window, size, model_name, traincsv, testcsv, b, flag, iter, spmodel):
    if spmodel:
        print("loading model")
        model = gensim.models.KeyedVectors.load_word2vec_format(model_name, binary=False)
    else:
        sentences = word2vec.LineSentence(trainword_file)
        model = word2vec.Word2Vec(sentences, sg=sg, hs=hs, min_count=1, window=window, size=size, iter=iter)
        model.wv.save_word2vec_format(model_name, binary=False)

    save_csv(trainword_file, model, traincsv, b)

    if flag:
        save_csv(testword_file, model, testcsv, b)


if __name__=='__main__':
    save_wordfile('data_after_150/wgEncodeAwgDnaseDuke8988tUniPk.txt','features_1/wordfile1.txt',1,3)


