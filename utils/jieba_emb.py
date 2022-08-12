import jieba
import os
import pandas as pd
import numpy as np
import sys
from text2vec import SentenceModel, EncoderType
from text2vec import Word2Vec
sys.path.append('..')

def pretrained_embdding(train_data):

    # 取出所有词
    vocab = []
    for line in train_data:
        words = line.split(' ')
        vocab.extend(words)
    vocab = set(vocab)

    word_embedding = Word2Vec("w2v-light-tencent-chinese").encode(vocab)
    all_embedding = {}
    for word, embdding in zip(vocab, word_embedding):
        all_embedding[word] = embdding
    pre_embedding = pd.DataFrame(data=all_embedding)

    return all_embedding, pre_embedding



def tokenizer(train_data):
    # 加载字典
    word_dict = './data/word_dictionaries/'
    filename = os.listdir(word_dict)
    for file in filename:
        path = os.path.join(word_dict, file)
        jieba.load_userdict(path)

    # 得到分词结果
    seg_data = []
    for line in train_data:
        seg_list = jieba.cut(line, use_paddle=True)
        seg_data.append(' '.join(seg_list))
    seg_data = pd.Series(seg_data).astype(str)



    # try:
    #     with open('./seg_word/seg_data.csv') as _:
    #         seg_data = pd.read_csv('./seg_word/seg_data.csv', sep=' ',header=None).astype(str)
    # except FileNotFoundError:
    #     seg_data.to_csv('./seg_word/seg_data.csv', sep=' ', header=False, index=False)

    # if os.path.exists('./seg_word/seg_data.csv'):
    #     seg_data = pd.read_csv('./seg_word/seg_data.csv', sep=',',header=None, squeeze=True).astype(str)
    # else:
    #     seg_data.to_csv('./seg_word/seg_data.csv', sep=',', header=False, index=False)
        

    return seg_data

