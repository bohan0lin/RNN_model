import itertools
import string;
import re;
import random
import math
import numpy as np
from gensim.models import Word2Vec
import pickle as pkl
from itertools import *
embed_d = 128
window_s = 5


def load_data(path="./paper_abstract.pkl", maxlen = None, n_words = 600000, sort_by_len = False):
    # f = open(path, 'r')
    # content_set = pkl.load(f)
    # f.close()
    with open(path, 'rb') as f:
        content_set = pkl.load(f)

    # def remove_unk(x):
    #     return [[1 if w >= n_words else w for w in sen] for sen in x]

    content_set_x, content_set_y = content_set

    # content_set_x = remove_unk(content_set_x)
    content_set_x = [[1 if w >= n_words else w for w in sen] for sen in content_set_x]

    # def len_argsort(seq):
    #     return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        # sorted_index = len_argsort(content_set_x)
        sorted_index = sorted(range(len(content_set_x)), key = lambda x: len(content_set_x[x]))
        content_set_x = [content_set_x[i] for i in sorted_index]

    return content_set_x


def word2vec():
    # generate word embedding file: word_embeddings.txt
    sentences = load_data()   
    w2v_model = Word2Vec(sentences = sentences, window = window_s, min_count = 0, vector_size = embed_d)
    # w2v_model.train(sentences, epochs = 10, total_examples = len(sentences))
    w2v_model.wv.save_word2vec_format('word_embeddings.txt', binary = False)

    return w2v_model

word2vec()








