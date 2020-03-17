# encoding: utf-8

import torch
from torch.utils.data import Dataset
import numpy as np
import re
import pandas
import itertools
from collections import Counter

import random
import pandas as pd
import pkuseg
import pickle

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_from_disk(file_name):
    # Load dataset from file
    # dataset = pandas.read_csv('/home/yyang2/data/yyang2/Data/交大NLP资料/data_cut.csv', encoding='utf-8', names=['comments', 'label'])
    dataset = pandas.read_csv(file_name)


    # Split by words
    seg = pkuseg.pkuseg()
    X = list(dataset['Analysis'].map(lambda x: seg.cut(str(x))))

    # X = [list(sentence) for sentence in X]
    Y = list(dataset['label_id'])
    # Y = list(dataset['Sex_id'])
    # Y = list(dataset['label'])
    cc = list(zip(X, Y))
    random.seed(0)
    random.shuffle(cc)
    X[:], Y[:] = zip(*cc)
    return [X, Y]


def pad_sentences(sentences, padding_word="<PAD/>", maxlen=0):
    """
    Pads all the sentences to the same length. The length is defined by the longest sentence.
     Returns padded sentences.
    """

    if maxlen > 0:
        sequence_length = maxlen
    else:
        sequence_length = max(len(s) for s in sentences)

    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)

        replaced_newline_sentence = []
        for char in list(sentence):
            if char == "\n":
                replaced_newline_sentence.append("<NEWLINE/>")
            elif char == " ":
                replaced_newline_sentence.append("<SPACE/>")
            else:
                replaced_newline_sentence.append(char)

        new_sentence = replaced_newline_sentence + [padding_word] * num_padding

        # new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences, sequence_length


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """

    # Build vocabulary
    word_counts = None
    for i in range(len(sentences)):
        if word_counts == None:
            word_counts = Counter(sentences[i])
        else:
            word_counts.update(sentences[i])

    # Map from index to word
    vocabulary_inv = [word[0] for word in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))

    # Map from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary
    """
    x = np.array([[vocabulary[word] if word in vocabulary else 0 for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


# def batch_iter(data, batch_size, num_epochs, shuffle=True):
#     """
#     Generates a batch iterator for a dataset.
#     """
#     data = np.array(data)
#     data_size = len(data)
#     num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
#     for epoch in range(num_epochs):
#         # Shuffle the data at each epoch
#         if shuffle:
#             shuffle_indices = np.random.permutation(np.arange(data_size))
#             shuffled_data = data[shuffle_indices]
#         else:
#             shuffled_data = data
#         for batch_num in range(num_batches_per_epoch):
#             start_index = batch_num * batch_size
#             end_index = min((batch_num + 1) * batch_size, data_size)
#             yield shuffled_data[start_index:end_index]
#
#
# def sentence_to_index(sentence, vocabulary, maxlen):
#     sentence = clean_str(sentence)
#     raw_input = [list(sentence)]
#     sentences_padded = pad_sentences(raw_input, maxlen=maxlen)
#     raw_x, dummy_y = build_input_data(sentences_padded, [0], vocabulary)
#     return raw_x

file_v_name = '/home/yyang2/data/yyang2/Data/交大NLP资料/dict.pkl'
# def save_obj(vocabulary, file_v_name ):
#     with open(file_v_name, 'wb') as f:
#         pickle.dump(vocabulary, f, pickle.HIGHEST_PROTOCOL)
# save_obj(vocabulary,file_v_name)

def load_obj(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)



def load_data(file_name):
    sentences, labels = load_data_from_disk(file_name)
    sentences_padded, sequence_length = pad_sentences(sentences,maxlen=400)
    vocabulary = load_obj(file_v_name)
    x, y = build_input_data(sentences_padded, labels, vocabulary)

    return [x, y, vocabulary, sequence_length]



class DatasetGenerator(Dataset):
    def __init__(self, data_raw, labels_raw,vocabulary, sequence_length, set_name='train'):
        self.data_raw, self.labels_raw,self.vocabulary, self.sequence_length = data_raw, labels_raw,vocabulary, sequence_length
        self.set_name = set_name

        self.data = self.data_raw
        self.labels = self.labels_raw
        #
        # if self.set_name == 'train':
        #     self.data = self.data_raw[:int(self.data_raw.shape[0]*0.8)]
        #     self.labels = self.labels_raw[:int(self.labels_raw.shape[0]*0.8)]
        # elif self.set_name == 'val':
        #     self.data = self.data_raw[int(self.data_raw.shape[0]*0.8):int(self.data_raw.shape[0]*0.9)]
        #     self.labels = self.labels_raw[int(self.labels_raw.shape[0]*0.8):int(self.labels_raw.shape[0]*0.9)]
        # else:
        #     self.data = self.data_raw[int(self.data_raw.shape[0] * 0.9):]
        #     self.labels = self.labels_raw[int(self.labels_raw.shape[0] * 0.9):]

    def __getitem__(self, index):
        data_1 = self.data[index]

        if self.set_name == 'train' or self.set_name == 'val':
            label = self.labels[index]
            return torch.LongTensor(data_1), label
        else:
            return torch.LongTensor(data_1)

    def __len__(self):
        return self.data.shape[0]





