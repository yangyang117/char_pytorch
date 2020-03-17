import torch
from torch import nn as nn

class CNNClassifier(nn.Module):
    def __init__(self, sequence_length, num_classes, vocab_size,
                 embedding_size, filter_sizes, num_filters):
        super(CNNClassifier,self).__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.num_filters = num_filters
        self.sequence_length = sequence_length
        self.filter_sizes = filter_sizes
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size,self.embedding_size)  # 词汇长度和每个单词对应向量的长度
        self.bottle0 = self.make_block(self.filter_sizes[0])
        self.bottle1 = self.make_block(self.filter_sizes[1])
        self.bottle2 = self.make_block(self.filter_sizes[2])
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(self.num_filters * len(self.filter_sizes), self.num_classes)


    def make_block(self, filter):
        return nn.Sequential(
            nn.Conv2d(1, self.num_filters, (filter,self.embedding_size)),
            nn.ReLU(self.num_filters),
            nn.MaxPool2d((self.sequence_length - filter + 1, 1))
        )


    def forward(self, x):
        x = self.embedding(x)
        n,h,w = x.shape
        x = x.reshape((n, 1, h, w))
        c0 = self.bottle0(x)
        c1 = self.bottle1(x)
        c2 = self.bottle2(x)
        out = torch.cat((c0, c1, c2), 1)
        out = out.reshape((-1, out.shape[1]))
        out = self.dropout(out)
        out = self.classifier(out)
        return  out

'''
词向量

# Embedding(num_embeddings,embedding_dim)
embedding = nn.Embedding(10, 56)  # 10个词，每个词2维
input = t.arange(0, 6).view(3, 2).long()  # 三个句子，每个句子有两个词

input = t.autograd.Variable(input)
output = embedding(input)
print(output.size())
print(embedding.weight.size())
'''






