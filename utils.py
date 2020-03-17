import numpy as np
import re
import pandas
import itertools
from collections import Counter
import torch
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
import shutil

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


def load_data_from_disk():
    # Load dataset from file
    dataset = pandas.read_csv('/home/yyang2/data/yyang2/pycharm/char-cnn-text-classification-tensorflow-master/data/reviews.csv', encoding='utf-8', names=['comments', 'label'])

    # Split by words
    X = [clean_str(sentence) for sentence in dataset['comments']]
    X = [list(sentence) for sentence in X]
    Y = [[0, 1] if (label == 'positive') else [1, 0] for label in dataset['label']]

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
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """

    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))

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


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def sentence_to_index(sentence, vocabulary, maxlen):
    sentence = clean_str(sentence)
    raw_input = [list(sentence)]
    sentences_padded = pad_sentences(raw_input, maxlen=maxlen)
    raw_x, dummy_y = build_input_data(sentences_padded, [0], vocabulary)
    return raw_x


def load_data():
    sentences, labels = load_data_from_disk()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)

    return [x, y, vocabulary, vocabulary_inv]


def train_step(train_loader, model, criterion, optimizer):
    model.train()
    y_tru = None
    y_p = None
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        if y_tru is None:
            y_tru = np.array(target)
        else:
            y_tru = np.append(y_tru, np.array(target))

        images = images.cuda().long()
        target = target.cuda().long()

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        y_pre = output.argmax(dim=1).detach().cpu().numpy()

        if y_p is None:
            y_p = np.array(y_pre)
        else:
            y_p = np.append(y_p, np.array(y_pre))
    y_tru = y_tru.reshape((-1))
    y_p = y_p.reshape((-1))
    kappa = cohen_kappa_score(y_tru, y_p)
    acc_t = accuracy_score(y_tru, y_p)
    confu = confusion_matrix(y_tru, y_p)
    print('训练的kappa：', kappa, '-----acc:', acc_t, '-----loss:', loss)
    print('confusion_matrix:\n', confu)
    return acc_t, kappa


def validate_step(val_loader, model, criterion):
    # switch to evaluate mode
    model.eval()
    y_tru = None
    y_p = None
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):

            if y_tru is None:
                y_tru = np.array(target)
            else:
                y_tru = np.append(y_tru, np.array(target))
            images = images.cuda().long()
            target = target.cuda().long()

            # compute output
            output = model(images)
            loss = criterion(output, target)

            y_pre = output.argmax(dim=1).detach().cpu().numpy()
            if y_p is None:
                y_p = np.array(y_pre)
            else:
                y_p = np.append(y_p, np.array(y_pre))
        # TODO: this should also be done with the ProgressMeter

    kappa = cohen_kappa_score(y_tru, y_p)
    acc_t = accuracy_score(y_tru, y_p)
    confu = confusion_matrix(y_tru, y_p)
    print('测试的kappa：', kappa, '-----acc:', acc_t, '-----loss:', loss)
    print('confusion_matrix:\n', confu)
    return acc_t, kappa


def save_checkpoint_step(state, is_best,
                    filename='/home/yyang2/data/yyang2/Data/char_cnn/checkpoint_jd_9_over.pth.tar',
                         filebest='/home/yyang2/data/yyang2/Data/char_cnn/model_best_jd_9_over.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filebest)
        print('保存完成')