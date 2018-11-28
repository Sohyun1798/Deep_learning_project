from abc import abstractmethod

import numpy as np
import sklearn.metrics

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseClassifier(nn.Module):
    def __init__(self, cuda_device=None, words_embed=None, len_vocab=0, len_embed=0):
        super(BaseClassifier, self).__init__()

        self.cuda_device = cuda_device

        if words_embed is None:
            self.len_vocab = len_vocab
            self.len_embed = len_embed
            self.embed = nn.Embedding(len_vocab, len_embed)
        else:
            self.len_vocab = words_embed.num_embeddings
            self.len_embed = words_embed.embedding_dim
            self.embed = words_embed
            if self.cuda_device is not None:
                self.embed.cuda()
            self.embed.weight.requires_grad = False # fixed embedding

    @abstractmethod
    def forward(self, *input):
        raise NotImplementedError('You should use child class')


def test_accuracy(clf, test_iterator, cuda_device, label_name, input_name='words',
                  return_info=False):
    clf.eval()
    labels = []
    predicts = []
    inputs = []

    if not isinstance(test_iterator, list):
        test_iterator = [test_iterator] # for cross-val

    for iterator in test_iterator:
        for batch in iter(iterator):
            label = getattr(batch, label_name)
            words = getattr(batch, input_name)

            outputs = clf(words)
            if cuda_device is None:
                predict = np.argmax(outputs.detach().numpy(), axis=-1)
            else:
                predict = np.argmax(outputs.detach().cpu().numpy(), axis=-1)

            labels.append(label.numpy())
            predicts.append(predict)
            inputs.append(words.numpy())

    labels = np.concatenate(labels)
    predicts = np.concatenate(predicts)
    inputs = np.concatenate(inputs)
    acc = sklearn.metrics.accuracy_score(labels, predicts)

    if return_info:
        return acc, labels, predicts, inputs
    else:
        return acc


def train_model(clf, optimizer, train_iterator, label_name, input_name='words',
                test_iterator=None, num_epoch=0, cuda_device=None, early_stopping=3,
                verbose=True):
    if num_epoch == 0 and early_stopping == 0:
        raise ValueError('if num_epoch == 0 and early_stopping == 0, trainig will run infinitely')

    if test_iterator is None:
        early_stopping = 0

    if num_epoch == 0:
        def inf_range():
            i = 0
            while True:
                yield i
                i += 1
        epoch_range = inf_range()
    else:
        epoch_range = range(num_epoch)

    if not isinstance(train_iterator, list): # for cross-val
        train_iterator = [train_iterator]

    patient = 0
    best_acc = 0

    for epoch in epoch_range:
        clf.train()
        total_loss = []
        for iterator in train_iterator:
            for batch in iter(iterator):
                label = getattr(batch, label_name)
                words = getattr(batch, input_name)

                if cuda_device is not None:
                    label = label.cuda()
                    words = words.cuda()

                optimizer.zero_grad()
                outputs = clf(words)

                loss = F.cross_entropy(outputs, label)
                total_loss.append(loss.item())
                loss.backward()
                optimizer.step()

        if verbose: print('epoch %d / loss %.3f' % (epoch+1, np.mean(total_loss)))
        train_acc = test_accuracy(clf, train_iterator, cuda_device, label_name=label_name)
        if verbose: print('train_accuracy: %.3f' % train_acc)

        if test_iterator is not None:
            test_acc = test_accuracy(clf, test_iterator, cuda_device, label_name=label_name)
            if verbose: print('test_accuracy: %.3f' % test_acc)

            if test_acc > best_acc:
                best_acc = test_acc
                patient = 0
            else:
                patient += 1

        if early_stopping != 0 and patient > early_stopping:
            if verbose: print('early_stopping')
            break

        print()