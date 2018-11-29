import os
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

    def predict(self, *input):
        logits = self.forward(*input)
        scores = F.softmax(logits, dim=-1)

        if self.cuda_device is not None:
            scores = scores.detach().cpu().numpy()

        return np.argmax(scores, axis=-1)

def test_map(clf, test_iterator, cuda_device, label_name, input_names=['words']):
    clf.eval()
    labels = []
    probas = []

    if not isinstance(test_iterator, list):
        test_iterator = [test_iterator]  # for cross-val

    for iterator in test_iterator:
        for batch in iter(iterator):
            label = getattr(batch, label_name)
            inputs = []
            for input_name in input_names:
                inputs.append(getattr(batch, input_name))

            outputs = clf(*inputs)
            if cuda_device is None:
                prob = outputs.detach().numpy()[:, 1]
            else:
                prob = outputs.detach().cpu().numpy()[:, 1]

            labels.append(label.numpy())
            probas.append(prob)

    labels = np.concatenate(labels)
    probas = np.concatenate(probas)
    score = sklearn.metrics.average_precision_score(labels, probas)

    return score

def test_score(clf, test_iterator, cuda_device, label_name, input_names=['words'],
               return_info=False, metric=sklearn.metrics.accuracy_score):
    clf.eval()
    labels = []
    predicts = []
    total_inputs = []

    if not isinstance(test_iterator, list):
        test_iterator = [test_iterator] # for cross-val

    for iterator in test_iterator:
        for batch in iter(iterator):
            label = getattr(batch, label_name)
            inputs = []
            for input_name in input_names:
                inputs.append(getattr(batch, input_name))

            outputs = clf(*inputs)
            if cuda_device is None:
                predict = np.argmax(outputs.detach().numpy(), axis=-1)
            else:
                predict = np.argmax(outputs.detach().cpu().numpy(), axis=-1)

            labels.append(label.numpy())
            predicts.append(predict)

            if len(input_names) == 1:
                total_inputs.append(inputs[0].numpy())
            else:
                total_inputs.append([input_data.numpy() for input_data in inputs])

    labels = np.concatenate(labels)
    predicts = np.concatenate(predicts)
    score = metric(labels, predicts)

    if return_info:
        if len(input_names) == 1:
            return score, labels, predicts, np.concatenate(total_inputs)
        else:
            return score, labels, predicts, [example for batch in total_inputs for example in zip(*batch)]
    else:
        return score


def train_model(clf, optimizer, train_iterator, label_name, input_names=['words'],
                num_epoch=0, cuda_device=None, early_stopping=3, verbose=True, callback=None, **callback_kwargs):
    """
    :param label_name: name of label field
    :param input_names: names of input field
    :param callbacks: [(callback_func, callback_args_dict), ...]
        callback function can return dev_score to increase patient for early stopping
        callback function must get verbose as keyword argument
    :return: best_state_dict
    """
    if num_epoch == 0 and early_stopping == 0:
        raise ValueError('if num_epoch == 0 and early_stopping == 0, trainig will run infinitely')

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
    best_score = 0
    best_state_dict = None

    for epoch in epoch_range:
        clf.train()
        total_loss = []
        for iterator in train_iterator:
            for batch in iter(iterator):
                label = getattr(batch, label_name)

                inputs = []
                for input_name in input_names:
                    inputs.append(getattr(batch, input_name))

                if cuda_device is not None:
                    label = label.cuda()
                    inputs = [input_data.cuda() for input_data in inputs]

                optimizer.zero_grad()
                outputs = clf(*inputs)

                loss = F.cross_entropy(outputs, label)
                total_loss.append(loss.item())
                loss.backward()
                optimizer.step()

        if verbose: print('epoch %d / loss %.3f' % (epoch+1, np.mean(total_loss)))

        if callback is not None:
            dev_score = callback(verbose=verbose, **callback_kwargs)

            if dev_score > best_score:
                best_score = dev_score
                patient = 0
                best_state_dict = clf.state_dict()
            else:
                patient += 1

        if early_stopping != 0 and patient > early_stopping:
            if verbose: print('early_stopping')
            break

        print()

    return best_state_dict