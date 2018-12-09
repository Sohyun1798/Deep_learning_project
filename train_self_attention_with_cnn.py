import argparse
import json
import os
import sys

import numpy as np
import sklearn.metrics
import torch
from torch import optim

from lib.suggested.self_attention_with_cnn import SelfAttentionCnnClassifier
from lib.embedding import load_full_embedding_with_vocab
from lib.reader import WikiqaReader, filtered_ref_generator
from lib.train import train_model, get_label_score
from lib.transformer import NoamOpt


def main(config_path):
    with open(config_path, 'r') as fread:
        config_dict = json.load(fread)

    # path
    path_config = config_dict['Path']
    model_dir = path_config['model_dir']
    train = path_config['train']
    dev = path_config['dev']
    dev_ref = path_config['dev_ref']
    test = path_config['test']
    test_ref = path_config['test_ref']

    # dataset
    dataset_config = config_dict['Dataset']
    batch_size = dataset_config['batch_size']

    print('Loading train data...')
    train_reader = WikiqaReader(train, PAD_TOKEN='<pad>')
    dev_reader = WikiqaReader(dev, PAD_TOKEN='<pad>')

    words_embed, words_vocab = load_full_embedding_with_vocab(path_config['embed_dir'])
    vocabs = {'q_words': words_vocab, 'a_words': words_vocab}
    train_reader.set_vocabs(vocabs)
    dev_reader.set_vocabs(vocabs)

    train_iterator = train_reader.get_dataset_iterator(batch_size, train=True)
    dev_iterator = dev_reader.get_dataset_iterator(batch_size, train=False, sort=False)

    test_reader = WikiqaReader(test, PAD_TOKEN='<pad>')
    test_reader.set_vocabs(vocabs)
    test_iterator = test_reader.get_dataset_iterator(batch_size, train=False, sort=False)

    # model
    model_config = config_dict['Model']
    conv_width = model_config['conv_width']
    out_channels = model_config['out_channels']
    hidden_size = model_config['hidden_size']
    cuda_device = model_config['cuda_device']
    dropout = model_config['dropout']
    h = model_config['h']

    clf = SelfAttentionCnnClassifier(words_embed=words_embed, out_channels=out_channels,
                conv_width=conv_width, hidden_size=hidden_size, cuda_device=cuda_device,
                h=h, dropout=dropout)

    # train
    train_config = config_dict['Train']
    num_epoch = train_config['epoch']
    weight_decay = train_config['weight_decay']
    lr = train_config['lr']
    early_stopping = train_config['early_stopping']
    factor = train_config['factor']
    warmup = train_config['warmup']

    input_names = ['q_words', 'a_words']

    # optimizer = optim.Adam(clf.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-5)
    optimizer = NoamOpt(clf.len_embed, factor, warmup,
            optim.Adam(clf.parameters(), lr=0, weight_decay=weight_decay, eps=1e-5))

    if cuda_device is not None:
        clf.cuda(device=cuda_device)

    def callback(verbose=True):
        train_labels, train_scores = get_label_score(clf, train_iterator, cuda_device, 'label', input_names=input_names)
        train_predicts = train_scores.argmax(axis=-1)
        train_scores = train_scores[:, 1]
        if verbose:
            print('train_acc: %.2f' % sklearn.metrics.accuracy_score(train_labels, train_predicts))
            print('train_precision: %.2f' % sklearn.metrics.precision_score(train_labels, train_predicts))
            print('train_average_precision: %.2f' % sklearn.metrics.average_precision_score(train_labels, train_scores))

        dev_labels, dev_scores = get_label_score(clf, dev_iterator, cuda_device, 'label', input_names=input_names)
        dev_predicts = dev_scores.argmax(axis=-1)
        dev_scores = dev_scores[:, 1]
        if verbose:
            print('dev_acc: %.2f' % sklearn.metrics.accuracy_score(dev_labels, dev_predicts))
            print('dev_precision: %.2f' % sklearn.metrics.precision_score(dev_labels, dev_predicts))
            print('dev_average_precision: %.2f' % sklearn.metrics.average_precision_score(dev_labels, dev_scores))

        index = 0
        dev_aps = [] # for mean average precision score
        rrs = [] # for mean reciprocal rank score

        for query_labels in filtered_ref_generator(dev_ref):
            query_scores = dev_scores[index:index+len(query_labels)]
            index += len(query_labels)

            dev_aps.append(sklearn.metrics.average_precision_score(query_labels, query_scores))
            query_rel_best = np.argmin( -query_scores * query_labels)
            rrs.append(1 / (np.argsort(np.argsort(-query_scores))[query_rel_best] + 1))

        if verbose:
            print('dev_MAP: %.2f' % np.mean(dev_aps))
            print('dev_MRR: %.2f' % np.mean(rrs))

        test_labels, test_scores = get_label_score(clf, test_iterator, cuda_device, 'label', input_names=input_names)
        test_predicts = test_scores.argmax(axis=-1)
        test_scores = test_scores[:, 1]
        if verbose:
            print('test_acc: %.2f' % sklearn.metrics.accuracy_score(test_labels, test_predicts))
            print('test_precision: %.2f' % sklearn.metrics.precision_score(test_labels, test_predicts))
            print('test_average_precision: %.2f' % sklearn.metrics.average_precision_score(test_labels, test_scores))

        index = 0
        test_aps = []  # for mean average precision score
        rrs = []  # for mean reciprocal rank score

        for query_labels in filtered_ref_generator(test_ref):
            query_scores = test_scores[index:index + len(query_labels)]
            index += len(query_labels)

            test_aps.append(sklearn.metrics.average_precision_score(query_labels, query_scores))
            query_rel_best = np.argmin(-query_scores * query_labels)
            rrs.append(1 / (np.argsort(np.argsort(-query_scores))[query_rel_best] + 1))

        if verbose:
            print('test_MAP: %.2f' % np.mean(test_aps))
            print('test_MRR: %.2f' % np.mean(rrs))
            
        return np.mean(dev_aps)

    print('Training...')
    best_state_dict = train_model(clf, optimizer, train_iterator, label_name='label',
                num_epoch=num_epoch, cuda_device=cuda_device, early_stopping=early_stopping,
                input_names=input_names, callback=callback)
    print()

    if best_state_dict is not None:
        clf.load_state_dict(best_state_dict)

    torch.save(clf.state_dict(), os.path.join(model_dir, './net.pt'))

    print('Testing...')

    test_labels, test_scores = get_label_score(clf, test_iterator, cuda_device, 'label', input_names=input_names)
    test_predicts = test_scores.argmax(axis=-1)
    test_scores = test_scores[:, 1]
    
    print('test_acc: %.2f' % sklearn.metrics.accuracy_score(test_labels, test_predicts))
    print('test_precision: %.2f' % sklearn.metrics.precision_score(test_labels, test_predicts))
    print('test_average_precision: %.2f' % sklearn.metrics.average_precision_score(test_labels, test_scores))

    index = 0
    aps = []  # for mean average precision score
    rrs = []  # for mean reciprocal rank score

    for query_labels in filtered_ref_generator(test_ref):
        query_scores = test_scores[index:index + len(query_labels)]
        index += len(query_labels)

        aps.append(sklearn.metrics.average_precision_score(query_labels, query_scores))
        query_rel_best = np.argmin(-query_scores * query_labels)
        rrs.append(1 / (np.argsort(np.argsort(-query_scores))[query_rel_best] + 1))

    print('test_MAP: %.4f' % np.mean(aps))
    print('test_MRR: %.4f' % np.mean(rrs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True, type=str, help='Path of config file')

    args = parser.parse_args()
    main(args.config_path)

