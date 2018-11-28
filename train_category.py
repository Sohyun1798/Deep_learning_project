import argparse
import json
import os

import numpy as np
import sklearn.metrics
import torch
from torch import optim
import torch.nn.functional as F

from lib.embedding import load_full_embedding_with_vocab
from lib.reader import UIUCReader, test_dataset_iterator
from lib.baseline.category import BaselineCategoryClassifier
from lib.train import test_score, train_model


def main(config_path):
    with open(config_path, 'r') as fread:
        config_dict = json.load(fread)

    path_config = config_dict['Path']
    model_dir = path_config['model_dir']
    vocab_dir = path_config['vocab_dir']
    train = path_config['train']
    test = path_config['test']
    test_result = path_config['test_result']

    # dataset
    dataset_config = config_dict['Dataset']
    pad_size = dataset_config['pad_size']
    batch_size = dataset_config['batch_size']

    print('Loading train data...')
    train_reader = UIUCReader(train, PAD_TOKEN='<pad>', pad_size=pad_size)
    train_reader.build_vocabs(vocab_dir)

    # model
    model_config = config_dict['Model']
    pad_size = dataset_config['pad_size']
    conv_widths = model_config['conv_widths']
    hidden_size = model_config['hidden_size']
    out_channels = model_config['out_channels']
    cuda_device = model_config['cuda_device']
    # cuda_device = None # debugging
    out_size = len(train_reader.get_vocab('category'))

    # load pretrained vocab
    words_embed, words_vocab = load_full_embedding_with_vocab(model_config['embed_dir'])
    train_reader.set_vocabs({'words': words_vocab})
    vocabs = train_reader.get_vocabs() # will be used to test time

    clf = BaselineCategoryClassifier(words_embed=words_embed, out_channels=out_channels,
                                     cuda_device=cuda_device, conv_widths=conv_widths,
                                     hidden_size=hidden_size, out_size=out_size)

    # train
    train_config = config_dict['Train']
    num_epoch = train_config['epoch']
    weight_decay = train_config['weight_decay']
    lr = train_config['lr']
    early_stopping = train_config['early_stopping']

    optimizer = optim.Adam(clf.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-5)
    if cuda_device is not None:
        clf.cuda(device=cuda_device)

    print('Loading test data...')
    test_reader = UIUCReader(test, PAD_TOKEN='<pad>', pad_size=pad_size)
    test_reader.set_vocabs(vocabs)

    print('Training...')
    train_model(clf, optimizer, train_reader.get_dataset_iterator(batch_size), label_name='category',
          test_iterator=test_reader.get_dataset_iterator(batch_size),
          num_epoch=num_epoch, cuda_device=cuda_device, early_stopping=early_stopping)
    print()

    torch.save(clf.state_dict(), os.path.join(model_dir, './net.pt'))

    # test
    print('Loading test data...')
    test_reader = UIUCReader(test, PAD_TOKEN='<pad>', pad_size=pad_size)
    test_reader.set_vocabs(vocabs)

    print('Testing...')
    acc, categories, predicts, sents = test_score(clf, test_reader.get_dataset_iterator(batch_size),
                                                  cuda_device, label_name='category', return_info=True)
    print('test accuracy:', acc)

    print('Writing test result...')
    with open(test_result, 'w') as fwrite:
        for category, predict, sent in zip(categories, predicts, sents):
            fwrite.write('%s\t%s\t%s\n' %
                         (test_reader.get_vocab('category').itos[category],
                          test_reader.get_vocab('category').itos[predict],
                          ' '.join([test_reader.get_vocab('words').itos[word] for word in sent])))

    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True, type=str, help='Path of config file')

    args = parser.parse_args()
    main(args.config_path)