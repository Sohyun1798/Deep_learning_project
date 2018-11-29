import argparse
import json
import os
import pickle

import numpy as np
import sklearn.metrics
import torch
from torch import optim
import torch.nn.functional as F

from lib.baseline.focus import BaselineFocusClassifier
from lib.embedding import load_full_embedding_with_vocab
from lib.reader import WikiqaPairReader, test_dataset_iterator
from lib.baseline.category import BaselineCategoryClassifier
from lib.baseline.baseline import BaselineAnswerSelectionClassifier
from lib.train import test_score, train_model, test_map


def main(config_path):
    with open(config_path, 'r') as fread:
        config_dict = json.load(fread)

    # path
    path_config = config_dict['Path']
    model_dir = path_config['model_dir']
    train = path_config['train']
    dev = path_config['dev']
    test = path_config['test']
    test_result = path_config['test_result']

    print('Loading question analysis models...')
    category_model = BaselineCategoryClassifier.load(path_config['category_model_config'])
    focus_model = BaselineFocusClassifier.load(path_config['focus_model_config'])

    words_embed, words_vocab = load_full_embedding_with_vocab(path_config['embed_dir'])
    with open(path_config['category_vocab'], 'rb') as fread:
        category_vocab = pickle.load(fread)

    # dataset
    dataset_config = config_dict['Dataset']
    pad_size = dataset_config['pad_size']
    batch_size = dataset_config['batch_size']

    print('Loading train data...')
    train_reader = WikiqaPairReader(train, category_model, focus_model, words_vocab.stoi,
                                    category_vocab.itos, PAD_TOKEN='<pad>', pad_size=pad_size)
    dev_reader = WikiqaPairReader(dev, category_model, focus_model, words_vocab.stoi,
                                    category_vocab.itos, PAD_TOKEN='<pad>', pad_size=pad_size)
    vocabs = {'q_words': words_vocab, 'a_words': words_vocab}
    train_reader.set_vocabs(vocabs)
    dev_reader.set_vocabs(vocabs)

    # model
    model_config = config_dict['Model']
    conv_width = model_config['conv_width']
    out_channels = model_config['out_channels']
    hidden_size = model_config['hidden_size']
    cuda_device = model_config['cuda_device']

    clf = BaselineAnswerSelectionClassifier(words_embed=words_embed, out_channels=out_channels,
                conv_width=conv_width, hidden_size=hidden_size, cuda_device=cuda_device)

    # train
    train_config = config_dict['Train']
    num_epoch = train_config['epoch']
    weight_decay = train_config['weight_decay']
    lr = train_config['lr']
    early_stopping = train_config['early_stopping']

    optimizer = optim.Adam(clf.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-5)
    if cuda_device is not None:
        clf.cuda(device=cuda_device)

    print('Training...')
    train_model(clf, optimizer, train_reader.get_dataset_iterator(batch_size), label_name='label',
                test_iterator=dev_reader.get_dataset_iterator(batch_size, train=False, sort=False),
                num_epoch=num_epoch, cuda_device=cuda_device, early_stopping=early_stopping,
                input_names=['q_words', 'a_words', 'q_word_over', 'a_word_over', 'q_sem_over', 'a_sem_over'],
                score_name='map', model_path=os.path.join(model_dir, 'net.pt'), score_func=test_map)
    print()

    # torch.save(clf.state_dict(), os.path.join(model_dir, './net.pt'))

    # test
    print('Loading test data...')
    test_reader = WikiqaPairReader(test, category_model, focus_model, words_vocab.stoi,
                                  category_vocab.itos, PAD_TOKEN='<pad>', pad_size=pad_size)
    test_reader.set_vocabs(vocabs)

    print('Testing...')
    acc, labels, predicts, inputs = test_score(clf, test_reader.get_dataset_iterator(batch_size, train=False, sort=False),
                                               cuda_device, label_name='label', return_info=True,
                                               input_names=['q_words', 'a_words', 'q_word_over', 'a_word_over',
                                                                  'q_sem_over', 'a_sem_over'])
    print('test accuracy:', acc)
    print('test map:', test_map(clf, test_reader.get_dataset_iterator(batch_size, train=False, sort=False),
                                               cuda_device, label_name='label',
                                               input_names=['q_words', 'a_words', 'q_word_over', 'a_word_over',
                                                                  'q_sem_over', 'a_sem_over']))

    print('Writing test result...')
    with open(test_result, 'w') as fwrite:
        for label, predict, input_data in zip(labels, predicts, inputs):
            q_words, a_words, q_word_over, a_word_over, q_sem_over, a_sem_over = input_data
            fwrite.write('%d\t%d\t%s\t%s\n' %
                         (label, predict,
                          ' '.join([test_reader.get_vocab('q_words').itos[word] for word in q_words]),
                          ' '.join([test_reader.get_vocab('a_words').itos[word] for word in a_words]),))

    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True, type=str, help='Path of config file')

    args = parser.parse_args()
    main(args.config_path)

