import argparse
import json
import os

import numpy as np
import sklearn.metrics
import torch
from torch import optim
import torch.nn.functional as F

from lib.embedding import load_full_embedding_with_vocab
from lib.reader import QFocusReader, test_dataset_iterator
from lib.baseline.focus import BaselineFocusClassifier
from lib.train import test_score, train_model


def main(config_path):
    with open(config_path, 'r') as fread:
        config_dict = json.load(fread)

    path_config = config_dict['Path']
    model_dir = path_config['model_dir']
    vocab_dir = path_config['vocab_dir']
    train = path_config['train']

    # dataset
    dataset_config = config_dict['Dataset']
    pad_size = dataset_config['pad_size']
    batch_size = dataset_config['batch_size']

    print('Loading train data...')
    train_reader = QFocusReader(train, PAD_TOKEN='<pad>', pad_size=pad_size)
    train_reader.build_vocabs(vocab_dir)

    # model
    model_config = config_dict['Model']
    conv_width = model_config['conv_width']
    hidden_size = model_config['hidden_size']
    out_channels = model_config['out_channels']
    cuda_device = model_config['cuda_device']
    num_filters = model_config['num_filters']

    # load pretrained vocab
    words_embed, words_vocab = load_full_embedding_with_vocab(model_config['embed_dir'])
    train_reader.set_vocabs({'words': words_vocab})
    vocabs = train_reader.get_vocabs()  # will be used to test time

    train_config = config_dict['Train']
    num_epoch = train_config['epoch']
    weight_decay = train_config['weight_decay']
    lr = train_config['lr']
    kfold = train_config['kfold']

    # cross-val
    folds = train_reader.get_cross_val_dataset_iterator(batch_size=batch_size, k_fold=kfold)
    fold_accs = []
    for test_idx in range(kfold):
        clf = BaselineFocusClassifier(words_embed=words_embed, out_channels=out_channels,
                                      cuda_device=cuda_device, conv_width=conv_width,
                                      hidden_size=hidden_size, num_filters=num_filters)
        optimizer = optim.Adam(clf.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-5)
        if cuda_device is not None:
            clf.cuda(device=cuda_device)

        train_iterator = [fold for fold_idx, fold in enumerate(folds) if fold_idx != test_idx]
        train_model(clf, optimizer, train_iterator, num_epoch=num_epoch, cuda_device=cuda_device, early_stopping=0, label_name='focus')

        # test
        print('Testing...')
        acc = test_score(clf, folds[test_idx], cuda_device, label_name='focus')
        print('test accuracy:', acc)
        fold_accs.append(acc)

    print()
    print('test accuracies:', fold_accs)
    print('mean accuracies:', np.mean(fold_accs))
    print()

    print('Final Training...')

    clf = BaselineFocusClassifier(words_embed=words_embed, out_channels=out_channels,
                                  cuda_device=cuda_device, conv_width=conv_width,
                                  hidden_size=hidden_size, num_filters=num_filters)
    optimizer = optim.Adam(clf.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-5)
    if cuda_device is not None:
        clf.cuda(device=cuda_device)

    train_iterator = train_reader.get_dataset_iterator(batch_size)
    def callback(verbose=False):
        train_acc = test_score(clf, train_iterator, cuda_device, 'category', return_info=False)
        if verbose: print('train_acc: %.3f' % (train_acc))

    # train
    best_state_dict = train_model(clf, optimizer, train_iterator, num_epoch=num_epoch, cuda_device=cuda_device,
                                  label_name='focus', callback=callback)

    if best_state_dict is not None:
        clf.load_state_dict(best_state_dict)

    torch.save(clf.state_dict(), os.path.join(model_dir, './net.pt'))
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True, type=str, help='Path of config file')

    args = parser.parse_args()
    main(args.config_path)