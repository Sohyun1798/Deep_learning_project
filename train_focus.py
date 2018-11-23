import argparse
import json
import os

import numpy as np
import sklearn.metrics
import torch
from torch import optim
import torch.nn.functional as F

from lib.embedding import load_full_embedding_with_vocab
from lib.reader import QFocusReader
from lib.baseline.focus import BaselineFocusClassifier


def test_dataset_iterator(dataset_loader, dataset_iterator):
    for i, batch in enumerate(dataset_iterator):
        if i > 3:
            break

        print('===== %d th batch =====' % i)
        print()
        print('batch info:')
        print(batch)
        print()
        for i in range(batch.batch_size):
            print('focus: %d' % (batch.focus[i]))
            for key in ['words']:
                print('%s:' % key)
                print(getattr(batch, key)[i])
                print('%s (decode):' % key)
                row = getattr(batch, key)[i]
                print('\t'.join([dataset_loader.dataset.fields[key].vocab.itos[w] for w in row]))
                print()
            print()


def test_accuracy(clf, test_iterator, cuda_device):
    clf.eval()
    focuses = []
    predicts = []
    sents = []
    for batch in iter(test_iterator):
        focus = batch.focus
        words = batch.words

        outputs = clf(words)
        if cuda_device is None:
            predict = np.argmax(outputs.detach().numpy(), axis=-1)
        else:
            predict = np.argmax(outputs.detach().cpu().numpy(), axis=-1)

        focuses.append(focus.numpy())
        predicts.append(predict)
        sents.append(words.numpy())

    focuses = np.concatenate(focuses)
    predicts = np.concatenate(predicts)
    sents = np.concatenate(sents)
    acc = sklearn.metrics.accuracy_score(focuses, predicts)
    return acc, focuses, predicts, sents



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
    pad_size = dataset_config['pad_size']
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
                                      pad_size=pad_size, hidden_size=hidden_size,
                                      num_filters=num_filters)
        optimizer = optim.Adam(clf.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-5)
        if cuda_device is not None:
            clf.cuda(device=cuda_device)

        # train
        clf.train()
        for epoch in range(num_epoch):
            total_loss = []
            for i, fold in enumerate(folds):
                if i == test_idx:
                    continue

                for batch in iter(fold):
                    focus = batch.focus
                    words = batch.words

                    if cuda_device is not None:
                        focus = focus.cuda()
                        words = words.cuda()

                    optimizer.zero_grad()
                    outputs = clf(words)

                    loss = F.cross_entropy(outputs, focus)
                    total_loss.append(loss.item())
                    loss.backward()
                    optimizer.step()

            print('epoch %d / loss %.3f' % (epoch + 1, np.mean(total_loss)))
        print()

        # test
        print('Testing...')
        acc, _, _, _ = test_accuracy(clf, folds[test_idx], cuda_device)
        print('test accuracy:', acc)
        fold_accs.append(acc)

    print()
    print('test accuracies:', fold_accs)
    print('mean accuracies:', np.mean(fold_accs))
    print()

    print('Final Training...')

    clf = BaselineFocusClassifier(words_embed=words_embed, out_channels=out_channels,
                                  cuda_device=cuda_device, conv_width=conv_width,
                                  pad_size=pad_size, hidden_size=hidden_size,
                                  num_filters=num_filters)
    optimizer = optim.Adam(clf.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-5)
    if cuda_device is not None:
        clf.cuda(device=cuda_device)

    # train
    clf.train()
    for epoch in range(num_epoch):
        total_loss = []

        for batch in iter(train_reader.get_dataset_iterator(batch_size)):
            focus = batch.focus
            words = batch.words

            if cuda_device is not None:
                focus = focus.cuda()
                words = words.cuda()

            optimizer.zero_grad()
            outputs = clf(words)

            loss = F.cross_entropy(outputs, focus)
            total_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        print('epoch %d / loss %.3f' % (epoch + 1, np.mean(total_loss)))

    torch.save(clf.state_dict(), os.path.join(model_dir, './net.pt'))
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True, type=str, help='Path of config file')

    args = parser.parse_args()
    main(args.config_path)