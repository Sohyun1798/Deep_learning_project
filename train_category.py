import argparse
import json
import os

import numpy as np
import sklearn.metrics
import torch
from torch import optim
import torch.nn.functional as F

from lib.embedding import load_embedding
from lib.reader import UIUCReader
from lib.baseline.category import BaselineCategoryClassifier


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
            print('category: %d %s' % (batch.category[i], dataset_loader.dataset.fields['category'].vocab.itos[batch.category[i]]))
            for key in ['words']:
                print('%s:' % key)
                print(getattr(batch, key)[i])
                print('%s (decode):' % key)
                row = getattr(batch, key)[i]
                print('\t'.join([dataset_loader.dataset.fields[key].vocab.itos[w] for w in row]))
                print()
            print()


def test_accuracy(clf, test_reader, cuda_device):
    clf.eval()
    categories = []
    predicts = []
    sents = []
    for batch in test_reader.get_dataset_iterator(batch_size=1, train=False, sort=False):
        category = batch.category
        words = batch.words

        outputs = clf(words)
        if cuda_device is None:
            predict = np.argmax(outputs.detach().numpy(), axis=-1)
        else:
            predict = np.argmax(outputs.detach().cpu().numpy(), axis=-1)

        categories.append(category.numpy())
        predicts.append(predict)
        sents.append(words.numpy())

    categories = np.concatenate(categories)
    predicts = np.concatenate(predicts)
    sents = np.concatenate(sents)
    acc = sklearn.metrics.accuracy_score(categories, predicts)
    return acc, categories, predicts, sents


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
    vocabs = train_reader.get_vocabs() # will be used to test time

    # model
    model_config = config_dict['Model']
    embed = load_embedding(model_config['embed'], vocabs['words'].stoi)
    pad_size = dataset_config['pad_size']
    conv_widths = model_config['conv_widths']
    hidden_size = model_config['hidden_size']
    out_channels = model_config['out_channels']
    cuda_device = model_config['cuda_device']
    out_size = len(train_reader.get_vocab('category'))

    clf = BaselineCategoryClassifier(embed=embed, out_channels=out_channels,
                                     cuda_device=cuda_device,
                                    conv_widths=conv_widths, pad_size=pad_size,
                                     hidden_size=hidden_size, out_size=out_size)

    # train
    train_config = config_dict['Train']
    num_epoch = train_config['epoch']
    weight_decay = train_config['weight_decay']
    lr = train_config['lr']

    optimizer = optim.Adam(clf.parameters(), lr=lr, weight_decay=weight_decay)
    if cuda_device is not None:
        clf.cuda(device=cuda_device)

    print('Training...')
    for epoch in range(num_epoch):
        clf.train()
        total_loss = []
        for batch in train_reader.get_dataset_iterator(batch_size):
            category = batch.category.cuda()
            words = batch.words.cuda()

            optimizer.zero_grad()
            outputs = clf(words)

            loss = F.cross_entropy(outputs, category)
            total_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        print('epoch %d / loss %.3f' % (epoch+1, np.mean(total_loss)))
        acc, _ ,_ ,_ = test_accuracy(clf, train_reader, cuda_device)
        print('train_accuracy: %.3f' % acc)
    print()

    torch.save(clf.state_dict(), os.path.join(model_dir, './net.pt'))

    # test
    print('Loading test data...')
    test_reader = UIUCReader(test, PAD_TOKEN='<pad>', **dataset_config)
    test_reader.set_vocabs(vocabs)

    print('Testing...')
    acc, categories, predicts, sents = test_accuracy(clf, test_reader, cuda_device)
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