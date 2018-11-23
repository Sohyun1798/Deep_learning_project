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

    # TODO here


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True, type=str, help='Path of config file')

    args = parser.parse_args()
    main(args.config_path)