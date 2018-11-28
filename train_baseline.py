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
from lib.train import test_accuracy, train_model


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

    print('Loading train data...') # Debugging
    train_reader = WikiqaPairReader(dev, category_model, focus_model, words_vocab.stoi,
                                    category_vocab.itos, PAD_TOKEN='<pad>', pad_size=pad_size)
    # train_reader.build_vocabs()
    train_reader.set_vocabs({'q_words': words_vocab, 'a_words': words_vocab})

    test_dataset_iterator(train_reader, train_reader.get_dataset_iterator(batch_size),
                          ['q_words', 'a_words', 'q_word_over', 'a_word_over', 'q_sem_over', 'a_sem_over', 'label'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True, type=str, help='Path of config file')

    args = parser.parse_args()
    main(args.config_path)

