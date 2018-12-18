import json
import os
import pickle

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.embedding import load_full_embedding_with_vocab
from lib.train import BaseClassifier


class BaselineFocusClassifier(BaseClassifier):
    def __init__(self, words_embed=None, len_vocab=0, len_embed=0, cuda_device=None,
                 conv_width=10, out_channels=100, hidden_size=100, num_filters=4):
        super(BaselineFocusClassifier, self).__init__(cuda_device=cuda_device,
                    words_embed=words_embed, len_vocab=len_vocab, len_embed=len_embed)


        self.conv1 = nn.Conv1d(self.len_embed, out_channels, conv_width, padding=conv_width//2)
        self.convs = nn.ModuleList([nn.Conv1d(out_channels, out_channels, conv_width, padding=conv_width//2)
                      for _ in range(num_filters-1)])
        self.hidden1 = nn.Linear(out_channels, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, 1)

    @classmethod
    def load(cls, config_path):
        with open(config_path, 'r') as fread:
            config_dict = json.load(fread)

        path_config = config_dict['Path']
        model_config = config_dict['Model']

        words_embed, words_vocab = load_full_embedding_with_vocab(model_config['embed_dir'])

        model = cls(words_embed=words_embed, out_channels=model_config['out_channels'],
                    conv_width=model_config['conv_width'], hidden_size=model_config['hidden_size'],
                    cuda_device=model_config['cuda_device'], num_filters=model_config['num_filters'])

        model.load_state_dict(torch.load(os.path.join(path_config['model_dir'], 'net.pt')))
        model.eval()
        return model

    def forward(self, *input):
        words = input[0]

        if self.cuda_device is not None:
            words = words.cuda(self.cuda_device)

        sentence_matrix = self.embed(words)

        conv_out = self.conv1(sentence_matrix.transpose(2,1))
        for conv in self.convs:
            conv_out = conv(conv_out)

        hidden1_out = self.hidden1(conv_out.transpose(2,1))
        hidden1_out = F.relu(hidden1_out)

        hidden2_out = self.hidden2(hidden1_out)
        return hidden2_out.squeeze(-1)


