import os
import pickle
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.embedding import load_full_embedding_with_vocab
from lib.train import BaseClassifier


class BaselineCategoryClassifier(BaseClassifier):
    def __init__(self, words_embed=None, len_vocab=0, len_embed=0, out_channels=300,
                 conv_widths=[1,2,3], hidden_size=100, out_size=6, cuda_device=None):
        super(BaselineCategoryClassifier, self).__init__(cuda_device=cuda_device,
                    words_embed=words_embed, len_vocab=len_vocab, len_embed=len_embed)

        self.convs = nn.ModuleList([nn.Conv1d(self.len_embed, out_channels, width, padding=(width//2))
                      for width in conv_widths])
        self.hidden1 = nn.Linear(out_channels * len(self.convs), hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.hidden2 = nn.Linear(hidden_size, out_size)


    @classmethod
    def load(cls, config_path):
        with open(config_path, 'r') as fread:
            config_dict = json.load(fread)

        path_config = config_dict['Path']
        model_config = config_dict['Model']

        with open(os.path.join(path_config['vocab_dir'], 'category'), 'rb') as fread:
            category_vocab = pickle.load(fread)

        out_size = len(category_vocab)
        words_embed, words_vocab = load_full_embedding_with_vocab(model_config['embed_dir'])

        model = cls(words_embed=words_embed, out_channels=model_config['out_channels'],
                    conv_widths=model_config['conv_widths'], hidden_size=model_config['hidden_size'],
                    out_size=out_size, cuda_device=model_config['cuda_device'])

        model.load_state_dict(torch.load(os.path.join(path_config['model_dir'], 'net.pt')))
        model.eval()
        return model

    def forward(self, *input):
        words = input[0]

        if self.cuda_device is not None:
            words = words.cuda(self.cuda_device)

        sentence_matrix = self.embed(words).transpose(2, 1)

        conv_outs = []
        for conv in self.convs:
            conv_out = conv(sentence_matrix)
            pool_out, _ = torch.max(conv_out, dim=-1)
            conv_outs.append(pool_out)

        conv_out = torch.cat(conv_outs, dim=-1)

        hidden1_out = self.hidden1(conv_out)
        hidden2_out = self.hidden2(F.relu(self.dropout(hidden1_out)))

        return hidden2_out
