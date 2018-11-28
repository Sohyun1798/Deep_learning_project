import os
import pickle
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.embedding import load_full_embedding_with_vocab
from lib.train import BaseClassifier


class BaselineAnswerSelectionClassifier(BaseClassifier):
    def __init__(self, cuda_device=None, words_embed=None, len_vocab=0, len_embed=0,
                 dim_word_over=5, len_sem_over=7, dim_sem_over=5,
                 out_channels=100, conv_width=5, hidden_size=200, out_size=2, dropout=0.5):
        super(BaselineAnswerSelectionClassifier, self).__init__(cuda_device=cuda_device,
                    words_embed=words_embed, len_vocab=len_vocab, len_embed=len_embed)

        self.word_over_embed = nn.Embedding(2, dim_word_over)
        self.sem_over_embed = nn.Embedding(len_sem_over, dim_sem_over)

        self.q_conv = nn.Conv1d(self.len_embed + dim_word_over + dim_sem_over,
                                out_channels, conv_width, padding=(conv_width//2))
        self.a_conv = nn.Conv1d(self.len_embed + dim_word_over + dim_sem_over,
                                out_channels, conv_width, padding=(conv_width // 2))

        self.hidden1 = nn.Linear(out_channels * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden2 = nn.Linear(hidden_size, out_size)

        if self.cuda_device is not None:
            self.word_over_embed = self.word_over_embed.cuda(self.cuda_device)
            self.sem_over_embed = self.sem_over_embed.cuda(self.cuda_device)
            self.q_conv = self.q_conv.cuda(self.cuda_device)
            self.a_conv = self.a_conv.cuda(self.cuda_device)
            self.hidden1 = self.hidden1.cuda(self.cuda_device)
            self.dropout = self.dropout.cuda(self.cuda_device)
            self.hidden2 = self.hidden2.cuda(self.cuda_device)

    @classmethod
    def load(cls, config_path):
        with open(config_path, 'r') as fread:
            config_dict = json.load(fread)

        path_config = config_dict['Path']
        model_config = config_dict['Model']

        words_embed, words_vocab = load_full_embedding_with_vocab(path_config['embed_dir'])

        model = cls(words_embed=words_embed, out_channels=model_config['out_channels'],
                    conv_width=model_config['conv_width'], hidden_size=model_config['hidden_size'],
                    cuda_device=model_config['cuda_device'])

        model.load_state_dict(torch.load(os.path.join(path_config['model_dir'], 'net.pt')))
        model.eval()
        return model

    def forward(self, *input):
        if self.cuda_device is not None:
            input = [data.cuda(self.cuda_device) for data in input]

        q_words, a_words, q_word_over, a_word_over, q_sem_over, a_sem_over = input

        def get_sentence_matrix(words, word_over, sem_over):
            sent_mat = torch.cat([
                self.embed(words),
                self.word_over_embed(word_over),
                self.sem_over_embed(sem_over)], dim=-1)
            return sent_mat.transpose(2,1)
        
        q_sentence_matrix = get_sentence_matrix(q_words, q_word_over, q_sem_over)
        q_conv_out = self.q_conv(q_sentence_matrix)
        q_pool_out, _ = torch.max(q_conv_out, dim=-1)

        a_sentence_matrix = get_sentence_matrix(a_words, a_word_over, a_sem_over)
        a_conv_out = self.a_conv(a_sentence_matrix)
        a_pool_out, _ = torch.max(a_conv_out, dim=-1)

        qa_repr = torch.cat([q_pool_out, a_pool_out], dim=-1)

        hidden1_out = self.hidden1(qa_repr)
        hidden2_out = self.hidden2(F.relu(self.dropout(hidden1_out)))

        return hidden2_out
