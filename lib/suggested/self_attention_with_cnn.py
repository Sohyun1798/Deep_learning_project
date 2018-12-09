import numpy as np
import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.embedding import load_full_embedding_with_vocab
from lib.train import BaseClassifier
from lib.transformer import MultiHeadedAttention, PositionwiseFeedForward, PositionalEncoding


class SelfAttentionCnnClassifier(BaseClassifier):
    def __init__(self, cuda_device=None, words_embed=None, len_vocab=0, len_embed=0,
                 out_channels=100, conv_width=5, hidden_size=200, out_size=2, dropout=0.5,
                 h=8):
        super(SelfAttentionCnnClassifier, self).__init__(cuda_device=cuda_device,
                    words_embed=words_embed, len_vocab=len_vocab, len_embed=len_embed)

        self.pos_enc = PositionalEncoding(self.len_embed, dropout)

        self.q_mha = MultiHeadedAttention(h=h, d_model=self.len_embed, dropout=dropout)
        self.q_layer_norms = nn.ModuleList([nn.LayerNorm([self.len_embed]) for _ in range(2)])

        self.a_mha = MultiHeadedAttention(h=h, d_model=self.len_embed, dropout=dropout)
        self.a_layer_norms = nn.ModuleList([nn.LayerNorm([self.len_embed]) for _ in range(2)])

        self.q_conv = nn.Conv1d(self.len_embed, out_channels, conv_width, padding=(conv_width//2))
        self.a_conv = nn.Conv1d(self.len_embed, out_channels, conv_width, padding=(conv_width//2))

        self.hidden1 = nn.Linear(out_channels * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden2 = nn.Linear(hidden_size, out_size)

        if self.cuda_device is not None:
            self.q_mha = self.q_mha.cuda(self.cuda_device)
            self.q_layer_norms = self.q_layer_norms.cuda(self.cuda_device)
            self.a_mha = self.a_mha.cuda(self.cuda_device)
            self.a_layer_norms = self.a_layer_norms.cuda(self.cuda_device)
            self.q_conv = self.q_conv.cuda(self.cuda_device)
            self.a_conv = self.a_conv.cuda(self.cuda_device)
            self.hidden1 = self.hidden1.cuda(self.cuda_device)
            self.dropout = self.dropout.cuda(self.cuda_device)
            self.hidden2 = self.hidden2.cuda(self.cuda_device)

        nn.init.kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')

    @classmethod
    def load(cls, config_path):
        with open(config_path, 'r') as fread:
            config_dict = json.load(fread)

        path_config = config_dict['Path']
        model_config = config_dict['Model']

        words_embed, words_vocab = load_full_embedding_with_vocab(path_config['embed_dir'])

        model = cls(words_embed=words_embed, out_channels=model_config['out_channels'],
                    conv_width=model_config['conv_width'], hidden_size=model_config['hidden_size'],
                    cuda_device=model_config['cuda_device'], dropout=model_config['dropout'],
                    h=model_config['h'])

        model.load_state_dict(torch.load(os.path.join(path_config['model_dir'], 'net.pt')))
        model.eval()
        return model

    def forward(self, *input):
        if self.cuda_device is not None:
            input = [data.cuda(self.cuda_device) for data in input]

        # words: (batch size, sentence length)
        q_words, a_words = input

        # mask: (batch size, 1, sentence length)
        # if word is special pad token (so have to be ignored in multi-head attention) or not
        q_mask = (q_words != 1).unsqueeze(-2)
        a_mask = (a_words != 1).unsqueeze(-2)

        # sent_mat: (batch size, sentence length, embedding dimension)
        # we added Layer Normalization layer after embedding
        q_sent_mat = self.q_layer_norms[0](self.embed(q_words))
        a_sent_mat = self.a_layer_norms[0](self.embed(a_words))

        # mha_out: (batch size, sentence length, embedding dimension)
        # multi-head attention output
        # self.q_mha = MultiHeadedAttention(h=h, d_model=self.len_embed, dropout=dropout)
        q_mha_out = self.q_mha(q_sent_mat, a_sent_mat, a_sent_mat, a_mask)
        q_mha_out = self.q_layer_norms[1](q_mha_out + q_sent_mat)

        a_mha_out = self.a_mha(a_sent_mat, q_sent_mat, q_sent_mat, q_mask)
        a_mha_out = self.a_layer_norms[1](a_mha_out + a_sent_mat)

        # conv_out: (batch size, out channels, sentence length)
        q_conv_out = self.q_conv(q_mha_out.transpose(2,1))
        # pool out: (batch size, out_channels)
        # we used max over time pooling
        q_pool_out, _ = torch.max(q_conv_out, dim=-1)

        a_conv_out = self.a_conv(a_mha_out.transpose(2, 1))
        a_pool_out, _ = torch.max(a_conv_out, dim=-1)

        # qa_repr: (batch size, out channels * 2)
        # joined representation
        qa_repr = torch.cat([q_pool_out, a_pool_out], dim=-1)

        # hidden layer and softmax layer
        hidden1_out = self.hidden1(qa_repr)
        hidden2_out = self.hidden2(self.dropout(F.relu(hidden1_out)))

        return hidden2_out
