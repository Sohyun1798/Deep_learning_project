import numpy as np
import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.embedding import load_full_embedding_with_vocab
from lib.train import BaseClassifier
from lib.transformer import MultiHeadedAttention


class SelfAttentionOnlyClassifier(BaseClassifier):
    def __init__(self, cuda_device=None, words_embed=None, len_vocab=0, len_embed=0,
                 out_channels=100, hidden_size=200, out_size=2, dropout=0.5,
                 h=8, num_mhas=1):
        super(SelfAttentionOnlyClassifier, self).__init__(cuda_device=cuda_device,
                    words_embed=words_embed, len_vocab=len_vocab, len_embed=len_embed)

        self.q_mhas = nn.ModuleList([MultiHeadedAttention(h=h, d_model=self.len_embed, dropout=dropout) for _ in range(num_mhas)])
        self.q_layer_norms = nn.ModuleList([nn.LayerNorm([self.len_embed]) for _ in range(num_mhas + 1)])
        self.q_lin = nn.Linear(self.len_embed, out_channels)
        self.q_pool_mha = MultiHeadedAttention(h=h, d_model=self.len_embed, dropout=dropout)

        self.a_mhas = nn.ModuleList(
            [MultiHeadedAttention(h=h, d_model=self.len_embed, dropout=dropout) for _ in range(num_mhas)])
        self.a_layer_norms = nn.ModuleList([nn.LayerNorm([self.len_embed]) for _ in range(num_mhas + 1)])
        self.a_lin = nn.Linear(self.len_embed, out_channels)
        self.a_pool_mha = MultiHeadedAttention(h=h, d_model=self.len_embed, dropout=dropout)

        self.hidden1 = nn.Linear(out_channels * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden2 = nn.Linear(hidden_size, out_size)

    @classmethod
    def load(cls, config_path):
        with open(config_path, 'r') as fread:
            config_dict = json.load(fread)

        path_config = config_dict['Path']
        model_config = config_dict['Model']

        words_embed, words_vocab = load_full_embedding_with_vocab(path_config['embed_dir'])

        model = cls(words_embed=words_embed, out_channels=model_config['out_channels'], hidden_size=model_config['hidden_size'],
                    cuda_device=model_config['cuda_device'], dropout=model_config['dropout'],
                    h=model_config['h'], num_mhas=model_config['num_mhas'])

        model.load_state_dict(torch.load(os.path.join(path_config['model_dir'], 'net.pt')))
        model.eval()
        return model

    def forward(self, *input):
        if self.cuda_device is not None:
            input = [data.cuda(self.cuda_device) for data in input]

        # words: (batch size, sentence length + 1)
        # + 1 caused by special "<cls>" token
        q_words, a_words = input
        q_words = torch.cat((torch.zeros(q_words.size(0), 1).type_as(q_words), q_words), dim=-1)
        a_words = torch.cat((torch.zeros(a_words.size(0), 1).type_as(a_words), a_words), dim=-1)

        # mask: (batch size, 1, sentence length)
        # if word is special pad token (so have to be ignored in multi-head attention) or not
        q_mask = (q_words != 1).unsqueeze(-2)
        a_mask = (a_words != 1).unsqueeze(-2)

        # sent_mat: (batch size, sentence length, embedding dimension)
        # we added Layer Normalization layer after embedding
        q_sent_mat = self.q_layer_norms[0](self.embed(q_words))
        a_sent_mat = self.a_layer_norms[0](self.embed(a_words))

        # sent_mat: (batch size, sentence length, embedding dimension)
        # multi-head attention output
        # self.q_mha = MultiHeadedAttention(h=h, d_model=self.len_embed, dropout=dropout)
        for q_mha, a_mha, q_layer_norm, a_layer_norm in zip(self.q_mhas, self.a_mhas, self.q_layer_norms[1:], self.a_layer_norms[1:]):
            q_mha_out = q_mha(q_sent_mat, a_sent_mat, a_sent_mat, a_mask)
            q_sent_mat = q_layer_norm(q_mha_out + q_sent_mat)

            a_mha_out = a_mha(a_sent_mat, a_sent_mat, a_sent_mat, a_mask)
            a_sent_mat = a_layer_norm(a_mha_out + a_sent_mat)

        # q_out: (batch size, out channels)
        # instead of using convolutional layer and max over time pooling layer,
        # we used linear layer and multi-head attention layer on special "<cls>" token
        # for create latent vector representing sentence.
        q_mask[:, 0] = 0 # mask special "<cls>" token
        q_out = self.q_pool_mha(q_sent_mat[:, 0].unsqueeze(-2), q_sent_mat, q_sent_mat, q_mask).squeeze(-2)
        q_repr = self.q_lin(q_out)

        a_mask[:, 0] = 0  # mask special "<cls>" token
        a_out = self.a_pool_mha(a_sent_mat[:, 0].unsqueeze(-2), a_sent_mat, a_sent_mat, a_mask).squeeze(-2)
        a_repr = self.a_lin(a_out)

        # qa_repr: (batch size, out channels * 2)
        # joined representation
        qa_repr = torch.cat([q_repr, a_repr], dim=-1)

        # hidden layer and softmax layer
        hidden1_out = self.hidden1(qa_repr)
        hidden2_out = self.hidden2(self.dropout(F.relu(hidden1_out)))

        return hidden2_out
