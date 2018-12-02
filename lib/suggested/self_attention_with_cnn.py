import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.embedding import load_full_embedding_with_vocab
from lib.train import BaseClassifier
from lib.transformer import MultiHeadedAttention, PositionwiseFeedForward


class SelfAttentionCnnClassifier(BaseClassifier):
    def __init__(self, cuda_device=None, words_embed=None, len_vocab=0, len_embed=0,
                 out_channels=100, conv_width=5, hidden_size=200, out_size=2, dropout=0.5,
                 h=8, d_ff=128):
        super(SelfAttentionCnnClassifier, self).__init__(cuda_device=cuda_device,
                    words_embed=words_embed, len_vocab=len_vocab, len_embed=len_embed)

        self.q_mha1 = MultiHeadedAttention(h=h, d_model=self.len_embed, dropout=dropout)
        self.q_mha2 = MultiHeadedAttention(h=h, d_model=self.len_embed, dropout=dropout)
        self.q_ffn = PositionwiseFeedForward(d_model=self.len_embed, d_ff=d_ff, dropout=dropout)
        self.q_layer_norms = [nn.LayerNorm([self.len_embed]) for _ in range(3)]

        self.a_mha1 = MultiHeadedAttention(h=h, d_model=self.len_embed, dropout=dropout)
        self.a_mha2 = MultiHeadedAttention(h=h, d_model=self.len_embed, dropout=dropout)
        self.a_ffn = PositionwiseFeedForward(d_model=self.len_embed, d_ff=d_ff, dropout=dropout)
        self.a_layer_norms = [nn.LayerNorm([self.len_embed]) for _ in range(3)]

        self.q_conv = nn.Conv1d(self.len_embed, out_channels, conv_width, padding=(conv_width//2))
        self.a_conv = nn.Conv1d(self.len_embed, out_channels, conv_width, padding=(conv_width//2))

        self.hidden1 = nn.Linear(out_channels * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden2 = nn.Linear(hidden_size, out_size)

        if self.cuda_device is not None:
            self.q_mha1 = self.q_mha1.cuda(self.cuda_device)
            self.q_mha2 = self.q_mha2.cuda(self.cuda_device)
            self.q_ffn = self.q_ffn.cuda(self.cuda_device)
            self.q_layer_norms = [ln.cuda(self.cuda_device) for ln in self.q_layer_norms]
            self.a_mha1 = self.a_mha1.cuda(self.cuda_device)
            self.a_mha2 = self.a_mha2.cuda(self.cuda_device)
            self.a_ffn = self.a_ffn.cuda(self.cuda_device)
            self.a_layer_norms = [ln.cuda(self.cuda_device) for ln in self.a_layer_norms]
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
                    cuda_device=model_config['cuda_device'], dropout=model_config['dropout'],
                    h=model_config['h'], d_ff=model_config['d_ff'])

        model.load_state_dict(torch.load(os.path.join(path_config['model_dir'], 'net.pt')))
        model.eval()
        return model

    def forward(self, *input):
        if self.cuda_device is not None:
            input = [data.cuda(self.cuda_device) for data in input]

        q_words, a_words = input
        q_mask = (q_words < 2).unsqueeze(-2)
        a_mask = (a_words < 2).unsqueeze(-2)
        
        q_sent_mat = self.embed(q_words)
        q_mha1_out = self.q_mha1(q_sent_mat, q_sent_mat, q_sent_mat, q_mask)
        q_mha1_out = self.q_layer_norms[0](q_mha1_out + q_sent_mat)

        a_sent_mat = self.embed(a_words)
        a_mha1_out = self.a_mha1(a_sent_mat, a_sent_mat, a_sent_mat, a_mask)
        a_mha1_out = self.a_layer_norms[0](a_mha1_out + a_sent_mat)

        q_mha2_out = self.q_mha2(q_mha1_out, a_mha1_out, a_mha1_out, a_mask)
        q_mha2_out = self.q_layer_norms[1](q_mha2_out + q_mha1_out)
        q_ffn_out = self.q_ffn(q_mha2_out)
        q_ffn_out = self.q_layer_norms[2](q_ffn_out + q_mha2_out)
        
        q_conv_out = self.q_conv(q_ffn_out.transpose(2,1))
        q_pool_out, _ = torch.max(q_conv_out, dim=-1)

        a_mha2_out = self.a_mha2(a_mha1_out, q_mha1_out, q_mha1_out, q_mask)
        a_mha2_out = self.a_layer_norms[1](a_mha2_out + a_mha1_out)
        a_ffn_out = self.a_ffn(a_mha2_out)
        a_ffn_out = self.a_layer_norms[2](a_ffn_out + a_mha2_out)
        
        a_conv_out = self.a_conv(a_ffn_out.transpose(2, 1))
        a_pool_out, _ = torch.max(a_conv_out, dim=-1)

        qa_repr = torch.cat([q_pool_out, a_pool_out], dim=-1)

        hidden1_out = self.hidden1(qa_repr)
        hidden2_out = self.hidden2(F.relu(self.dropout(hidden1_out)))

        return hidden2_out
