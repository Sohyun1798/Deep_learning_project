import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineCategoryClassifier(nn.Module):
    def __init__(self, embed=None, len_vocab=0, len_embed=0, out_channels=300,
                 conv_widths=[1,2,3], hidden_size=100, out_size=6, pad_size=60,
                 cuda_device=None):
        super(BaselineCategoryClassifier, self).__init__()

        self.cuda_device = cuda_device

        if embed is None:
            self.embed = nn.Embedding(len_vocab, len_embed)
        else:
            len_vocab, len_embed = embed.shape
            embed = torch.Tensor(embed)
            if self.cuda_device is not None:
                embed = torch.Tensor(embed).cuda(device=self.cuda_device)
            self.embed = nn.Embedding(len_vocab, len_embed, _weight=embed)

        if self.cuda_device is None:
            self.convs = [nn.Conv1d(1, out_channels, (width, len_embed)) for width in conv_widths]
        else:
            self.convs = [nn.Conv1d(1, out_channels, (width, len_embed)).cuda(self.cuda_device)
                        for width in conv_widths]
        self.hidden1 = nn.Linear(out_channels * len(self.convs), hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.hidden2 = nn.Linear(hidden_size, out_size)

        self.pad_size = pad_size

    def forward(self, words):
        if self.cuda_device is not None:
            words = words.cuda(self.cuda_device)

        sentence_matrix = self.embed(words)
        sentence_matrix = torch.unsqueeze(sentence_matrix, dim=1)

        conv_outs = []
        for conv in self.convs:
            conv_out = conv(sentence_matrix).squeeze(dim=3)
            pool_out, _ = torch.max(conv_out, dim=-1)
            # pool_out = F.max_pool1d(conv_out, kernel_size=self.pad_size)
            conv_outs.append(pool_out)

        conv_out = torch.cat(conv_outs, dim=1)
        conv_out = torch.squeeze(conv_out, dim=-1)

        hidden1_out = self.hidden1(conv_out)
        hidden2_out = self.hidden2(self.dropout(F.relu(hidden1_out)))
        softmax_out = F.softmax(hidden2_out, dim=-1)

        return softmax_out