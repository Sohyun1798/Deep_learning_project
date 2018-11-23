import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineFocusClassifier(nn.Module):
    def __init__(self, words_embed=None, len_vocab=0, len_embed=0, pad_size=60, cuda_device=None,
                 conv_width=10, out_channels=100, hidden_size=100, num_filters=4):
        super(BaselineFocusClassifier, self).__init__()

        self.cuda_device = cuda_device

        if words_embed is None:
            self.embed = nn.Embedding(len_vocab, len_embed)
        else:
            len_vocab = words_embed.num_embeddings
            len_embed = words_embed.embedding_dim
            self.embed = words_embed
            if self.cuda_device is not None:
                self.embed.cuda()
            self.embed.weight.requires_grad = False # fixed embedding

        if self.cuda_device is None:
            self.conv1 = nn.Conv1d(len_embed, out_channels, conv_width, padding=conv_width//2)
            self.convs = [nn.Conv1d(out_channels, out_channels, conv_width, padding=conv_width//2)
                          for _ in range(num_filters-1)]
        else:
            self.conv1 = nn.Conv1d(len_embed, out_channels, conv_width, padding=conv_width // 2).cuda(self.cuda_device)
            self.convs = [nn.Conv1d(out_channels, out_channels, conv_width, padding=conv_width // 2).cuda(self.cuda_device)
                          for _ in range(num_filters-1)]

        self.hidden1 = nn.Linear(out_channels, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, 1)

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
