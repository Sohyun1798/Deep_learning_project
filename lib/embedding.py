import os
import pickle

import numpy as np
import torch


def load_trimmed_embedding(embed_path, stoi):
    with open(embed_path, 'rb') as fread:
        vocab_size, embed_size = map(int, fread.readline().strip().split())
        embed = np.zeros((len(stoi), embed_size))
        words = []

        binary_len = np.dtype('float32').itemsize * embed_size
        for i in range(vocab_size):
            word = []
            while True:
                ch = fread.read(1)
                if ch == b' ':
                    word = b''.join(word).decode('utf-8')
                    break
                if ch != b'\n':
                    word.append(ch)

            words.append(word)
            if stoi.get(word) != 0:
                embed[stoi.get(word)] = np.fromstring(fread.read(binary_len), dtype='float32')
            else:
                fread.read(binary_len)

    return embed


def load_full_embedding_with_vocab(words_vocab_dir, embed_size=50):
    vocab_path = os.path.join(words_vocab_dir, 'words')
    with open(vocab_path, 'rb') as fread:
        words_vocab = pickle.load(fread)

    words_embed = torch.nn.Embedding(len(words_vocab), embed_size)
    words_embed.load_state_dict(torch.load(os.path.join(words_vocab_dir, 'embed.pt')))

    return words_embed, words_vocab
