import os
import pickle

import numpy as np
import argparse

import torch
from torch.nn import Embedding
from torchtext.vocab import Vocab
from collections import Counter


def main(vocab_dir, embed_path):
    with open(embed_path, 'rb') as fread:
        vocab_size, embed_size = map(int, fread.readline().strip().split())
        embed = np.zeros((vocab_size + 2, embed_size)) # <unk>, <pad> added
        embed_stoi = {}

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

            embed_stoi[word] = i + 2
            embed[i + 2] = np.fromstring(fread.read(binary_len), dtype='float32')

    vocab = Vocab(Counter(list(embed_stoi)), specials=['<unk>', '<pad>'], specials_first=True)

    embed_torch = Embedding(vocab_size + 2, embed_size)
    for idx, key in enumerate(vocab.itos):
        if embed_stoi.get(key, 0) != 0:
            embed_torch.weight[idx] = torch.from_numpy(embed[embed_stoi[key]])
        else:
            embed_torch.weight[idx] = torch.from_numpy(np.zeros(embed_size))

    with open(os.path.join(vocab_dir, 'words'), 'wb') as fwrite:
        pickle.dump(vocab, fwrite)

    torch.save(embed_torch.state_dict(), os.path.join(vocab_dir, 'embed.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_dir', type=str, required=True)
    parser.add_argument('--embed_path', type=str, required=True)

    args = parser.parse_args()
    vocab_dir = args.vocab_dir
    embed_path = args.embed_path

    main(vocab_dir, embed_path)