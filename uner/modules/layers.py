import sys

import json
import numpy as np
import torch
import torch.nn as nn


def load_vocab(vocab_file):
    idx = 0
    word2id = {}
    id2word = {}
    for line in open(vocab_file):
        word = line.strip()
        word2id[word] = idx
        id2word[idx] = word
        idx += 1
    return word2id, id2word


class Embedding(nn.Module):

    def __init__(self, config):
        super(Embedding, self).__init__()
        self.vocab_file = config['vocab']
        self.word2id, self.id2word = load_vocab(self.vocab_file)
        self.vocab_size = config['vocab_size']
        assert self.vocab_size == len(self.word2id)

        self.width = config['width']
        embedding = np.zeros((self.vocab_size, self.width))
        if 'embedding_file' in config:
            for line in open(config['embedding_file']):
                fields = line.strip().split(' ')
                if len(fields) == 2:
                    continue
                assert self.width == len(fields) - 1
                if fields[0] in self.word2id:
                    idx = self.word2id[fields[0]]
                    word_embed = [float(x) for x in fields[1:]]
                    embedding[idx] = word_embed

            self.embedding = nn.Embedding.from_pretrained(
                torch.tensor(embedding, dtype=torch.float32), freeze=False)
        else:
            self.embedding = nn.Embedding(self.vocab_size, self.width)

    def forward(self, input_ids):
        return self.embedding(input_ids)
