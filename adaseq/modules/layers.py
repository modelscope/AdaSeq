# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


def load_vocab(vocab_file: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Load from vocab file

    Args:
        vocab_file (str): vocab file to be loaded

    Returns:
        word2id (Dict[str, int]): a dict mapping word to id
        id2word (Dict[int, str]): a dict mapping id to word
    """
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
    """A simple lookup table for word embedding.

    This module implements a simple lookup table that stores embeddings of a fixed dictionary and size.
    It is often used to store word embeddings and retrieve them using indices.
    The input to the module is a list of indices, and the output is the corresponding word embeddings.

    Args:
        config (Dict): a config dict containing
            vocab (str): vocab file
            vocab_size (int): size of vocabulary
    """

    def __init__(self, config: Dict):
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
                torch.tensor(embedding, dtype=torch.float32), freeze=False
            )
        else:
            self.embedding = nn.Embedding(self.vocab_size, self.width)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Retrieve word embeddings using indices

        Args:
            input_ids (torch.Tensor): a list of indices

        Returns:
            embeddings (torch.Tensor): the corresponding word embeddings.
        """
        return self.embedding(input_ids)
