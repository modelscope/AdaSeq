# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download

from adaseq.metainfo import Embedders

from .base import EMBEDDERS, Embedder


@EMBEDDERS.register_module(module_name=Embedders.embedding)
class Embedding(Embedder):
    """A simple lookup table for word embedding.

    This module implements a simple lookup table that stores embeddings of a fixed dictionary and size.
    It is often used to store word embeddings and retrieve them using indices.

    It can be initialized in 2 ways:
        1) with a vocab file and a pretrained embedding file
        2) with a vocab file and embedding_dim

    The input to the module is a list of indices, and the output is the corresponding word embeddings.

    Args:
        model_name_or_path (str): Word embedding hf_repo_id or local_path.
        vocab_file (str): Vocabulary file name. Each line contains a word. Default: vocab.txt
        embedding_file (str): Embedding file name. Each line contains several rows separated by white space.
            The first row should be a word. The rest rows are floats representing the word's embedding vector.
        embedding_dim (int): Embedding dimension. Required when embedding_file is not given.
        padding_idx (int): If specified, the entries at padding_idx do not contribute to the gradient.
        freeze (bool): If True, the tensor does not get updated in the learning process.
            This works only when loading a pretrained embedding file.
    """

    def __init__(
        self,
        model_name_or_path: str,
        vocab_file: Optional[str] = 'vocab.txt',
        embedding_file: Optional[str] = 'embedding.txt',
        embedding_dim: int = 0,
        padding_idx: Optional[int] = None,
        freeze: bool = False,
    ) -> None:
        super().__init__()

        if not osp.isdir(model_name_or_path):
            local_path = snapshot_download(model_name_or_path)
        else:
            local_path = model_name_or_path

        vocab_path = osp.join(local_path, vocab_file)
        assert osp.exists(vocab_path)
        self._word2id, self._id2word = self._load_vocab(vocab_path)
        self._vocab_size = len(self._word2id)

        embedding_path = osp.join(local_path, embedding_file)
        if osp.exists(embedding_path):
            self._embedding_dim = self._infer_embedding_dim(embedding_path)
            weight = np.zeros((self._vocab_size, self._embedding_dim))
            for line in open(embedding_path):
                fields = line.strip().split(' ')
                if len(fields) == 2:
                    continue
                assert len(fields) - 1 == self._embedding_dim
                if fields[0] in self._word2id:
                    idx = self._word2id[fields[0]]
                    word_embed = [float(x) for x in fields[1:]]
                    weight[idx] = word_embed
            weight = torch.tensor(weight, dtype=torch.float32)
            self.embedding = nn.Embedding.from_pretrained(
                weight, padding_idx=padding_idx, freeze=freeze
            )
        else:
            assert embedding_dim > 0, 'must pass a `embedding_dim` if use random embedding'
            self._embedding_dim = embedding_dim
            self.embedding = nn.Embedding(
                self._vocab_size, self._embedding_dim, padding_idx=padding_idx
            )

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:  # noqa: D102
        return self.embedding(input_ids)

    def get_output_dim(self) -> int:
        """
        Get the output embedding dim.
        """
        return self._embedding_dim

    @property
    def embedding_dim(self) -> int:
        """Dimension of word embedding"""
        return self._embedding_dim

    @property
    def vocab_size(self) -> int:
        """Number of words in vocabulary"""
        return self._vocab_size

    @property
    def word2id(self) -> Dict[str, int]:
        """Mapping word to index"""
        return self._word2id

    @property
    def id2word(self) -> Dict[int, str]:
        """Mapping index to word"""
        return self._id2word

    @staticmethod
    def _load_vocab(vocab_file: str) -> Tuple[Dict[str, int], Dict[int, str]]:
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

    @staticmethod
    def _infer_embedding_dim(embedding_file: str) -> int:
        """Infer embedding dimension from embedding file

        Args:
            embedding_file (str): local embedding file path

        Returns:
            embedding_dim (int): embedding dimension
        """
        embedding_dim = 0
        with open(embedding_file) as f:
            for line in f:
                fields = line.strip().split(' ')
                if len(fields) == 2:
                    continue
                embedding_dim = len(fields) - 1
                break
        assert embedding_dim > 0
        return embedding_dim
