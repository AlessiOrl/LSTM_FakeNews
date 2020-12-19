from argparse import ArgumentParser
from functools import partial

import gensim
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import DatasetLSTM
from typing import Any, Dict, List, Sequence, Tuple, Iterable
import torch.nn.functional as F


class FakenewsLSTM(nn.Module):
    def __init__(
        self, word_vectors, hidden_size=256, num_layers=1, freeze=True, *args, **kwargs
    ):
        super(FakenewsLSTM, self).__init__()
        # model definition
        self.word_vectors = word_vectors
        self.embeddings = self._get_embeddings_layer(self.word_vectors.vectors, freeze)
        self.lstms = nn.LSTM(
            self.word_vectors.vector_size,
            hidden_size,
            num_layers=num_layers,
            dropout=0.5 if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(hidden_size * 2, 1)

    def forward(self, sentence, *args, **kwargs):
        outputs = self.embeddings(sentence)

        outputs, (hidden, cell) = self.lstms(outputs)
        # concat the final forward and backward hidden state

        outputs = self.classifier(outputs[:, -1, :])
        return outputs


    @staticmethod
    def _get_embeddings_layer(weights, freeze: bool):
        # zero vector for pad, 1 in position 1
        # random vector for pad
        pad = np.random.rand(1, weights.shape[1])
        # mean vector for unknowns
        unk = np.mean(weights, axis=0, keepdims=True)
        weights = np.concatenate((pad, unk, weights))
        weights = torch.FloatTensor(weights)
        return nn.Embedding.from_pretrained(weights, padding_idx=0, freeze=freeze)

