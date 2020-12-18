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
        # outputs, _ = self.lstms(outputs)
        # hidden = [batch size, num layers * num directions,hid dim]
        # cell = [batch size, num layers * num directions,hid dim]
        outputs, (hidden, cell) = self.lstms(outputs)
        # concat the final forward and backward hidden state
        # outputs = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=0)
        # if self.training:
        #     outputs = self.dropout(outputs)
        # hidden = [batch size, hid dim * num directions]
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


class F1(nn.Module):
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    '''
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true,):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, 2).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)
        
        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2* (precision*recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        return 1 - f1.mean()