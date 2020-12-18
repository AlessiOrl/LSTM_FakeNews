import abc
from itertools import chain, tee
import torch
import numpy as np
import gensim
from typing import Any, Dict, List, Sequence, Tuple, Iterable

class Dataset(torch.utils.data.Dataset):
    def __init__(self, fake_sent: List[Sequence[str]], true_sent: List[Sequence[str]]):
        self.bies_dict = {"F": 1, "T": 2}
        self.fake_sent = fake_sent

        self.true_sent = true_sent
        self.features, self.labels = [], []
        self.max_length = 0

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        # Select sample
        x = self.features[index]
        y = self.labels[index]
        return x, y

    def process_data(self) -> Tuple[list, List[Sequence[int]]]:
        """
        Load the dataset from files, building features and labels
        :return: list of features and labels
        """
        features, labels = self.load_vects()

        return features, labels

    def load_vects(self) -> Tuple[List[str], List[str]]:
        """
        Load features and labels from files
        :return: list of features and labels
        """
        features = [s for s in self.fake_sent] + [s for s in self.true_sent]
        labels = [0] * len(self.fake_sent) + [1] * len(self.true_sent)
        return features, labels


class DatasetLSTM(Dataset):
    def __init__(self, fake_sent: List, true_sent: List, word_vectors: gensim.models.word2vec.Word2Vec):
        super().__init__(fake_sent, true_sent)
        self.word_vectors = word_vectors
        # build the vocab from the w2v model
        self.vocab = self.vocab_from_w2v(self.word_vectors)
        self.features, self.labels = self.process_data()

    @staticmethod
    def vocab_from_w2v(word_vectors: gensim.models.word2vec.Word2Vec) -> Dict[str, int]:
        """
        Builds the vocab from the Word2Vec matrix
        :param word_vectors: trained Gensim Word2Vec model
        :return: a dictionary from token to int
        """
        vocab = {"<PAD>": 0, "<UNK>": 1}
        for index, word in enumerate(word_vectors.wv.index2word):
            vocab[word] = index + 2
        return vocab

    @staticmethod
    def generate_batch(
        batch, vocab: Dict[str, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate the batch for the DataLoader
        :param batch: batch to process
        :param vocab: Vocab token -> int
        :return: the sentence [words], labels to feed to the model
        """
        u_input_sentence = [DatasetLSTM.encode_sequence(b[0], vocab) for b in batch]
        max_len = len(max(u_input_sentence,key=len))
        d_input_sentence = [s + [0] * (max_len - len(s)) for s in u_input_sentence] 
        input_sentence = torch.tensor(d_input_sentence)
        labels = torch.tensor([b[1] for b in batch])
        return input_sentence, labels

    @staticmethod
    def encode_sequence(sentence: List[str], vocab: Dict) -> Sequence[int]:
        """
        Encode the sequence follwoing the vocab in input
        :return: the text in input encoded
        """
        return [vocab[word] if word in vocab else vocab["<UNK>"] for word in sentence]