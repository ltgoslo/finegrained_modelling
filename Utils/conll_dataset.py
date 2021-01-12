from collections import defaultdict
import os

import torch
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data.dataloader import default_collate

import torchtext
from torchtext.datasets import SST
import string

class Split(object):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def pack_words(self, ws):
        return pack_sequence(ws)

    def collate_fn(self, batch):
        batch = sorted(batch, key=lambda item : len(item[0]), reverse=True)

        words = pack_sequence([w for w,_ in batch])
        targets = default_collate([t for _,t in batch])

        return words, targets


class ConllDataset(object):
    def __init__(self, vocab, lower_case, data_dir="../data/processed/"):
        #
        self.vocab = vocab
        self.splits = {}
        #
        for name in ["train", "dev", "test"]:
            filename = os.path.join(data_dir, name) + ".conll"
            self.splits[name] = self.open_data(filename, lower_case)

        self.label2idx, self.idx2label = self.get_labels()
        #
        for name in ["train", "dev", "test"]:
            self.splits[name] = self.open_splits(self.splits[name])
        #
    def get_labels(self):
        label2idx = {}

        for i in self.splits["train"]:
            for label in i.label:
                if label not in label2idx:
                    label2idx[label] = len(label2idx)
        idx2label = dict([(i, w) for w, i in label2idx.items()])
        return label2idx, idx2label

    def tag2idx(self, labels):
        return [self.label2idx[i] for i in labels]

    def open_data(self, data_file, lower_case):
        text = torchtext.data.Field(lower=lower_case, include_lengths=True, batch_first=True)
        label = torchtext.data.Field(include_lengths=True, batch_first=True)

        examples = []
        t, l = [], []

        for line in open(data_file):
            if line.strip() == "":
                e = torchtext.data.example.Example.fromlist([t, l], fields=[("text", text), ("label", label)])
                examples.append(e)
                t, l = [], []
            elif line.startswith("#"):
                pass
            else:
                tok, lab = line.strip().split("\t")
                t.append(tok)
                l.append(lab)

        data = torchtext.data.Dataset(examples, fields=[("text", text),
                                                        ("label", label)])
        return data

    def open_splits(self, data):

        data_split = [(torch.LongTensor(self.vocab.ws2ids(item.text)),
                       torch.LongTensor(self.tag2idx(item.label))) for item in data]
        return data_split
        #return examples
    #
    def get_split(self, name):
        return Split(self.splits[name])
