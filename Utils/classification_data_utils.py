# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team and authors from University of Illinois at Chicago.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import os
from collections import defaultdict
import random

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert import BertModel


class TargetPooledClassification(nn.Module):
    def __init__(self,
                 pretrained_model_dir,
                 hidden_dropout_prob,
                 num_labels,
                 pool_target="cls"
                 ):
        super(TargetPooledClassification, self).__init__()
        self.num_labels = num_labels
        self.pool_target = pool_target
        #
        self.bert = BertModel.from_pretrained(pretrained_model_dir)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size,
                                    num_labels)
        #
        #self.init_weights()
        #
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        target_indices=None
    ):
        #
        encoded_layers, _ = self.bert(input_ids,
                                      attention_mask=attention_mask,
                                      token_type_ids=token_type_ids
                                      )
        #
        sequence_output = encoded_layers[-1]
        sequence_output = sequence_output.squeeze(0)
        #
        # Get the pooled representation of the embeddings that correspond
        # to the target
        batch_size, sequence_length, embedding_dim = sequence_output.shape
        # target_mask.shape() == (batch_size, sequence_length)
        target_mask = torch.zeros((batch_size, sequence_length))
        # set the target indices to 1, everything else to 0
        for (bidx, eidx), mask in zip(target_indices, target_mask):
            # If self.pool_target is True, take all the target token embs
            if self.pool_target is "pooled":
                for i in range(bidx, eidx):
                    mask[i] = 1
            # Otherwise, just take the first embedding
            elif self.pool_target is "first":
                mask[bidx] = 1
            # Otherwise, use the [CLS] embedding
            else:
                mask[0] = 1
        target_mask = target_mask.unsqueeze(2) # (batch_size, sequence_length, 1)
        #
        # multiply sequence_output by target_mask to keep only target
        # embeddings and then take their mean pooled representation
        sequence_output = sequence_output * target_mask
        pooled_output = sequence_output.mean(dim=1)
        #
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        #
        outputs = (logits,)
        #
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = loss
        #
        return outputs

class ABSATokenizer(BertTokenizer):
    def subword_tokenize(self, tokens, labels): # for AE
        split_tokens, split_labels= [], []
        idx_map=[]
        for ix, token in enumerate(tokens):
            sub_tokens=self.wordpiece_tokenizer.tokenize(token)
            for jx, sub_token in enumerate(sub_tokens):
                split_tokens.append(sub_token)
                if labels[ix]=="B" and jx>0:
                    split_labels.append("I")
                else:
                    split_labels.append(labels[ix])
                idx_map.append(ix)
        return split_tokens, split_labels, idx_map

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id,
                 target_indices):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.target_indices = target_indices


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file):
        """Reads a json file for tasks in sentiment analysis."""
        with open(input_file) as f:
            return json.load(f)

    @classmethod
    def _read_conll(cls, input_file):
    #def read_conll(input_file):
        """Reads a conll type file for sentiment classification"""
        sents = []
        sent, labels = [], []
        for line in open(input_file):
            if line.startswith("# sent_id"):
                current_id = line.strip().split(" = ")[1]
            elif line.strip() == "":
                if len(sent) > 0:
                    sents.append((current_id, sent, labels))
                    sent, labels = [], []
            else:
                token, label = line.strip().split("\t")
                sent.append(token)
                labels.append(label)
        return sents

def get_targets_polarities(tokens, labels):
    labeled = []
    target = []
    polarity = "neutral"
    for token, label in zip(tokens, labels):
        if "B-targ" in label:
            target.append(token)
            polarity = label.split("-")[-1]
        elif "I-targ" in label:
            target.append(token)
        else:
            if len(target) > 0:
                labeled.append((" ".join(target), polarity))
                target = []
                polarity = "neutral"
    return labeled

def add_span_tags_to_text(tokens, labels):
    new_sent = []
    for i, (tok, label) in enumerate(zip(tokens, labels)):
        if "B-targ" in label:
            new_sent.append("[<<TARG]")
            new_sent.append(tok)
            if i+1 < len(labels):
                if "I-targ" not in labels[i+1]:
                    new_sent.append("[TARG>>]")
            else:
                new_sent.append("[TARG>>]")
        elif "I-targ" in label:
            new_sent.append(tok)
            if i+1 < len(labels):
                if "I-targ" not in labels[i+1]:
                    new_sent.append("[TARG>>]")
            else:
                new_sent.append("[TARG>>]")
        elif "B-holder" in label:
            new_sent.append("[<<HOLDER]")
            new_sent.append(tok)
            if i+1 < len(labels):
                if "I-holder" not in labels[i+1]:
                    new_sent.append("[HOLDER>>]")
            else:
                new_sent.append("[HOLDER>>]")
        elif "I-holder" in label:
            new_sent.append(tok)
            if i+1 < len(labels):
                if "I-holder" not in labels[i+1]:
                    new_sent.append("[HOLDER>>]")
            else:
                new_sent.append("[HOLDER>>]")
        elif "B-exp" in label:
            new_sent.append("[<<EXP]")
            new_sent.append(tok)
            if i+1 < len(labels):
                if "I-exp" not in labels[i+1]:
                    new_sent.append("[EXP>>]")
            else:
                new_sent.append("[EXP>>]")
        elif "I-exp" in label:
            new_sent.append(tok)
            if i+1 < len(labels):
                if "I-exp" not in labels[i+1]:
                    new_sent.append("[EXP>>]")
            else:
                new_sent.append("[EXP>>]")
        else:
            new_sent.append(tok)
    return " ".join(new_sent)

class AscProcessor(DataProcessor):
    """Processor for the SemEval Aspect Sentiment Classification."""

    def get_conll_examples(self, data_file, name):
        """See base class."""
        return self._create_examples(
            self._read_conll(data_file), name)

    def get_train_examples(self, data_dir, fn="train.conll"):
        """See base class."""
        return self._create_examples(
            self._read_conll(os.path.join(data_dir, fn)), "train")

    def get_dev_examples(self, data_dir, fn="dev.conll"):
        """See base class."""
        return self._create_examples(
            self._read_conll(os.path.join(data_dir, fn)), "dev")

    def get_test_examples(self, data_dir, fn="test.conll"):
        """See base class."""
        return self._create_examples(
            self._read_conll(os.path.join(data_dir, fn)), "test")

    def get_labels(self):
        """See base class."""
        return ["positive", "negative", "neutral", "conflict"]

    def _create_examples(self, lines, set_type):
        examples = []
        for idx, tokens, labels in lines:
            guid = "%s-%s" % (set_type, idx)
            text_b = add_span_tags_to_text(tokens, labels)
            targets = get_targets_polarities(tokens, labels)
            if len(targets) > 0:
                for target, label in targets:
                        examples.append(
                        InputExample(guid=guid, text_a=target,
                                     text_b=text_b, label=label))
        return examples

def find_target_indices(target, tokens):
    target_length = len(target)
    # The first occurence in tokens will be text_a, which is the target
    # by itself
    first_occurence = True
    for ind in (i for i, e in enumerate(tokens) if e == target[0]):
        if tokens[ind:ind + target_length] == target:
            if first_occurence:
                first_occurence = False
            else:
                return ind, ind + target_length

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, mode):
    """Loads a data file into a list of `InputBatch`s.""" #check later if we can merge this function with the SQuAD preprocessing
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        if mode!="ae":
            tokens_a = tokenizer.tokenize(example.text_a)
        else: #only do subword tokenization.
            tokens_a, labels_a, example.idx_map= tokenizer.subword_tokenize([token.lower() for token in example.text_a], example.label )

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        target_indices = find_target_indices(tokens_a, tokens)
        if target_indices is None:
            target_indices = (1, 1 + len(tokens_a))
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if mode!="ae":
            label_id = label_map[example.label]
        else:
            label_id = [-1] * len(input_ids) #-1 is the index to ignore
            #truncate the label length if it exceeds the limit.
            lb=[label_map[label] for label in labels_a]
            if len(lb) > max_seq_length - 2:
                lb = lb[0:(max_seq_length - 2)]
            label_id[1:len(lb)+1] = lb

        features.append(
                InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_id,
                        target_indices=target_indices))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
