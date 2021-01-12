import json
import nltk
import re

def get_bioul_target(opinion):
    try:
        text, idxs = opinion["target"]
        if text == "":
            return []
    # will throw exception if the opinion target is None type
    except TypeError:
        return []
    # get the beginning and ending indices
    if ";" in text:
        updates = []
        texts = text.split(";")
        idxs = idxs.split(";")
        for text, idx in zip(texts, idxs):
            bidx, eidx = idx.split(":")
            bidx = int(bidx)
            eidx = int(eidx)
            polarity = opinion["label"]
            target_tokens = text.split()
            label = "-targ-{0}".format(polarity)
            #
            tags = []
            if len(target_tokens) == 1:
                tags.append("U" + label)
            else:
                for i, token in enumerate(target_tokens):
                    if i == 0:
                        tags.append("B" + label)
                    elif i == len(target_tokens) - 1:
                        tags.append("L" + label)
                    else:
                        tags.append("I" + label)
            new_target = " ".join([tok + "/" + tag for tok, tag in zip(target_tokens, tags)])
            difference = len(new_target) - len(text)
            updates.append((bidx, eidx, new_target, difference))
        return updates
    else:
        bidx, eidx = idxs.split(":")
        bidx = int(bidx)
        eidx = int(eidx)
        polarity = opinion["label"]
        target_tokens = text.split()
        label = "-targ-{0}".format(polarity)
        #
        tags = []
        if len(target_tokens) == 1:
            tags.append("U" + label)
        else:
            for i, token in enumerate(target_tokens):
                if i == 0:
                    tags.append("B" + label)
                elif i == len(target_tokens) - 1:
                    tags.append("L" + label)
                else:
                    tags.append("I" + label)
        new_target = " ".join([tok + "/" + tag for tok, tag in zip(target_tokens, tags)])
        difference = len(new_target) - len(text)
        return [(bidx, eidx, new_target, difference)]

def get_bioul_expression(opinion):
    #
    try:
        text, idxs = opinion["expression"]
        if text == "":
            return []
    # will throw exception if the opinion expression is None type
    except TypeError:
        return []
    if ";" in text:
        updates = []
        texts = text.split(";")
        idxs = idxs.split(";")
        for text, idx in zip(texts, idxs):
            bidx, eidx = idx.split(":")
            bidx = int(bidx)
            eidx = int(eidx)
            polarity = opinion["label"]
            target_tokens = text.split()
            label = "-exp-{0}".format(polarity)
            #
            tags = []
            if len(target_tokens) == 1:
                tags.append("U" + label)
            else:
                for i, token in enumerate(target_tokens):
                    if i == 0:
                        tags.append("B" + label)
                    elif i == len(target_tokens) - 1:
                        tags.append("L" + label)
                    else:
                        tags.append("I" + label)
            new_target = " ".join([tok + "/" + tag for tok, tag in zip(target_tokens, tags)])
            difference = len(new_target) - len(text)
            updates.append((bidx, eidx, new_target, difference))
        return updates
    else:
        bidx, eidx = idxs.split(":")
        bidx = int(bidx)
        eidx = int(eidx)
        polarity = opinion["label"]
        target_tokens = text.split()
        label = "-exp-{0}".format(polarity)
        #
        tags = []
        if len(target_tokens) == 1:
            tags.append("U" + label)
        else:
            for i, token in enumerate(target_tokens):
                if i == 0:
                    tags.append("B" + label)
                elif i == len(target_tokens) - 1:
                    tags.append("L" + label)
                else:
                    tags.append("I" + label)
        new_target = " ".join([tok + "/" + tag for tok, tag in zip(target_tokens, tags)])
        difference = len(new_target) - len(text)
        return [(bidx, eidx, new_target, difference)]

def get_bioul_holder(opinion):
    #
    try:
        text, idxs = opinion["holder"]
        if text == "":
            return []
    # will throw exception if the opinion expression is None type
    except TypeError:
        return []
    if ";" in text:
        updates = []
        texts = text.split(";")
        idxs = idxs.split(";")
        for text, idx in zip(texts, idxs):
            bidx, eidx = idx.split(":")
            bidx = int(bidx)
            eidx = int(eidx)
            target_tokens = text.split()
            label = "-holder"
            #
            tags = []
            if len(target_tokens) == 1:
                tags.append("U" + label)
            else:
                for i, token in enumerate(target_tokens):
                    if i == 0:
                        tags.append("B" + label)
                    elif i == len(target_tokens) - 1:
                        tags.append("L" + label)
                    else:
                        tags.append("I" + label)
            new_target = " ".join([tok + "/" + tag for tok, tag in zip(target_tokens, tags)])
            difference = len(new_target) - len(text)
            updates.append((bidx, eidx, new_target, difference))
        return updates
    else:
        bidx, eidx = idxs.split(":")
        bidx = int(bidx)
        eidx = int(eidx)
        target_tokens = text.split()
        label = "-holder"
        #
        tags = []
        if len(target_tokens) == 1:
            tags.append("U" + label)
        else:
            for i, token in enumerate(target_tokens):
                if i == 0:
                    tags.append("B" + label)
                elif i == len(target_tokens) - 1:
                    tags.append("L" + label)
                else:
                    tags.append("I" + label)
        new_target = " ".join([tok + "/" + tag for tok, tag in zip(target_tokens, tags)])
        difference = len(new_target) - len(text)
        return [(bidx, eidx, new_target, difference)]


def replace_with_labels(text, bidx, eidx, new_span):
	b = text[:bidx]
	e = text[eidx:]
	return b + new_span + e

def update_text_with_target_expression_opinion_labels(text, opinions):
    new_text = text
    new_offset = 0

    labels = []
    for o in opinions:
        try:
            labels.extend(get_bioul_holder(o))
            labels.extend(get_bioul_target(o))
            labels.extend(get_bioul_expression(o))
        except:
            pass
    labels = sorted(set(labels))
    #print(labels)

    for bidx, eidx, new_span, diff in labels:
        bidx += new_offset
        eidx += new_offset
        new_text = replace_with_labels(new_text, bidx, eidx, new_span)
        new_offset += diff

    return new_text

def tag_all(sent):
    tagged = []
    for w in nltk.word_tokenize(sent):
        if not bool(re.search("/[BIOUL]-", w)):
            tagged.append(w + "/O")
        elif w is not "``":
            tagged.append(w)
    return tagged

#data_file = "processed/darmstadt/universities.json"
data_file = "processed/mpqa/train.json"
#data_file = "processed/mitchell/train.json"
#data_file = "processed/semeval/restaurant.json"
#data_file = "processed/wang/train.json"

with open(data_file) as o:
    data = json.load(o)

sents = []
for d in data:
    text = d["text"]
    opinions = d["opinions"]
    bioul_replaced = update_text_with_target_expression_opinion_labels(text, opinions)
    if "mitchell" in data_file or "wang" in data_file:
        sents.append(bioul_replaced)
    else:
        for sent in nltk.sent_tokenize(bioul_replaced):
            if bool(re.search("/[BIOUL]-targ", sent)):
                sents.append(sent)

