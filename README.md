# finegrained_modelling
Datasets and models for fine-grained sentiment
==============

This repo contains a number of datasets for fine-grained sentiment (aspect-level and targeted sentiment) which have been converted to a single data format. Concretely, each dataset is kept as a json where entries have the following format:

for the full document:
---
"idx": unique identifier for document + sentence

"text": raw text

"opinions": list of all opinions in the sentence

for opinions:
---
"holder": a list of text and offsets for the opinion holder

"target": a list of text and offsets for the opinion target

"expression": a list of text and offsets for the opinion expression

"label": sentiment label ("negative", "neutral", "positive")

"intensity": sentiment intensity ("average", "strong", "weak")

```
{
    "idx": "my_example_id_001",
    "text": "I don't like this example text",
    "opinions":
               [
                  { "holder": ["I", "0:1"],
                    "target": ["this example text", "13:30"],
                    "expression": ["don't like", "2:12"],
                    "label": "negative",
                    "intensity": "average"
                  }
                ]

}
```

Note that for a single text, it is common to have many opinions. At the same time, it is common for many datasets to lack one of the elements of an opinion, e.g. the holder. In this case, the value for that element is None.


Requirements to run the experiments
--------
python3
lxml



Usage
--------

First, clone the repository

```
git clone https://github.uio.no/SANT/finegrained_modelling
```

Download the [MPQA 2.0 dataset](http://mpqa.cs.pitt.edu/corpora/mpqa_corpus/mpqa_corpus_2_0) as a tar file and put it in the /data directory.

Then run the following commands to download and preprocess the remaining datasets.

```
cd data
bash ./get_data.sh
bash ./process_data.sh
```


License
-------

Copyright (C) 2019, Jeremy Barnes

Licensed under the terms of the Creative Commons CC-BY public license
