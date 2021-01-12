#!/bin/bash

# only targets
for dataset in darmstadt/services darmstadt/universities jiang mitchell mpqa semeval/laptop semeval/restaurant wang; do
    mkdir processed/"$dataset"/targeted
    for split in train.json dev.json test.json; do
        data_file=processed/"$dataset"/"$split"
        split_name=${split[@]::-5}
        python3 convert_to_bioul.py -d processed/"$dataset"/$split -f only-tags -l targets -o processed/"$dataset"/targeted/"$split_name".conll
    done;
done;

# targets expressions
for dataset in darmstadt/services darmstadt/universities jiang mitchell mpqa semeval/laptop semeval/restaurant wang; do
    mkdir processed/"$dataset"/targets_exps
    for split in train.json dev.json test.json; do
        data_file=processed/"$dataset"/"$split"
        split_name=${split[@]::-5}
        python3 convert_to_bioul.py -d processed/"$dataset"/$split -f only-tags -l targets expressions -o processed/"$dataset"/targets_exps/"$split_name".conll
    done;
done;

# targets expressions holders
for dataset in darmstadt/services darmstadt/universities jiang mitchell mpqa semeval/laptop semeval/restaurant wang; do
    mkdir processed/"$dataset"/targets_exps_holders
    for split in train.json dev.json test.json; do
        data_file=processed/"$dataset"/"$split"
        split_name=${split[@]::-5}
        python3 convert_to_bioul.py -d processed/"$dataset"/$split -f only-tags -l targets expressions holders -o processed/"$dataset"/targets_exps_holders/"$split_name".conll
    done;
done;

# targets polarity
for dataset in darmstadt/services darmstadt/universities jiang mitchell mpqa semeval/laptop semeval/restaurant wang; do
    mkdir processed/"$dataset"/targets_polarity
    for split in train.json dev.json test.json; do
        data_file=processed/"$dataset"/"$split"
        split_name=${split[@]::-5}
        python3 convert_to_bioul.py -d processed/"$dataset"/$split -f full -l targets -o processed/"$dataset"/targets_polarity/"$split_name".conll
    done;
done;

# targets expressions polarity
for dataset in darmstadt/services darmstadt/universities jiang mitchell mpqa semeval/laptop semeval/restaurant wang; do
    mkdir processed/"$dataset"/targets_exps_polarity
    for split in train.json dev.json test.json; do
        data_file=processed/"$dataset"/"$split"
        split_name=${split[@]::-5}
        python3 convert_to_bioul.py -d processed/"$dataset"/$split -f full -l targets expressions -o processed/"$dataset"/targets_exps_polarity/"$split_name".conll
    done;
done;

# targets expressions holders polarity
for dataset in darmstadt/services darmstadt/universities jiang mitchell mpqa semeval/laptop semeval/restaurant wang; do
    mkdir processed/"$dataset"/targets_exps_holders_polarity
    for split in train.json dev.json test.json; do
        data_file=processed/"$dataset"/"$split"
        split_name=${split[@]::-5}
        python3 convert_to_bioul.py -d processed/"$dataset"/$split -f full -l targets expressions holders -o processed/"$dataset"/targets_exps_holders_polarity/"$split_name".conll
    done;
done;
