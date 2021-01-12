#!/bin/bash

mkdir data
mkdir data/train
mkdir data/dev
mkdir data/test

mv finished_curations04_03_2019.tar.gz data
mv split_data.py data

cd data
tar xf finished_curations04_03_2019.tar.gz --strip-components 1
python3 split_data.py

