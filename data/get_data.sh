#!/bin/bash

# First download the mpqa 2.0 data from http://mpqa.cs.pitt.edu/corpora/mpqa_corpus/mpqa_corpus_2_0 and change the following path to point to the tar file
mpqa_tar_file="./mpqa_2_0_database.tar.gz"
tar -xvf $mpqa_tar_file

# download darmstadt data
wget https://www.informatik.tu-darmstadt.de/media/ukp/data/fileupload_2/DarmstadtServiceReviewCorpus.zip

# download semeval data
git init SemEval
cd SemEval
git remote add -f origin https://github.com/jiangqn/aspect_extraction
git config core.sparseCheckout true
echo "/data/official_data/*" >> .git/info/sparse-checkout
git pull origin master
cd ..

# download sentihood
git init sentihood
cd sentihood
git remote add -f origin https://github.com/uclnlp/jack
git config core.sparseCheckout true
echo "/data/sentihood/*" >> .git/info/sparse-checkout
git pull origin master
cd ..

# download mitchell et al (2013) Open Domain Targeted Sentiment
wget www.m-mitchell.com/code/MitchellEtAl-13-OpenSentiment.tgz


# download wang, et al. (2017) Multi-target-specific sentiment recognition on twitter
wget -O wangetal.zip https://ndownloader.figshare.com/articles/4479563/versions/1

# # download dong et al. (2014) Adaptive recursive neural networkfor target-dependent twitter sentiment classification
# git init dong-et-al
# cd dong-et-al
# git remote add -f origin https://github.com/bluemonk482/tdparse
# git config core.sparseCheckout true
# echo "/data/lidong/*" >> .git/info/sparse-checkout
# git pull origin master
# cd ..


# download Jiang et al. (2019) A Challenge Dataset and Effective Models for Aspect-Based Sentiment Analysis
git init jiang-et-al
cd jiang-et-al
git remote add -f origin https://github.com/siat-nlp/MAMS-for-ABSA
git config core.sparseCheckout true
echo "/data/*" >> .git/info/sparse-checkout
git pull origin master
cd ..
