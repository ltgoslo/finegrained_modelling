#!/bin/bash
#
#SBATCH --job-name=bert_classification --account=nn9447k
#SBATCH --output=bert_classification-%j.out
#
#SBATCH --partition=accel --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --array=35-51
#
#
# Max memory usage:
#SBATCH --ntasks-per-node=2 --nodes=1
#SBATCH --mem-per-cpu=8G

## Set up job environtment:
# source /cluster/bin/jobsetup
module restore system  # clear any inherited modules
set -o errexit # exit on errors

# List imported modules
module list

# Load models
module load Python/3.6.6-fosscuda-2018b

# load virtual environment
source ../OLD/doc_level_transfer_hierarchical/my_virtualenv/bin/activate

#language array
DATASETS=([1]="darmstadt/services" [2]="darmstadt/universities" [3]="jiang" [4]="mitchell" [5]="mpqa" [6]="semeval/Laptop" [7]="semeval/Restaurant" [8]="wang"
    [9]="darmstadt/services" [10]="darmstadt/universities" [11]="mpqa"
    [12]="darmstadt/services" [13]="darmstadt/universities" [14]="mpqa"
    [15]="darmstadt/services" [16]="darmstadt/universities" [17]="mpqa" [18]="darmstadt/services" [19]="darmstadt/universities" [20]="jiang" [21]="mitchell" [22]="mpqa" [23]="semeval/Laptop" [24]="semeval/Restaurant" [25]="wang"
    [26]="darmstadt/services" [27]="darmstadt/universities" [28]="mpqa"
    [29]="darmstadt/services" [30]="darmstadt/universities" [31]="mpqa"
    [32]="darmstadt/services" [33]="darmstadt/universities" [34]="mpqa" [35]="darmstadt/services" [36]="darmstadt/universities" [37]="jiang" [38]="mitchell" [39]="mpqa" [40]="semeval/Laptop" [41]="semeval/Restaurant" [42]="wang"
    [43]="darmstadt/services" [44]="darmstadt/universities" [45]="mpqa"
    [46]="darmstadt/services" [47]="darmstadt/universities" [48]="mpqa"
    [49]="darmstadt/services" [50]="darmstadt/universities" [51]="mpqa")

DOMAIN_TUNED_MODELS=([1]="darmstadt/services" [2]="darmstadt/universities" [3]="jiang" [4]="mitchell"
     [5]="mpqa" [6]="laptop" [7]="restaurant" [8]="wang"
    [9]="darmstadt/services" [10]="darmstadt/universities" [11]="mpqa"
    [12]="darmstadt/services" [13]="darmstadt/universities" [14]="mpqa"
    [15]="darmstadt/services" [16]="darmstadt/universities" [17]="mpqa" [18]="darmstadt/services" [19]="darmstadt/universities" [20]="jiang" [21]="mitchell" [22]="mpqa" [23]="laptop" [24]="restaurant" [25]="wang"
    [26]="darmstadt/services" [27]="darmstadt/universities" [28]="mpqa"
    [29]="darmstadt/services" [30]="darmstadt/universities" [31]="mpqa"
    [32]="darmstadt/services" [33]="darmstadt/universities" [34]="mpqa" [35]="darmstadt/services" [36]="darmstadt/universities" [37]="jiang" [38]="mitchell" [39]="mpqa" [40]="laptop" [41]="restaurant" [42]="wang"
    [43]="darmstadt/services" [44]="darmstadt/universities" [45]="mpqa"
    [46]="darmstadt/services" [47]="darmstadt/universities" [48]="mpqa"
    [49]="darmstadt/services" [50]="darmstadt/universities" [51]="mpqa")


ANN_TYPES=([1]="targets_polarity" [2]="targets_polarity" [3]="targets_polarity" [4]="targets_polarity" [5]="targets_polarity" [6]="targets_polarity" [7]="targets_polarity" [8]="targets_polarity"
    [9]="targets_expressions_polarity" [10]="targets_expressions_polarity" [11]="targets_expressions_polarity"
    [12]="targets_holders_polarity" [13]="targets_holders_polarity" [14]="targets_holders_polarity"
    [15]="full_polarity" [16]="full_polarity" [17]="full_polarity" [18]="targets_polarity" [19]="targets_polarity" [20]="targets_polarity" [21]="targets_polarity" [22]="targets_polarity" [23]="targets_polarity" [24]="targets_polarity" [25]="targets_polarity"
    [26]="targets_expressions_polarity" [27]="targets_expressions_polarity" [28]="targets_expressions_polarity"
    [29]="targets_holders_polarity" [30]="targets_holders_polarity" [31]="targets_holders_polarity"
    [32]="full_polarity" [33]="full_polarity" [34]="full_polarity" [35]="targets_polarity" [36]="targets_polarity" [37]="targets_polarity" [38]="targets_polarity" [39]="targets_polarity" [40]="targets_polarity" [41]="targets_polarity" [42]="targets_polarity"
    [43]="targets_expressions_polarity" [44]="targets_expressions_polarity" [45]="targets_expressions_polarity"
    [46]="targets_holders_polarity" [47]="targets_holders_polarity" [48]="targets_holders_polarity"
    [49]="full_polarity" [50]="full_polarity" [51]="full_polarity")

POOLING_TYPE=([1]="cls" [2]="cls" [3]="cls" [4]="cls" [5]="cls" [6]="cls" [7]="cls" [8]="cls" [9]="cls" [10]="cls" [11]="cls" [12]="cls" [13]="cls" [14]="cls" [15]="cls" [16]="cls" [17]="cls" [18]="first" [19]="first" [20]="first" [21]="first" [22]="first" [23]="first" [24]="first" [25]="first" [26]="first" [27]="first" [28]="first" [29]="first" [30]="first" [31]="first" [32]="first" [33]="first" [34]="first" [35]="pooled" [36]="pooled" [37]="pooled" [38]="pooled" [39]="pooled" [40]="pooled" [41]="pooled" [42]="pooled" [43]="pooled" [44]="pooled" [45]="pooled" [46]="pooled" [47]="pooled" [48]="pooled" [49]="pooled" [50]="pooled" [51]="pooled")
#POOLING_TYPE=([1]="max" [2]="max" [3]="max" [4]="max" [5]="max" [6]="max" [7]="max" [8]="max" [9]="max" [10]="max" [11]="max" [12]="max" [13]="max" [14]="max" [15]="max" [16]="max" [17]="max" [18]="max-min-mean" [19]="max-min-mean" [20]="max-min-mean" [21]="max-min-mean" [22]="max-min-mean" [23]="max-min-mean" [24]="max-min-mean" [25]="max-min-mean" [26]="max-min-mean" [27]="max-min-mean" [28]="max-min-mean" [29]="max-min-mean" [30]="max-min-mean" [31]="max-min-mean" [32]="max-min-mean" [33]="max-min-mean" [34]="max-min-mean")

i=$SLURM_ARRAY_TASK_ID
dataset=${DATASETS[$i]}
domain_tuned=${DOMAIN_TUNED_MODELS[$i]}
ann_type=${ANN_TYPES[$i]}
pooling=${POOLING_TYPE[$i]}

TRAIN_FILE=data/processed/"$dataset"/"$ann_type"/train.conll
DEV_FILE=data/processed/"$dataset"/"$ann_type"/dev.conll
TEST_FILE=data/processed/"$dataset"/"$ann_type"/test.conll
OUTPUT="new_classification_results/$ann_type"/$dataset


SEEDS=([1]="12345" [2]="23456" [3]="34567" [4]="45678" [5]="56789")

# 5 runs
for run in 1 2 3 4 5; do
    seed=${SEEDS[$run]}
    python polarity_classification.py --train_data="$TRAIN_FILE" --dev_data="$DEV_FILE" --test_data="$TEST_FILE" --bert_model=bert-base-cased --output_dir="$OUTPUT"/"$pooling"/"$run" --pretrained_model_dir=finetuned_models/"$domain_tuned" --do_train --do_valid --do_eval --num_train_epochs 50 --seed "$seed" --pooling "$pooling"
done;

