#!/bin/bash
#
#SBATCH --job-name=domain_classification --account=nn9447k
#SBATCH --output=domain_classification-%j.out
#
#SBATCH --partition=accel --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --array=1-17
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
    [15]="darmstadt/services" [16]="darmstadt/universities" [17]="mpqa")

DOMAIN_TUNED_MODELS=([1]="darmstadt/services" [2]="darmstadt/universities" [3]="jiang" [4]="mitchell"
     [5]="mpqa" [6]="laptop" [7]="restaurant" [8]="wang"
    [9]="darmstadt/services" [10]="darmstadt/universities" [11]="mpqa"
    [12]="darmstadt/services" [13]="darmstadt/universities" [14]="mpqa"
    [15]="darmstadt/services" [16]="darmstadt/universities" [17]="mpqa")


ANN_TYPES=([1]="targets_polarity" [2]="targets_polarity" [3]="targets_polarity" [4]="targets_polarity" [5]="targets_polarity" [6]="targets_polarity" [7]="targets_polarity" [8]="targets_polarity"
    [9]="targets_expressions_polarity" [10]="targets_expressions_polarity" [11]="targets_expressions_polarity"
    [12]="targets_holders_polarity" [13]="targets_holders_polarity" [14]="targets_holders_polarity"
    [15]="full_polarity" [16]="full_polarity" [17]="full_polarity")

i=$SLURM_ARRAY_TASK_ID
dataset=${DATASETS[$i]}
domain_tuned=${DOMAIN_TUNED_MODELS[$i]}
ann_type=${ANN_TYPES[$i]}


TRAIN_FILE=data/processed/"$dataset"/"$ann_type"/train.conll
DEV_FILE=data/processed/"$dataset"/"$ann_type"/dev.conll

SEEDS=([1]="12345" [2]="23456" [3]="34567" [4]="45678" [5]="56789")

# 5 runs
for test_data in "${DATASETS[@]:1:8}"; do
    TEST_FILE=data/processed/"$test_data"/"$ann_type"/test.conll
    OUTPUT="domain_classification_results/$ann_type"/"$dataset"-"$test_data"
    for run in 1 2 3 4 5; do
        seed=${SEEDS[$run]}
        python polarity_classification.py --annotation_type="$ann_type" --train_data="$TRAIN_FILE" --dev_data="$DEV_FILE" --test_data="$TEST_FILE" --bert_model=bert-base-cased --output_dir="$OUTPUT"/"$run" --pretrained_model_dir=finetuned_models/"$domain_tuned" --do_train --do_eval --do_test --save_all_epochs --num_train_epochs 50 --seed "$seed"
    done;
done;

