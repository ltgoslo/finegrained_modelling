#!/bin/bash
#
#SBATCH --job-name=domain_extraction --account=nn9447k
#SBATCH --output=domain_extraction-%j.out
#
#SBATCH --partition=accel --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --array=1-34
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
    [9]="darmstadt/services" [10]="darmstadt/universities" [11]="jiang" [12]="mitchell" [13]="mpqa" [14]="semeval/Laptop" [15]="semeval/Restaurant" [16]="wang"
    [17]="darmstadt/services" [18]="darmstadt/universities" [19]="mpqa"
    [20]="darmstadt/services" [21]="darmstadt/universities" [22]="mpqa"
    [23]="darmstadt/services" [24]="darmstadt/universities" [25]="mpqa"
    [26]="darmstadt/services" [27]="darmstadt/universities" [28]="mpqa"
    [29]="darmstadt/services" [30]="darmstadt/universities" [31]="mpqa"
    [32]="darmstadt/services" [33]="darmstadt/universities" [34]="mpqa")

DOMAIN_TUNED_MODELS=([1]="darmstadt/services" [2]="darmstadt/universities" [3]="jiang" [4]="mitchell" [5]="mpqa" [6]="laptop" [7]="restaurant" [8]="wang"
                     [9]="darmstadt/services" [10]="darmstadt/universities" [11]="jiang" [12]="mitchell"
     [13]="mpqa" [14]="laptop" [15]="restaurant" [16]="wang"
     [17]="darmstadt/services" [18]="darmstadt/universities" [19]="mpqa"
    [20]="darmstadt/services" [21]="darmstadt/universities" [22]="mpqa"
    [23]="darmstadt/services" [24]="darmstadt/universities" [25]="mpqa"
    [26]="darmstadt/services" [27]="darmstadt/universities" [28]="mpqa"
    [29]="darmstadt/services" [30]="darmstadt/universities" [31]="mpqa"
    [32]="darmstadt/services" [33]="darmstadt/universities" [34]="mpqa")


ANN_TYPES=([1]="targets" [2]="targets" [3]="targets" [4]="targets" [5]="targets" [6]="targets" [7]="targets" [8]="targets"
    [9]="targets_polarity" [10]="targets_polarity" [11]="targets_polarity" [12]="targets_polarity" [13]="targets_polarity" [14]="targets_polarity" [15]="targets_polarity" [16]="targets_polarity"
    [17]="targets_expressions" [18]="targets_expressions" [19]="targets_expressions"
    [20]="targets_expressions_polarity" [21]="targets_expressions_polarity" [22]="targets_expressions_polarity"
    [23]="targets_holders" [24]="targets_holders" [25]="targets_holders"
    [26]="targets_holders_polarity" [27]="targets_holders_polarity" [28]="targets_holders_polarity"
    [29]="full" [30]="full" [31]="full"
    [32]="full_polarity" [33]="full_polarity" [34]="full_polarity")

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
    OUTPUT="domain_extraction_results/$ann_type"/"$dataset"-"$test_data"
    for run in 1 2 3 4 5; do
        seed=${SEEDS[$run]}
        python target_extraction.py --annotation_type="$ann_type" --train_data="$TRAIN_FILE" --dev_data="$DEV_FILE" --test_data="$TEST_FILE" --bert_model=bert-base-cased --output_dir="$OUTPUT"/"$run" --trained_model_dir=finetuned_models/"$domain_tuned" --do_train --do_eval --do_test --save_all_epochs --num_train_epochs 50 --seed "$seed"
    done;
done;

