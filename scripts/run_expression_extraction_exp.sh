#!/bin/bash
#
#SBATCH --job-name=exp_extraction --account=nn9447k
#SBATCH --output=exp_extraction-%j.out
#
#SBATCH --partition=accel --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --array=1-3
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
DATASETS=([1]="darmstadt/services" [2]="darmstadt/universities" [3]="mpqa"
          [4]="darmstadt/services" [5]="darmstadt/universities" [6]="mpqa")

DOMAIN_TUNED_MODELS=([1]="darmstadt/services" [2]="darmstadt/universities" [3]="mpqa" [4]="darmstadt/services" [5]="darmstadt/universities" [6]="mpqa")

TEST_DATA=([1]="darmstadt/services" [2]="darmstadt/universities" [3]="mpqa"
          [4]="jiang" [5]="mitchell" [6]="semeval/Laptop" [7]="semeval/Restaurant" [8]="wang")

ANN_TYPES=([1]="expressions" [2]="expressions" [3]="expressions"
           [4]="expressions_polarity" [5]="expressions_polarity" [6]="expressions_polarity")

i=$SLURM_ARRAY_TASK_ID
dataset=${DATASETS[$i]}
domain_tuned=${DOMAIN_TUNED_MODELS[$i]}
ann_type=${ANN_TYPES[$i]}


TRAIN_FILE=data/processed/"$dataset"/"$ann_type"/train.conll
DEV_FILE=data/processed/"$dataset"/"$ann_type"/dev.conll
TEST_FILE=data/processed/"$dataset"/"$ann_type"/test.conll
OUTPUT="expression_extraction_results/$ann_type"/"$dataset"

SEEDS=([1]="12345" [2]="23456" [3]="34567" [4]="45678" [5]="56789")

# 5 runs
for run in 1 2 3 4 5; do
    seed=${SEEDS[$run]}
    python target_extraction.py --annotation_type="$ann_type" --test_ann=exp --train_data="$TRAIN_FILE" --dev_data="$DEV_FILE" --test_data="$TEST_FILE" --bert_model=bert-base-cased --output_dir="$OUTPUT"/"$run" --trained_model_dir=finetuned_models/"$domain_tuned" --do_train --do_eval --do_test --save_all_epochs --num_train_epochs 50 --seed "$seed"
done;
