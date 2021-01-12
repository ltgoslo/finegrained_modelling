#!/bin/bash
#
#SBATCH --job-name=expression_prediction --account=nn9447k
#SBATCH --output=expression_prediction-%j.out
#
#SBATCH --partition=accel --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --array=1-8
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
DATASETS=([1]="darmstadt/services" [2]="darmstadt/universities" [3]="jiang" [4]="mitchell" [5]="mpqa" [6]="semeval/Laptop" [7]="semeval/Restaurant" [8]="wang")

DOMAIN_TUNED_MODELS=([1]="darmstadt/services" [2]="darmstadt/universities" [3]="jiang" [4]="mitchell" [5]="mpqa" [6]="laptop" [7]="restaurant" [8]="wang")


ANN_TYPES=([1]="targets_polarity" [2]="targets_polarity" [3]="targets_polarity" [4]="targets_polarity" [5]="targets_polarity" [6]="targets_polarity" [7]="targets_polarity" [8]="targets_polarity")

i=$SLURM_ARRAY_TASK_ID
dataset=${DATASETS[$i]}
domain_tuned=${DOMAIN_TUNED_MODELS[$i]}
ann_type=${ANN_TYPES[$i]}


DATA_DIR=data/processed/"$dataset"/"$ann_type"
OUTPUT="predicted_data/$ann_type"/$dataset

SEEDS=([1]="12345" [2]="23456" [3]="34567" [4]="45678" [5]="56789")

# 5 runs
for exp_model in darmstadt/services darmstadt/universities mpqa; do
    for run in 1 2 3 4 5; do
        python tag_data.py --annotation_type=expressions --bert_model=bert-base-cased --trained_model_dir=finetuned_models/"$domain_tuned" --data_dir="$DATA_DIR" --output_dir="$OUTPUT"/"$exp_model"/"$run" --finetuned_model_dir expression_extraction_results/expressions/"$exp_model"/"$run"
    done;
done;

