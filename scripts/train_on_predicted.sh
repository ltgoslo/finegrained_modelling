#!/bin/bash
#
#SBATCH --job-name=bert_classification --account=nn9447k
#SBATCH --output=bert_classification-%j.out
#
#SBATCH --partition=accel --gres=gpu:1
#SBATCH --time=40:00:00
#SBATCH --array=3,7,8,11,15,16,19,23,24,27,31,32,35,39,40
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
DATASETS=([1]="darmstadt/services" [2]="darmstadt/universities" [3]="jiang" [4]="mitchell" [5]="mpqa" [6]="semeval/Laptop" [7]="semeval/Restaurant" [8]="wang" [9]="darmstadt/services" [10]="darmstadt/universities" [11]="jiang" [12]="mitchell" [13]="mpqa" [14]="semeval/Laptop" [15]="semeval/Restaurant" [16]="wang" [17]="darmstadt/services" [18]="darmstadt/universities" [19]="jiang" [20]="mitchell" [21]="mpqa" [22]="semeval/Laptop" [23]="semeval/Restaurant" [24]="wang" [25]="darmstadt/services" [26]="darmstadt/universities" [27]="jiang" [28]="mitchell" [29]="mpqa" [30]="semeval/Laptop" [31]="semeval/Restaurant" [32]="wang" [33]="darmstadt/services" [34]="darmstadt/universities" [35]="jiang" [36]="mitchell" [37]="mpqa" [38]="semeval/Laptop" [39]="semeval/Restaurant" [40]="wang")

DOMAIN_TUNED_MODELS=([1]="darmstadt/services" [2]="darmstadt/universities" [3]="jiang" [4]="mitchell" [5]="mpqa" [6]="laptop" [7]="restaurant" [8]="wang" [9]="darmstadt/services" [10]="darmstadt/universities" [11]="jiang" [12]="mitchell" [13]="mpqa" [14]="laptop" [15]="restaurant" [16]="wang" [17]="darmstadt/services" [18]="darmstadt/universities" [19]="jiang" [20]="mitchell" [21]="mpqa" [22]="laptop" [23]="restaurant" [24]="wang" [25]="darmstadt/services" [26]="darmstadt/universities" [27]="jiang" [28]="mitchell" [29]="mpqa" [30]="laptop" [31]="restaurant" [32]="wang" [33]="darmstadt/services" [34]="darmstadt/universities" [35]="jiang" [36]="mitchell" [37]="mpqa" [38]="laptop" [39]="restaurant" [40]="wang")

ANN_TYPES=([1]="targets_polarity" [2]="targets_polarity" [3]="targets_polarity" [4]="targets_polarity" [5]="targets_polarity" [6]="targets_polarity" [7]="targets_polarity" [8]="targets_polarity" [9]="targets_polarity" [10]="targets_polarity" [11]="targets_polarity" [12]="targets_polarity" [13]="targets_polarity" [14]="targets_polarity" [15]="targets_polarity" [16]="targets_polarity" [17]="targets_polarity" [18]="targets_polarity" [19]="targets_polarity" [20]="targets_polarity" [21]="targets_polarity" [22]="targets_polarity" [23]="targets_polarity" [24]="targets_polarity" [25]="targets_polarity" [26]="targets_polarity" [27]="targets_polarity" [28]="targets_polarity" [29]="targets_polarity" [30]="targets_polarity" [31]="targets_polarity" [32]="targets_polarity" [33]="targets_polarity" [34]="targets_polarity" [35]="targets_polarity" [36]="targets_polarity" [37]="targets_polarity" [38]="targets_polarity" [39]="targets_polarity" [40]="targets_polarity")

POOLING_TYPE=([1]="cls" [2]="cls" [3]="cls" [4]="cls" [5]="cls" [6]="cls" [7]="cls" [8]="cls" [9]="first" [10]="first" [11]="first" [12]="first" [13]="first" [14]="first" [15]="first" [16]="first" [17]="pooled" [18]="pooled" [19]="pooled" [20]="pooled" [21]="pooled" [22]="pooled" [23]="pooled" [24]="pooled" [25]="max" [26]="max" [27]="max" [28]="max" [29]="max" [30]="max" [31]="max" [32]="max" [33]="max-min-mean" [34]="max-min-mean" [35]="max-min-mean" [36]="max-min-mean" [37]="max-min-mean" [38]="max-min-mean" [39]="max-min-mean" [40]="max-min-mean")


i=$SLURM_ARRAY_TASK_ID
dataset=${DATASETS[$i]}
domain_tuned=${DOMAIN_TUNED_MODELS[$i]}
ann_type=${ANN_TYPES[$i]}
pooling=${POOLING_TYPE[$i]}


SEEDS=([1]="12345" [2]="23456" [3]="34567" [4]="45678" [5]="56789")

for expression_model in darmstadt/services darmstadt/universities mpqa; do
    # 5 runs
    for run in 1 2 3 4 5; do

        TRAIN_FILE=predicted_data/"$ann_type"/"$dataset"/"$expression_model"/train.conll
        DEV_FILE=predicted_data/"$ann_type"/"$dataset"/"$expression_model"/dev.conll
        TEST_FILE=predicted_data/"$ann_type"/"$dataset"/"$expression_model"/test.conll
        OUTPUT="classification_results/predicted_data/$dataset"/$ann_type"/$expression_model"/

        seed=${SEEDS[$run]}
        python polarity_classification.py --train_data="$TRAIN_FILE" --dev_data="$DEV_FILE" --test_data="$TEST_FILE" --bert_model=bert-base-cased --output_dir="$OUTPUT"/"$pooling"/"$run" --pretrained_model_dir=finetuned_models/"$domain_tuned" --do_train --do_valid --do_eval --num_train_epochs 50 --seed "$seed" --pooling "$pooling"
    done;
done;
