#!/bin/bash

# Create new folder where you will keep all processed data
mkdir processed

cd processing_scripts

# Process mpqa data
python3 process_mpqa.py

# Process darmstadt data
cd ..
unzip DarmstadtServiceReviewCorpus.zip
cd DarmstadtServiceReviewCorpus
unzip services
unzip universities
grep -rl "&" universities/basedata | xargs sed -i 's/&/and/g'
cd ..
cd processing_scripts
python3 process_darmstadt.py

# Process semeval data
python3 process_semeval.py

# Process sentihood
python3 process_sentihood.py


# Process mitchell et al
cd ..
mkdir mitchell
tar -xvf MitchellEtAl-13-OpenSentiment.tgz -C mitchell
grep -Prl "TELEPHONE\tNUMBER" mitchell/en/10-fold/* | xargs sed -iE 's/TELEPHONE\tNUMBER/TELEPHONE-NUMBER/g'
cd processing_scripts
python3 process_mitchell.py

# Process wang, et al.
cd ..
mkdir wangetal
unzip wangetal.zip -d wangetal
cd wangetal
tar -xvf annotations.tar.gz
tar -xvf tweets.tar.gz
cd ../processing_scripts
python3 process_wang.py


# Process Jiang et al.
python3 process_jiang.py

cd ..

# Create all of the conll datasets
for corpus in processed/darmstadt/services processed/darmstadt/universities processed/jiang processed/mitchell processed/mpqa processed/semeval/Laptop processed/semeval/Restaurant processed/wang; do
    # TARGETS
    python3 convert_to_bio.py --indir $corpus --outdir $corpus/targets --to_add Target --no_polarity
    python3 convert_to_bio.py --indir $corpus --outdir $corpus/targets_polarity --to_add Target
    # HOLDERS
    python3 convert_to_bio.py --indir $corpus --outdir $corpus/holders --to_add Source --no_polarity
    python3 convert_to_bio.py --indir $corpus --outdir $corpus/holders_polarity --to_add Source
    # EXPRESSIONS
    python3 convert_to_bio.py --indir $corpus --outdir $corpus/expressions --to_add Polar_expression --no_polarity
    python3 convert_to_bio.py --indir $corpus --outdir $corpus/expressions_polarity --to_add Polar_expresion
    # TARGETS AND EXPRESSIONS
    python3 convert_to_bio.py --indir $corpus --outdir $corpus/targets_expressions --to_add Target Polar_expression --no_polarity
    python3 convert_to_bio.py --indir $corpus --outdir $corpus/targets_expressions_polarity --to_add Target Polar_expresion
    # TARGETS AND HOLDERS
    python3 convert_to_bio.py --indir $corpus --outdir $corpus/targets_holders --to_add Target Source --no_polarity
    python3 convert_to_bio.py --indir $corpus --outdir $corpus/targets_holders_polarity --to_add Target Source
    # FULL
    python3 convert_to_bio.py --indir $corpus --outdir $corpus/full --no_polarity
    python3 convert_to_bio.py --indir $corpus --outdir $corpus/full_polarity
done
