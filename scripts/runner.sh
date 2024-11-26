#!/usr/bin/env bash
#training_keys=(10 20 30 40 50 60 70 80 90 100 110)
#training_keys=(170 180 190 200 210 220 230 240 250 260 270)
#training_keys=(120 130 140 150 160 280 290 300 310 320)

#prediction_keys=(15485863)
#cd /tmp/pycharm_project_68 || exit
#conda activate synthid
#export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export CUDA_VISIBLE_DEVICES=0,1,2
# shellcheck disable=SC2043
for algorithm in "SynthID" "EXP" "KGW" "SIR"
do
    python3 evaluation/examples/assess_overall.py --mode end_to_end --algorithm $algorithm --suff _new --hash_keys 123456 789101 112131
done
# "Unigram" "SynthID" "EXP"
# "Unbiased" "DIP"  "Unbiased" "DIP"
# "Unbiased" "SWEET" "ITSEdit" "TS" "DIP"  "SIR" "XSIR" "UPV" "EWD" "EXPEdit" "EXPGumbel"

#python3 evaluation/examples/assess_overall.py --algorithm Unigram --mode base_paraphrase --hash_keys 170 180 190 200 210 220 230 240 250 260 270