#!/bin/bash

ablation=$1
fromSeed=$2
toSeed=$3

for i in {$fromSeed..$toSeed}; do
    python decs/decs.py \
    --ablate $ablation \
    --seed $i \
    --eval_config_path "data/confs/confs_100.npy" \
    --eval_env_path "data/confs/envs_100.npy" \
    --results_path "data/results/$ablation" \
    --checkpoints_path "data/models/$ablation" \
    --save_checkpoints True \
    --device "cuda" \
    --direct True \
    --wandb False
done
