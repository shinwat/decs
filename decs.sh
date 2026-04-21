#!/bin/bash

python decs/decs.py \
--seed 0 \
--hidden_dim 128 \
--batch_size 128 \
--eval_config_path "data/confs/100" \
--wandb False
