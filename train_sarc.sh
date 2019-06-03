#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main_sarc.py \
  --batch_size 1 \
  --lr 1 \
  --dropout 0.45 \
  --dropouth 0.3 \
  --dropouti 0.5 \
  --wdrop 0.45 \
  --chunk_size 10 \
  --seed 141 \
  --epoch 1000 \
  --data ./dataset/SARC/2.0/main
