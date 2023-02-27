#!/usr/bin/env bash
python ./test.py \
--batch_size 16 --max_workers 16 --topk 5 --gpu 3 --reinforce False \
--model_dir ../../data/tadalog_model_v2/model_tadalog_c2_50000_top10_joslin_pre