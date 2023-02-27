#!/usr/bin/env bash
python train.py \
--attr_lr 0.0001 --name_lr 0.0001 --rela_lr 0.0001 --batch_size 16 --max_workers 16 --topk 10 --gpu 2 --reinforce False \
--model_dir ../../data/tadalog_model_v2/model_tadalog_c2_50000_top10_joslin_pre \
--train_f ../../data/dataset/task_list/train_tasks_c2_1000.pkl \
--val_f ../../data/dataset/task_list/val_tasks_c2_1000.pkl \
