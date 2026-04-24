#!/bin/bash
# nohup ./run.sh > train_comparison.txt 2>&1 &
# Khai báo danh sách các method
methods=("flyprompt" "l2p" "dualprompt" "codaprompt" "mvp" "slca" "sprompt" "ranpac" "hide" "norga" "sdlora")

# Vòng lặp qua từng method
for METHOD in "${methods[@]}"
do
    echo "------------------------------------------------"
    echo "Running method: $METHOD"
    echo "------------------------------------------------"

    python main.py \
      --method "$METHOD" \
      --dataset cifar100 \
      --data_dir ./data/CIFAR \
      --backbone vit_base_patch16_224 \
      --n_tasks 5 --n 50 --m 10 \
      --batchsize 64 --lr 0.005 \
      --online_iter 1 --num_epochs 1 \
      --use_amp --eval_period 1000 \
      --note "${METHOD}_cifar100"

    echo "Finished $METHOD"
done