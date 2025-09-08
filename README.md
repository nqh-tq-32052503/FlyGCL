# FlyGCL: A Lightweight Continual Learning Framework (with FlyPrompt)

FlyGCL is a minimal, practical framework for Class-Incremental/Generalized Continual Learning on images. It supports true online learning in the Si-Blurry setting and includes strong prompt-based baselines built on ViT.

FlyPrompt (ours) is the default and recommended method in this repo. It uses per-task expert prompts, a random-projection gating head to route to experts, and EMA classifier heads for stable online learning.

## What's inside
- Methods: flyprompt (ours), l2p, dualprompt, codaprompt, mvp, slca, ranpac, moeranpac
- Backbones: ViT via timm (e.g., `vit_base_patch16_224`)
- Scenarios: Si-Blurry online CIL with configurable disjoint/blurry ratios
- Results: simple JSON logs + text logs under `results/`

## Installation
- Python 3.10+
- PyTorch 1.13.0 (CUDA 11.7)

```bash
conda create -n flygcl python=3.10 -y
conda activate flygcl
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 \
  --index-url https://download.pytorch.org/whl/cu117
pip install timm numpy pandas scikit-learn matplotlib
```

## Datasets (examples)
Place datasets under a directory of your choice and pass it via `--data_dir`.
- CIFAR-100: `/data/datasets/CIFAR`
- ImageNet-R: `/data/datasets/imagenet-r`
- CUB-200-2011: `/data/datasets/CUB200_2011`

Minimal structure (example):
```
/data/datasets/
  CIFAR/
  imagenet-r/
  CUB200_2011/
```

## Quick start (FlyPrompt)
Single-GPU, CIFAR-100, Si-Blurry (n=50, m=10), 5 tasks:
```bash
python main.py \
  --method flyprompt \
  --dataset cifar100 \
  --data_dir /data/datasets/CIFAR \
  --backbone vit_base_patch16_224 \
  --n_tasks 5 --n 50 --m 10 \
  --batchsize 64 --lr 0.005 \
  --online_iter 3 --num_epochs 1 \
  --use_amp --eval_period 1000 \
  --note flyprompt_cifar100
```

ImageNet-R example:
```bash
python main.py \
  --method flyprompt \
  --dataset imagenet-r \
  --data_dir /data/datasets/imagenet-r \
  --backbone vit_base_patch16_224 \
  --n_tasks 5 --n 50 --m 10 \
  --batchsize 32 --lr 0.005 \
  --online_iter 3 --num_epochs 1 \
  --use_amp --eval_period 1000 \
  --note flyprompt_imagenet_r
```

Run the prepared script (GPU 0, seeds "1 2 3", dataset cifar100):
```bash
bash scripts/run_baselines_flyprompt.sh 0 "1 2 3" cifar100 flyprompt_minimal
```

## Key arguments (most used)
- Method/dataset: `--method {flyprompt|l2p|dualprompt|codaprompt|mvp|slca|ranpac|moeranpac}` `--dataset {cifar100|imagenet-r|cub200|...}` `--data_dir /path`
- Tasks/setting: `--n_tasks 5` `--n 50` (disjoint class ratio, %) `--m 10` (blurry sample ratio, %)
- Training: `--batchsize 64` `--lr 0.005` `--online_iter 3` `--num_epochs 1` `--use_amp` `--eval_period 1000`
- Backbone: `--backbone vit_base_patch16_224`
- Repro: `--seeds 1 2 3` `--note my_experiment`

## FlyPrompt-specific arguments
- `--len_prompt 20` length per expert prompt
- `--pos_prompt 0 1 2 3 4` layers to insert prompts
- `--rp_dim 10000` random-projection dim for gating
- `--rp_ridge 1e4` ridge for closed-form RP head
- `--ema_ratio 0.9 0.99` EMA factors for expert FCs

Defaults are chosen to work well across common settings. See `configuration/config.py` for all flags.

## Results and logs
Outputs are stored under `results/`:
```
results/
  logs/{dataset}/{method_dataset_note}/
    seed_{seed}_log.txt
  *.json  # structured results
```
Monitor a running job:
```bash
tail -f results/logs/cifar100/flyprompt_cifar100/seed_1_log.txt
```

## Project layout (minimal)
- `main.py` entry point (loads args, builds trainer, runs)
- `configuration/config.py` argument definitions (incl. FlyPrompt flags)
- `methods/` trainers (FlyPrompt in `methods/flyprompt.py`)
- `models/` model components (FlyPrompt model in `models/flyprompt.py`)
- `datasets/` dataset wrappers; `utils/` training helpers
- `scripts/` ready-to-run baselines

## Citation
If this repository helps your research, please cite it. FlyPrompt is our proposed algorithm in this project.

## License
MIT. See LICENSE.
