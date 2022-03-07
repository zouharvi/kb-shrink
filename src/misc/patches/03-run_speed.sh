#!/usr/bin/env bash

# CUDA_VISIBLE_DEVICES=2 ./src/reduce_dim/pca/run_speed.py --logfile computed/speed_pca_scikit_reddrapes_1.json --pca-model scikit --seed 1 >> runs/speed_pca_scikit_reddrapes.log
# CUDA_VISIBLE_DEVICES=2 ./src/reduce_dim/pca/run_speed.py --logfile computed/speed_pca_scikit_reddrapes_2.json --pca-model scikit --seed 2 >> runs/speed_pca_scikit_reddrapes.log
# CUDA_VISIBLE_DEVICES=2 ./src/reduce_dim/pca/run_speed.py --logfile computed/speed_pca_scikit_reddrapes_3.json --pca-model scikit --seed 3 >> runs/speed_pca_scikit_reddrapes.log
# CUDA_VISIBLE_DEVICES=2 ./src/reduce_dim/pca/run_speed.py --logfile computed/speed_pca_scikit_reddrapes_4.json --pca-model scikit --seed 4 >> runs/speed_pca_scikit_reddrapes.log
# CUDA_VISIBLE_DEVICES=2 ./src/reduce_dim/pca/run_speed.py --logfile computed/speed_pca_scikit_reddrapes_5.json --pca-model scikit --seed 5 >> runs/speed_pca_scikit_reddrapes.log

# CUDA_VISIBLE_DEVICES= ./src/reduce_dim/pca/run_speed.py --logfile computed/speed_pca_torch_cpu_reddrapes_1.json --pca-model torch --seed 1 >> runs/speed_pca_torch_cpu_reddrapes.log
# CUDA_VISIBLE_DEVICES= ./src/reduce_dim/pca/run_speed.py --logfile computed/speed_pca_torch_cpu_reddrapes_2.json --pca-model torch --seed 2 >> runs/speed_pca_torch_cpu_reddrapes.log
# CUDA_VISIBLE_DEVICES= ./src/reduce_dim/pca/run_speed.py --logfile computed/speed_pca_torch_cpu_reddrapes_3.json --pca-model torch --seed 3 >> runs/speed_pca_torch_cpu_reddrapes.log
# CUDA_VISIBLE_DEVICES= ./src/reduce_dim/pca/run_speed.py --logfile computed/speed_pca_torch_cpu_reddrapes_4.json --pca-model torch --seed 4 >> runs/speed_pca_torch_cpu_reddrapes.log
# CUDA_VISIBLE_DEVICES= ./src/reduce_dim/pca/run_speed.py --logfile computed/speed_pca_torch_cpu_reddrapes_5.json --pca-model torch --seed 5 >> runs/speed_pca_torch_cpu_reddrapes.log

# CUDA_VISIBLE_DEVICES=2 ./src/reduce_dim/pca/run_speed.py --logfile computed/speed_pca_torch_gpu_reddrapes_1.json --pca-model torch --seed 1 >> runs/speed_pca_torch_gpu_reddrapes.log
# CUDA_VISIBLE_DEVICES=2 ./src/reduce_dim/pca/run_speed.py --logfile computed/speed_pca_torch_gpu_reddrapes_2.json --pca-model torch --seed 2 >> runs/speed_pca_torch_gpu_reddrapes.log
# CUDA_VISIBLE_DEVICES=2 ./src/reduce_dim/pca/run_speed.py --logfile computed/speed_pca_torch_gpu_reddrapes_3.json --pca-model torch --seed 3 >> runs/speed_pca_torch_gpu_reddrapes.log
# CUDA_VISIBLE_DEVICES=2 ./src/reduce_dim/pca/run_speed.py --logfile computed/speed_pca_torch_gpu_reddrapes_4.json --pca-model torch --seed 4 >> runs/speed_pca_torch_gpu_reddrapes.log
# CUDA_VISIBLE_DEVICES=2 ./src/reduce_dim/pca/run_speed.py --logfile computed/speed_pca_torch_gpu_reddrapes_5.json --pca-model torch --seed 5 >> runs/speed_pca_torch_gpu_reddrapes.log

CUDA_VISIBLE_DEVICES=2 ./src/reduce_dim/autoencoder/run_speed.py --logfile computed/speed_auto_torch_gpu_reddrapes_1.json --model 1 --seed 1 >> runs/speed_auto_torch_gpu_reddrapes.log
CUDA_VISIBLE_DEVICES=2 ./src/reduce_dim/autoencoder/run_speed.py --logfile computed/speed_auto_torch_gpu_reddrapes_2.json --model 1 --seed 2 >> runs/speed_auto_torch_gpu_reddrapes.log
CUDA_VISIBLE_DEVICES=2 ./src/reduce_dim/autoencoder/run_speed.py --logfile computed/speed_auto_torch_gpu_reddrapes_3.json --model 1 --seed 3 >> runs/speed_auto_torch_gpu_reddrapes.log
CUDA_VISIBLE_DEVICES=2 ./src/reduce_dim/autoencoder/run_speed.py --logfile computed/speed_auto_torch_gpu_reddrapes_4.json --model 1 --seed 4 >> runs/speed_auto_torch_gpu_reddrapes.log
CUDA_VISIBLE_DEVICES=2 ./src/reduce_dim/autoencoder/run_speed.py --logfile computed/speed_auto_torch_gpu_reddrapes_5.json --model 1 --seed 5 >> runs/speed_auto_torch_gpu_reddrapes.log

CUDA_VISIBLE_DEVICES= ./src/reduce_dim/autoencoder/run_speed.py --logfile computed/speed_auto_torch_cpu_reddrapes_1.json --model 1 --seed 1 >> runs/speed_auto_torch_cpu_reddrapes.log
CUDA_VISIBLE_DEVICES= ./src/reduce_dim/autoencoder/run_speed.py --logfile computed/speed_auto_torch_cpu_reddrapes_2.json --model 1 --seed 2 >> runs/speed_auto_torch_cpu_reddrapes.log
CUDA_VISIBLE_DEVICES= ./src/reduce_dim/autoencoder/run_speed.py --logfile computed/speed_auto_torch_cpu_reddrapes_3.json --model 1 --seed 3 >> runs/speed_auto_torch_cpu_reddrapes.log
CUDA_VISIBLE_DEVICES= ./src/reduce_dim/autoencoder/run_speed.py --logfile computed/speed_auto_torch_cpu_reddrapes_4.json --model 1 --seed 4 >> runs/speed_auto_torch_cpu_reddrapes.log
CUDA_VISIBLE_DEVICES= ./src/reduce_dim/autoencoder/run_speed.py --logfile computed/speed_auto_torch_cpu_reddrapes_5.json --model 1 --seed 5 >> runs/speed_auto_torch_cpu_reddrapes.log

