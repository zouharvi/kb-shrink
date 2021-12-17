#!/usr/bin/env bash

./src/reduce_dim/autoencoder/run_speed.py --data /data/big-hp/dpr-c-pruned.embd_cn --model 3 --logfile computed/auto_speed_puffyemery_gpu.log;
CUDA_VISIBLE_DEVICES= ./src/reduce_dim/autoencoder/run_speed.py --data /data/big-hp/dpr-c-pruned.embd_cn --model 3 --logfile computed/auto_speed_puffyemery.log;
./src/reduce_dim/pca/run_speed.py --data /data/big-hp/dpr-c-pruned.embd_cn --pca-model scikit --logfile computed/pca_speed_puffyemery.log;
./src/reduce_dim/pca/run_speed.py --data /data/big-hp/dpr-c-pruned.embd_cn --pca-model torch --logfile computed/pca_speed_puffyemery_torch_gpu.log;
CUDA_VISIBLE_DEVICES= ./src/reduce_dim/pca/run_speed.py --data /data/big-hp/dpr-c-pruned.embd_cn --pca-model torch --logfile computed/pca_speed_puffyemery_torch.log;