#!/usr/bin/env bash

./src/reduce_count/splitting/kilt_preprocessing.py --data-out /data/split/fixed040.pkl --splitter fixed --splitter-width 40 &
./src/reduce_count/splitting/kilt_preprocessing.py --data-out /data/split/sent04.pkl --splitter sent --splitter-width 4 &
./src/reduce_count/splitting/kilt_preprocessing.py --data-out /data/split/sent06.pkl --splitter sent --splitter-width 6 &
CUDA_VISIBLE_DEVICES=1 ./src/misc/embedding.py --data-in /data/split/fixed040.pkl --data-out /data/split/fixed040.embd --model dpr --type-out cls &
CUDA_VISIBLE_DEVICES=2 ./src/misc/embedding.py --data-in /data/split/sent04.pkl --data-out /data/split/sent04.embd --model dpr --type-out cls &
CUDA_VISIBLE_DEVICES=3 ./src/misc/embedding.py --data-in /data/split/sent06.pkl --data-out /data/split/sent06.embd --model dpr --type-out cls &