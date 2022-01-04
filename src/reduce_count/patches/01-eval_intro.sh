#!/usr/bin/env bash

echo "fixed060" >> runs/split_intro_darkrouter.log
./src/misc/uncompressed.py --metric acc --center --norm --data /data/split/fixed060.embd >> runs/split_intro_darkrouter.log
echo "fixed080" >> runs/split_intro_darkrouter.log
./src/misc/uncompressed.py --metric acc --center --norm --data /data/split/fixed080.embd >> runs/split_intro_darkrouter.log
echo "fixed100" >> runs/split_intro_darkrouter.log
./src/misc/uncompressed.py --metric acc --center --norm --data /data/split/fixed100.embd >> runs/split_intro_darkrouter.log
echo "fixed120" >> runs/split_intro_darkrouter.log
./src/misc/uncompressed.py --metric acc --center --norm --data /data/split/fixed120.embd >> runs/split_intro_darkrouter.log
echo "fixed080_overlap20" >> runs/split_intro_darkrouter.log
./src/misc/uncompressed.py --metric acc --center --norm --data /data/split/fixed080_overlap20.embd >> runs/split_intro_darkrouter.log

echo "sent01" >> runs/split_intro_darkrouter.log
./src/misc/uncompressed.py --metric acc --center --norm --data /data/split/sent01.embd >> runs/split_intro_darkrouter.log
echo "sent02" >> runs/split_intro_darkrouter.log
./src/misc/uncompressed.py --metric acc --center --norm --data /data/split/sent02.embd >> runs/split_intro_darkrouter.log
echo "sent03" >> runs/split_intro_darkrouter.log
./src/misc/uncompressed.py --metric acc --center --norm --data /data/split/sent03.embd >> runs/split_intro_darkrouter.log
echo "sent05" >> runs/split_intro_darkrouter.log
./src/misc/uncompressed.py --metric acc --center --norm --data /data/split/sent05.embd >> runs/split_intro_darkrouter.log
