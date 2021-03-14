# Knowledge Base Shrink

Notes:

- Observe distribution of keys in KB
- Reduce embedding vector size
- - Truncated SVD/PCA, autoencoder, random projection, crop/fold embedding
- - Observe performance drop vs. size
- - Pre-train vs. post-train reduction effects
- Decrease knowledge base size by clustering.
- - Observe performance vs. cluster count
- - Pre-train vs. post-train reduction effects
- - MIPS has to be modified to gravitate towards the averages - store cluster size.

## Usage

```
pip3 install -r requirementss.txt
mkdir -p data
# download ELI5 dataset
wget -O data/eli5-dev.jsonl http://dl.fbaipublicfiles.com/KILT/eli5-dev-kilt.jsonl
# compute sentence prompt embeddings
python3 misc/embedding.py --dataset "data/eli5-dev-kilt.jsonl" --embd-out "data/eli5.dev-kilt.embd"
```

Run all scripts from the top directory of the repository.

```
$ python3 src/misc/analyze_size.py 
Whole dataset size:   11.9MB
Prompts size:        571.8KB  4.7%
Values size:          11.7MB 98.2%
Keys size:             8.8MB  0.8x values size
Keys size (calc):      8.8MB
One key size:         24.1KB
One key size (calc):   6.0KB
Number of entries:      1507
```

## Misc.

Based on KILT research & dataset:
- https://arxiv.org/abs/2009.02252
- https://github.com/facebookresearch/KILT