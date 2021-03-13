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
python3 embedding.py --dataset "data/eli5-dev-kilt.jsonl" --embd-out "data/eli5.dev-kilt.embd"
```

## Misc.

Based on KILT research & dataset:
- https://arxiv.org/abs/2009.02252
- https://github.com/facebookresearch/KILT