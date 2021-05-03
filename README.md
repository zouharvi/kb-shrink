# Knowledge Base Shrink

Topics experimented with:

- Observe distribution of keys in KB
- Reduce embedding vector size
- - Truncated SVD/PCA, autoencoder, random projections, manifold learning
- - Similarity distillation
- - Observe performance drop vs. size
- - Pre-train vs. post-train reduction effects (fine-tuning)
- - Effect on different retrieval metrics (IP vs L2)
- Decrease knowledge base size by clustering.
- - Observe performance vs. cluster count
- - Cluster aggregation
- - Pre-train vs. post-train reduction effects
- - MIPS has to be modified to gravitate towards the averages - store cluster size.

## Compute Embeddings

Used in other scripts

```
pip3 install -r requirementss.txt
mkdir -p data
# download ELI5 dataset
wget -O data/hotpot.jsonl http://dl.fbaipublicfiles.com/KILT/hotpotqa-train-kilt.jsonl
# compute sentence prompt embeddings
python3 src/misc/embedding.py --dataset "data/hotpot.jsonl" --embd "data/hotpot.embd"
```

Run all scripts from the top directory of the repository.

## Misc.

Based on KILT research & dataset:
- https://arxiv.org/abs/2009.02252
- https://github.com/facebookresearch/KILT