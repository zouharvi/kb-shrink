# Knowledge Base Shrink

The 768-dimensional embedding of 2019 Wikipedia dump (split to 100 token segment) takes almost 150GB.
This poses practical issues for both research and applications.
We aim to reduce the size through two methods:

Dimensionality reduction of the embedding:
- PCA, autoencoder, random projections, manifold learning
- Similarity distillation
- Effect on IP vs L2
- Downstream performance

Document splitting & filtering:
- TBD

<!--
- Observe distribution of keys in KB
- Reduce embedding vector size
- - Observe performance drop vs. size
- - Pre-train vs. post-train reduction effects (fine-tuning)
- Decrease knowledge base size by clustering.
- - Observe performance vs. cluster count
- - Cluster aggregation
- - Pre-train vs. post-train reduction effects
- - MIPS has to be modified to gravitate towards the averages - store cluster size. -->

## Pipeline

1. Process Wikipedia dump and create document segments with relevancy annotation: <br>
`./src/misc/kilt_preprocessing.py --data-out /data/big-hp/full.pkl`

2. Compute embedding of segments: (may take a lot of time) <br>
`./src/misc/embedding.py --data-in /data/big-hp/full.pkl --data-out /data/big-hp/dpr-c.pkl --model dpr --type-out cls`

3. Evaluate retrieval performance: <br>
`./src/misc/uncompressed.py --data /data/big-hp/dpr-c.pkl`

4. Use PCA and Autoencoder for comparison.

Run all scripts from the top directory of the repository.

## Acknowledgement

Based on KILT research & dataset:
- https://arxiv.org/abs/2009.02252
- https://github.com/facebookresearch/KILT