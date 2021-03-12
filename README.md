# Knowledge Base Shrink

Goals:

- Observe distribution of keys in KB
- Reduce embedding vector size
- - Truncated SVD/PCA, autoencoder, random projection, crop/fold embedding
- - Observe performance drop vs. size
- - Pre-train vs. post-train reduction effects
- Decrease knowledge base size by clustering.
- - Observe performance vs. cluster count
- - Pre-train vs. post-train reduction effects
- - MIPS has to be modified to gravitate towards the averages - store cluster size.

Based on KILT research & dataset https://arxiv.org/abs/2009.02252.