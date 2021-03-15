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
python3 src/misc/embedding.py --dataset "data/eli5-dev.jsonl" --embd-out "data/eli5-dev.embd"
```

Run all scripts from the top directory of the repository.

```
$ python3 src/misc/analyze_size.py 
Whole dataset size:   11.9MB
Prompts size:        571.8KB  4.7%
Values size:          11.7MB 98.2%
Keys size (comp):      4.4MB  0.4x values size
Keys size (calc):      4.4MB
One key size (comp):  24.1KB
One key size (calc):   3.0KB
Key shape:            (768,)
Key element type:    float32
Number of entries:      1507
```

```
$ python3 src/reduce_dim/prec_pca.py 
Method                           Size                Loss
Original (float32)              4.4MB (1.0x)    0.0000000
PCA (512)                       2.9MB (0.7x)    0.0068601
PCA (256)                       1.5MB (0.3x)    0.1049010
Precision (float16)             2.2MB (0.5x)    0.0033717
Precision (float16), PCA (512)  2.9MB (0.7x)    0.0074680
PCA (512), Precision (float16)  1.5MB (0.3x)    0.0068810
PCA (256), Precision (float16)  0.7MB (0.2x)    0.1049825
```

```
$ python3 src/reduce_dim/autoencoder.py 
Method                           Size                Loss
Autoencoder (256)               1.5MB (0.3x)    0.0001672
Autoencoder (128)               1.5MB (0.17x)   0.0002453
Autoencoder (64)                0.8MB (0.08x)   0.0002842
```

## Misc.

Based on KILT research & dataset:
- https://arxiv.org/abs/2009.02252
- https://github.com/facebookresearch/KILT