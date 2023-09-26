# Knowledge Base Shrink

Read the [ACL paper](https://aclanthology.org/2022.spanlp-1.5/), the [master thesis](meta/thesis/zouhar_thesis_lct.pdf) or watch the [presentation](https://youtu.be/qczabc2oyMA).

The 768-dimensional embedding of 2019 Wikipedia dump (split to 100 token segment) takes almost 150GB.
This poses practical issues for both research and applications.
We aim to reduce the size through two methods:

Dimensionality reduction of the embedding:
- PCA, autoencoder, random projections
- Effect on IP vs L2
- Pre-processing
- Training/evaluation data size dependency

Document splitting & filtering: 
- Split on segments respecting semantic boundaries
- Get retrievability annotations and train a filtering system
- Decrease knowledge base size by clustering (join neighbours pointing to the same doc)
  - Observe performance vs. cluster count
  - Cluster aggregation
  - Pre-train vs. post-train reduction effects

# Recommendations

- Always use pre- and post-processing (centering & normalization).
- PCA is a good enough solution that requires very little data (1k vectors) to fit and is stable. The autoencoder provides a slight improvement but is less stable.
- 8-bit floats are supported and offer very little performance drop. Combine PCA and this precision reduction for the best trade-off.

# Citation

```
@inproceedings{zouhar2022knowledge,
  title={Knowledge Base Index Compression via Dimensionality and Precision Reduction},
  author={Zouhar, Vil{\'e}m and Mosbach, Marius and Zhang, Miaoran and Klakow, Dietrich},
  booktitle={Proceedings of the 1st Workshop on Semiparametric Methods in NLP: Decoupling Logic from Knowledge},
  pages={41--53},
  year={2022},
  url={https://aclanthology.org/2022.spanlp-1.5/},
}
```

<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/qczabc2oyMA?si=Ae6kdER8egRUU46b" title="Paper video presentation" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

Furthermore, this project is also [a Master thesis](https://raw.githubusercontent.com/zouharvi/kb-shrink/main/meta/thesis/zouhar_thesis_lct.pdf).

<!--
## Pipeline

Dimensionality reduction:

1. Process Wikipedia dump and create document segments with relevancy annotation: <br>
`./src/misc/kilt_preprocessing.py --data-out /data/hp/full.pkl`

2. Compute embedding of segments: (may take a lot of time) <br>
`./src/misc/embedding.py --data-in /data/hp/full.pkl --data-out /data/hp/dpr-c.pkl --model dpr --type-out cls`

3. Evaluate retrieval performance: <br>
`./src/misc/uncompressed.py --data /data/hp/dpr-c.pkl`

4. Use PCA and Autoencoder for comparison.

Run all scripts from the top directory of the repository.

-->

## Acknowledgement

- Based on KILT [research](https://arxiv.org/abs/2009.02252) & [dataset](https://github.com/facebookresearch/KILT).
- This work was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – Project-ID 232722074 – SFB 1102.
