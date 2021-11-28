(re)start|action|machine|status
-|-|-|-
||||running
||||
24 nov|pca base_big `dpr-c` `--data-small pruned` *tupperware*|13|running
26 nov|autoencoder base (test with loss and multiple dims) `dpr-c-pruned_cn` `--post-cn` *pleasingkeep*|14|3 failed, running
28 nov|sparse projection base `dpr-c_cn` `--dims 128` *unequalapparel*, `base_big_rproj.out`|15|stopped, running
||||
||||finished
||||
25 nov|autoencoder base `dpr-c_cn` `--post-cn` *yellowneuron*|14|ok
25 nov|analyze size `dpr-c` live|14|ok
12 nov|sparse projection base `dpr-c_cn` `--post-cn` max of 3 *kindlysherry_3* (continue partial), `base_big.out`|13|5 failed,ok
19 nov|rproj incremental greedy `dpr-c_cn` `--post-cn` *recentyarn* (continue partial, 4)|14|ok 
12 nov|pca base_big `dpr-c` `--data-small pruned` *tupperwear*|14|4 failed
18 nov|rproj incremental greedy `dpr-c_cn` `--post-cn` *recentyarn* (continue partial)|15|4 failed, partial 
12 nov|rproj incremental greedy `dpr-c_cn` `--post-cn` *recentyarn*|15|3 failed, running
28 oct|sparse projection base `dpr-c_cn` `--post-cn` max of 3 *kindlysherry_3* (continue partial)|13|killed
22 oct|dimension single `dpr-c-pruned_cn` `--post-cn` *dimsingle_unusedoven*|14|finished
23 oct|sparse projection base `dpr-c_cn` `--post-cn` max of 3 *kindlysherry_2*|13|partial fail
24 oct|pca base_big `dpr-c` `--data-small pruned` *tupperwear*|15|failed (memory)
23 oct|pca base_big `dpr-c` `--data-small pruned` *tupperwear*|15|failed (memory)
16 oct|sparse projection base `dpr-c_cn` `--post-cn` max of 3 *kindlysherry*|13|killed
16 oct|sparse projection base `dpr-c-pruned_cn` `--post-cn` *finickyking* (test)|14|ok
16 oct|norm dpr-c-pruned_cn|14|ok
7 oct|uncompressed dpr-t `--without-faiss`|14|ok
7 oct|uncompressed bert-t `--without-faiss`|14|ok
7 oct|uncompressed bert-c `--without-faiss`|13|ok
18 sep|embd dpr-t|14|ok
18 sep|embd bert-t|14|ok
18 sep|embd bert-c|14|ok
7 oct|uncompressed sbert-c `--without-faiss`|13|ok
28 sep|embd sbert-c|15|ok, removed (quota)
4 oct|uncompressed sbert-t `--without-faiss`|13|restarted, ok
3 oct|uncompressed dpr-c_n|13|ok
3 oct|uncompressed dpr-c_csn|13|ok
3 oct|norm dpr-c_n|13|ok, removed (quota) 
2 oct|norm dpr-c_csn|13|ok
2 oct|norm dpr-c_c||ok, removed (quota)
2 oct|uncompressed dpr-c_cn||ok
2 oct|uncompressed dpr-c_cs||ok
2 oct|run dpr-c norm -cs||ok
2 oct|run dpr-c norm -c||fail quota
1 oct|run dpr-c norm -cn || ok
1 oct|run dpr-c norm -cn, -cs || multiple fails
1 oct|run dpr-c uncompressed `--all`||fail
28 sep|uncompressed dpr-c (fast+slow)||ok
1 oct|run dpr-c zscores (sequentially with `--std` and `--std --center`||fail
10 sep| embd for dpr-t||ok, removed (quota)
4 sep| embd for dpr-c||ok
