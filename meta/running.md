(re)start|action|machine|status
-|-|-|-
||||running/staged
||||
||||
||||finished
||||
1 feb|intersection multipass *distantball*|12,13,14|ok
1 feb|intersection all *distantball*|13|ok
31 jan|intersection small 1000 queries *distantball*|13|ok, ok
15 jan|pca mult `{,--post-cn}` *orangeball*|15|ok
15 jan|pca 1bit (245) `{,--post-cn}` *orangeball*|15|ok
15 jan|pca 1bit (245) `nq` `{,--post-cn}` *orangeball*|13|ok
14 jan|pca 8bit `{,--post-cn}` *blacklid*|15|ok
14 jan|pca 16bit `{,--post-cn}` *blacklid*|14|ok
13 jan|irrelevant *oilynoodles_a* (redirect to nohup)|14|ok
13 jan|irrelevant *oilynoodles_b*|15|ok
13 jan|irrelevant *oilynoodles_c*|13|ok
08 jan|rproj greedy `nq/dpr-c` `--post-cn` *dryrope*|13|ok
08 jan|rproj greedy `nq/dpr-c` *dryrope*|14|ok
07 jan|pca_prec `dpr-c` `--post-cn` *clearwindow*|14|ok
02 jan|rproj dimsingle `nq/dpr-c` *dryrope*|13|ok
03 jan|embd `{sent0{4,6},fixed040}` dpr-c`|14|ok
02 jan|splitting & embedding ELI5, TriviaQA, Trex, WoW `fixed100` (pruned), `dpr-c`|15|ok
04 jan|uncompressed `wow/dpr-c-pruned`|13|ok
03 jan|split `{sent0{4,6},fixed040}` dpr-c`|15|ok
03 jan|prec base `nq/dpr-c` `{,--post-cn}` *dryrope*|15|ok
03 jan|pca_prec `nq/dpr-c` `{,--post-cn}` *dryrope*|12|ok
03 jan|auto base `--model {1,2,3}` `{,--post-cn}` `{,--regularize}` *cluelessmoon* (should have been called dryrope)|12-14|ok
03 jan|uncompressed `sent0{1,2,3,5}` `dpr-c-pruned` *cluelessmoon*|12-15|ok
02 jan|pca scaled nocn `nq/dpr-c` *dryrope*|14|ok
02 jan|pca scaled `nq/dpr-c` *dryrope*|14|ok
02 jan|pca base `nq/dpr-c` *dryrope*|14|ok
02 jan|rproj nocn `nq/dpr-c` *dryrope*|14|ok
02 jan|rproj `nq/dpr-c` *dryrope*|13|ok
01 jan|auto base `dpr-c-pruned` rescaled, *silentheadpohnes_0*|12|bad code, ok
01 jan|auto base `dpr-c-pruned` rescaled, `--norm` *silentheadpohnes_1*|13|bad code, ok
01 jan|auto base `dpr-c-pruned` rescaled, `--center` *silentheadpohnes_2*|14|bad code, ok
01 jan|auto base `dpr-c-pruned` rescaled, `--center --norm` *silentheadpohnes_3*|15|bad code, ok
02 jan|pca scaled `--dims 128` `nq/dpr-c-pruned` *dryrope*|14|ok
02 jan|pca base `--dims 128` `nq/dpr-c-pruned` *dryrope*|14|ok
02 jan|uncompressed `--dims 128` `nq/dpr-c-pruned` *dryrope*|13|ok
31 dec|pca base `dpr-c-pruned` rescaled, `--norm` *silentheadpohnes_1*|13|bad code, ok
31 dec|pca base `dpr-c-pruned` rescaled, `--center` *silentheadpohnes_2*|12|bad code, ok
31 dec|pca base `dpr-c-pruned` rescaled, `--center --norm` *silentheadpohnes_3*|14|bad code, ok
27 dec|embedding NQ `fixed100` (pruned), `dpr-c`|15|ok
27 dec|splitting & embedding `sent{01,02,03,05}` (pruned), `dpr-c`|14,15|ok
27 dec|splitting & embedding `fixed{080,100,120}, fixed080_overlap20` (pruned), `dpr-c`|15|ok
27 dec|split fixed100 nq|15|ok
27 dec|split fixed100 hp|15|ok
11 dec|rproj single `dpr-c-pruned_cn` *fretfulwill*|15|ok
21 dec|umap `dpr-c-pruned_cn` `--train-size 1000` *babyquest*|14|did not finish
21 dec|prec 1bit pca `dpr-c-pruned_cn` *filthysuit*|12|ok
21 dec|prec 1bit fagglo `dpr-c-pruned_cn` *filthysuit*|12|ok
21 dec|umap `dpr-c-pruned_cn` `--train-size 10000` *babyquest* (bad post-cn)|13|killed
21 dec|prec base `dpr-c-pruned_cn` {:,cn:`--post-cn`} *filthysuit*|12|ok
21 dec|feature agglomeration `dpr-c-pruned_cn` *babyquest*|14|ok
19 dec|autoencoder irrelevant `dpr-c-pruned_cn` `dpr-c_cn` `--post-cn` *uneasiness* {:accidentally stopped,2: pre on 13,3: continuation,4: redo }|14|ok
19 dec|rproj greedy `dpr-c-pruned_cn` *unusedoven* {:,cn:`--post-cn`|12|ok
19 dec|pca irrelevant `dpr-c-pruned_cn` `dpr-c_cn` `--post-cn` *uneasiness* {,2: pre}|13|ok, ok
18 dec|pca irrelevant `dpr-c-pruned_cn` `dpr-c-pruned_cn` `--post-cn` *uneasiness* (test pt 2)|13|ok
18 dec|autoencoder irrelevant `dpr-c-pruned_cn` `dpr-c-pruned_cn` `--post-cn` *uneasiness* (test pt 2)|14|stopped
18 dec|autoencoder base `dpr-c-pruned cn` `--model 1` `--dims 128` `--post-cn` (and without) *merequinoa_model_1_{cn,}_l1*|14|ok, ok
18 dec|autoencoder base `dpr-c-pruned_cn` `--model 2` `--dims 128` `--post-cn` (and without) *merequinoa_model_2_{cn,}_l1*|14|ok, ok
18 dec|autoencoder base `dpr-c-pruned_cn` `--model 3` `--dims 128` `--post-cn` (and without) *merequinoa_model_3_{cn,}_l1*|14|ok, ok
17 dec|pca base_big `dpr-c-pruned` *keyrespite_0*|12|bad code, ok
17 dec|pca base_big `dpr-c-pruned` `--norm` *keyrespite_1*|12|bad code, ok
17 dec|pca base_big `dpr-c-pruned` `--center` *keyrespite_2*|13|bad code, ok
17 dec|pca base_big `dpr-c-pruned` `--center --norm` *keyrespite_3*|13|bad code, ok
17 dec|autoencoder speed `dpr-c-pruned_cn` `--model 3` *puffyemery_gpu*|14|ok
17 dec|autoencoder speed `dpr-c-pruned_cn` `--model 3` *puffyemery*|14|ok
17 dec|pca speed `dpr-c-pruned_cn` `--model scikit` *puffyemery*|13|ok
17 dec|pca speed `dpr-c-pruned_cn` `--model torch gpu` *puffyemery*|13|ok
17 dec|pca speed `dpr-c-pruned_cn` `--model torch` *puffyemery*|13|ok
16 dec|autoencoder base `dpr-c-pruned` `--model 1` *merequinoa_0*|14|2 bad code, ok
16 dec|autoencoder base `dpr-c-pruned` `--model 1` `--norm` *merequinoa_1*|14|2 bad code, ok
16 dec|autoencoder base `dpr-c-pruned` `--model 1` `--center` *merequinoa_2*|13|2 bad code, ok
16 dec|autoencoder base `dpr-c-pruned` `--model 1` `--center --norm` *merequinoa_3*|13|2 bad code, ok
16 dec|autoencoder base `dpr-c-pruned cn` `--model 1` `--dims 128` *merequinoa_model_1*|13|2 bad code, ok
16 dec|autoencoder base `dpr-c-pruned_cn` `--model 2` `--dims 128` *merequinoa_model_2*|13|2 bad code, ok
16 dec|autoencoder base `dpr-c-pruned_cn` `--model 3` `--dims 128` *merequinoa_model_3*|13|2 bad code, ok
16 dec|autoencoder base `dpr-c-pruned_cn` `--model 1` `--dims 128` `--post-cn` *merequinoa_model_1_cn*|14|2 bad code, ok
16 dec|autoencoder base `dpr-c-pruned_cn` `--model 2` `--dims 128` `--post-cn` *merequinoa_model_2_cn*|14|2 bad code, ok
16 dec|autoencoder base `dpr-c-pruned_cn` `--model 3` `--dims 128` `--post-cn` *merequinoa_model_3_cn*|14|2 bad code, ok
14 dec|autoencoder base `dpr-c-pruned cn` `--post-cn` `--model 1` `--epochs 2` *merequinoa_model_1_e2*|13|ok
13 dec|uncompressed `--all` `bert-c bert-t`|14|ok
13 dec|uncompressed `--without-faiss` bert-t`|14|ok
13 dec|uncompressed `--all` `sbert-c sbert-t`|13|ok
13 dec|uncompressed `--all` `dpr-c dpr-t`|12|ok
11 dec|embd pruned `bert-c bert-t sbert-c sbert-t`|14|ok
11 dec|rproj `--dims 128` `dpr-c-pruned_cn` *yearlyquiz* (0, 1: `--post-cn`)|14|ok
11 dec|pca scaled `--dims 128` `dpr-c-pruned_cn` *neatwasting* (0, 1: `--post-cn`)|12|ok, ok
11 dec|pca `--dims 128` `dpr-c-pruned_cn` *loathsomewrapper* (0, 1: `--post-cn`)|13|ok
10 dec|rproj `dpr-c-pruned_cn` `--post-cn` *informedimmunity*|15|ok
10 dec|uncompressed `dpr-c-pruned` `--post-cn` *blueintersect*|15|ok
04 dec|pca base_big `dpr-c_cn` `--data-small pruned` `--post-cn` `--dims 128` *highuser*|12|test ok,2 failed
24 nov|pca base_big `dpr-c` `--data-small pruned` *tupperware*|13|killed
30 nov|autoencoder base `dpr-c_cn` `--post-cn` *merequinoa_bigold*|14|killed
07 dec|pca scaled `dpr-c_cn` `--data-small pruned` `--skip-loss` *measlygratitude_2*|15|ok
05 dec|pca scaled `dpr-c_cn` `--data-small pruned` `--post-cn` `--skip-loss` *measlygratitude_1*|15|ok
04 dec|pca scaled `dpr-c_cn-pruned` `--data-small pruned` `--skip-loss` *measlygratitude*|15|3 failed, ok
02 dec|rproj incremental greedy `dpr-c_cn` `--dims 128` *necessary heritage*|15|ok
28 nov|sparse projection base `dpr-c_cn` `--dims 128` *unequalapparel*, `base_big_rproj.out`|15|4 failed, ok
26 nov|autoencoder base (test with loss and multiple dims) `dpr-c-pruned_cn` `--post-cn` *pleasingkeep*|14|3 failed, ok
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
07 oct|uncompressed dpr-t `--without-faiss`|14|ok
07 oct|uncompressed bert-t `--without-faiss`|14|ok
07 oct|uncompressed bert-c `--without-faiss`|13|ok
18 sep|embd dpr-t|14|ok
18 sep|embd bert-t|14|ok
18 sep|embd bert-c|14|ok
07 oct|uncompressed sbert-c `--without-faiss`|13|ok
028 sep|embd sbert-c|15|ok, removed (quota)
04 oct|uncompressed sbert-t `--without-faiss`|13|restarted, ok
03 oct|uncompressed dpr-c_n|13|ok
03 oct|uncompressed dpr-c_csn|13|ok
03 oct|norm dpr-c_n|13|ok, removed (quota) 
02 oct|norm dpr-c_csn|13|ok
02 oct|norm dpr-c_c||ok, removed (quota)
02 oct|uncompressed dpr-c_cn||ok
02 oct|uncompressed dpr-c_cs||ok
02 oct|run dpr-c norm -cs||ok
02 oct|run dpr-c norm -c||fail quota
01 oct|run dpr-c norm -cn || ok
01 oct|run dpr-c norm -cn, -cs || multiple fails
01 oct|run dpr-c uncompressed `--all`||fail
28 sep|uncompressed dpr-c (fast+slow)||ok
01 oct|run dpr-c zscores (sequentially with `--std` and `--std --center`||fail
10 sep| embd for dpr-t||ok, removed (quota)
04 sep| embd for dpr-c||ok