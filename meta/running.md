date|action|machine|status
-|-|-|-
||||running
7 oct|uncompressed dpr-t `--without-faiss`|14|running
7 oct|uncompressed bert-t `--without-faiss`|14|running
7 oct|uncompressed bert-c `--without-faiss`|13|running
||||
||||
||||finished
||||
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
