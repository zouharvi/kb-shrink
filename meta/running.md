date|action|machine|status
-|-|-|-
||||running
||||
28 sep|embd sbert-c|15|
18 sep|embd dpr-t|14|
18 sep|embd bert-t|14|
18 sep|embd bert-c|14|
4 oct|uncompressed sbert-t|13|running
||||
||||finished
||||
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
4 sep| full embd for dpr-c||ok
