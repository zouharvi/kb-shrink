Experiments on /data/kilt-hp/dpr-c-5000.embd_cn

## Uncompressed

```
rprec_ip (fast) 0.45405500000000004
rprec_l2 (fast) 0.4524616666666667
```

# Reduction to 128

## PCA

```
PCA-D (128)           0.00021 0.00029 0.422 0.425
PCA-Q (128)           0.00034 0.00016 0.429 0.419
PCA-DQ (128)          0.00022 0.00021 0.423 0.422
```

## Similarity Distillation

```
random_n=20, close_n=5, batchSize=512, similarity=ip-ip
epoch [57/10000], loss_l2: 0.000173140, rprec_ip: 0.437, rprec_l2: 0.432

random_n=40, close_n=5, batchSize=512, similarity=ip-ip
epoch [50/10000], loss_l2: 0.000123448, rprec_ip: 0.434, rprec_l2: 0.433
epoch [242/10000], loss_l2: 0.000129274, rprec_ip: 0.438, rprec_l2: 0.434
epoch [267/10000], loss_l2: 0.000098024, rprec_ip: 0.439, rprec_l2: 0.434

random_n=40, close_n=10, batchSize=512, similarity=ip-ip
epoch [50/10000], loss_l2: 0.000138991, rprec_ip: 0.434, rprec_l2: 0.432
epoch [242/10000], loss_l2: 0.000135034, rprec_ip: 0.437, rprec_l2: 0.433

random_n=40, close_n=10, batchSize=512, similarity=l2-l2
epoch [50/10000], loss_l2: 0.000215809, rprec_ip: 0.426, rprec_l2: 0.430
epoch [242/10000], loss_l2: 0.000202751, rprec_ip: 0.432, rprec_l2: 0.435

random_n=40, close_n=5, batchSize=512, similarity=ip-ip, not-merge
epoch [50/10000], loss_l2: 0.000305185, rprec_ip: 0.431, rprec_l2: 0.422
epoch [114/10000], loss_l2: 0.000113884, rprec_ip: 0.444, rprec_l2: 0.432
```

Whole data:

```
random_n=20, close_n=5, batchSize=512, similarity=ip-ip
Last batch: 250
epoch [1/10000], loss_l2: 0.001022680, rprec_ip: 0.221, rprec_l2: 0.231
Last batch: 250
epoch [2/10000], loss_l2: 0.000439441, rprec_ip: 0.268, rprec_l2: 0.261
Last batch: 250
epoch [3/10000], loss_l2: 0.000268612, rprec_ip: 0.280, rprec_l2: 0.271
Last batch: 250
epoch [4/10000], loss_l2: 0.000231095, rprec_ip: 0.286, rprec_l2: 0.277
Last batch: 250
epoch [5/10000], loss_l2: 0.000183212, rprec_ip: 0.289, rprec_l2: 0.280
Last batch: 250
epoch [6/10000], loss_l2: 0.000169597, rprec_ip: 0.291, rprec_l2: 0.281
Last batch: 250
epoch [7/10000], loss_l2: 0.000181348, rprec_ip: 0.292, rprec_l2: 0.282
```

# Reduction to 64

## PCA

```
Method                Loss-D  Loss-Q  IPRPR L2RPR
PCA-D (64)            0.00043 0.00056 0.327 0.348
PCA-Q (64)            0.00060 0.00039 0.341 0.351
PCA-DQ (64)           0.00044 0.00046 0.327 0.345
```

## Autoencoder

```
epoch [990/1000], loss_l2: 0.0004401, rprec_ip: 0.335, rprec_l2: 0.351
```


## Similarity Distillation

```
random_n=20, close_n=5, batchSize=512, similarity=ip-ip
epoch [84/10000], loss_l2: 0.000841000, rprec_ip: 0.381, rprec_l2: 0.379

random_n=20, close_n=5, batchSize=512, similarity=ip-ip, weights=0.5,0.5
epoch [124/10000], loss_l2: 0.000619625, rprec_ip: 0.378, rprec_l2: 0.377

random_n=20, close_n=5, batchSize=512, similarity=ip-ip, weights=0.95,0.05
epoch [40/10000], loss_l2: 0.001004403, rprec_ip: 0.368, rprec_l2: 0.371

random_n=20, close_n=5, batchSize=128, learningRate=0.001, similarity=ip-ip, weights=0.5,0.5
epoch [27/10000], loss_l2: 0.000848025, rprec_ip: 0.377, rprec_l2: 0.377

random_n=20, close_n=5, batchSize=128, learningRate=0.001, similarity=ip-ip, weights=dynamic,center@5
epoch [20/10000], loss_l2: 0.001266007, rprec_ip: 0.373, rprec_l2: 0.379
```