# Simple OGB baselines
Improving upon the existing OGB baselines with basic techniques. :)

# ogbl_ddi
Adds JKnet style connections to basic GCN + trains for 500 epochs (instead of the default 200)

Hyperparameters chosen are the ones set by default.

```
python gnn.py
------------
Hits@10
All runs:
Highest Train: 73.28 ± 1.71
Highest Valid: 64.00 ± 1.57
  Final Train: 73.25 ± 1.72
   Final Test: 37.09 ± 8.99
Hits@20
All runs:
Highest Train: 77.26 ± 1.07
Highest Valid: 67.76 ± 0.95
  Final Train: 77.26 ± 1.07
   Final Test: 60.56 ± 8.69
Hits@30
All runs:
Highest Train: 78.85 ± 0.91
Highest Valid: 69.31 ± 0.78
Final Train: 78.85 ± 0.91
Final Test: 73.71 ± 9.19
```

# ogbn_arxiv
## Submission 1
Adds JKNet style residuals, GCNII style residuals, and also concatenates node2vec embeddings.

```
python node2vec.py --batch_size 128
python gnn.py --use_node_embedding --num_layers 2

Highest Train: 86.95 ± 0.21
Highest Valid: 74.14 ± 0.08
Final Train: 81.86 ± 1.36
Final Test: 72.78 ± 0.13
```

## Submission 2
Adds JKNet style residuals, GCNII style residuals, and uses 6 layers.
```
python gnn.py  --num_layers 6 --hidden_channels 128 --epochs 500

Highest Train: 78.01 ± 0.10
Highest Valid: 73.82 ± 0.07
Final Train: 77.64 ± 0.29
Final Test: 72.86 ± 0.16
```

# ogbn_products
Adds GCNII style residuals and uses 3 layers. Interestingly, these improvements improve ClusterGCN but not GraphSAINT. :/

```
python cluster_gcn.py --num_layers 3

All runs:
Highest Train: 93.69 ± 0.06
Highest Valid: 91.88 ± 0.08
Final Train: 93.53 ± 0.12
Final Test: 79.71 ± 0.42
```


