# ogb_ddi
A simple baseline for OGBL DDI

Adds JKnet style connections + trains for 500 epochs (instead of the default 200)

Hyperparameters chosen are the ones set by default.

```
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
