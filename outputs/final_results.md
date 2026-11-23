# Final results (single evaluation of frozen models)

## test: 10,000 reviews (4,900 neg / 5,100 pos)

| Model | Macro-F1 | Accuracy | Neg F1 | Pos F1 | Inference (s) |
|---|---:|---:|---:|---:|---:|
| Majority class | 0.338 | 0.510 | 0.000 | 0.675 | – |
| TF-IDF + LR | 0.906 | 0.906 | 0.905 | 0.906 | 0.18 |
| GRU seed 42 | 0.909 | 0.909 | 0.907 | 0.910 | 1.03 |
| GRU seed 1 | 0.907 | 0.907 | 0.907 | 0.906 | 1.19 |
| GRU seed 2 | 0.910 | 0.910 | 0.909 | 0.912 | 1.10 |
| GRU mean ± sd | 0.908 ± 0.002 | 0.908 | | | 1.11 |

## unseen_zenithBank: 12,041 reviews (7,069 neg / 4,972 pos)

| Model | Macro-F1 | Accuracy | Neg F1 | Pos F1 | Inference (s) |
|---|---:|---:|---:|---:|---:|
| Majority class | 0.292 | 0.413 | 0.000 | 0.585 | – |
| TF-IDF + LR | 0.883 | 0.887 | 0.906 | 0.859 | 0.28 |
| GRU seed 42 | 0.885 | 0.889 | 0.907 | 0.862 | 1.52 |
| GRU seed 1 | 0.883 | 0.888 | 0.908 | 0.858 | 1.24 |
| GRU seed 2 | 0.884 | 0.888 | 0.906 | 0.861 | 1.34 |
| GRU mean ± sd | 0.884 ± 0.001 | 0.888 | | | 1.37 |

## Training cost (same 2-core CPU)

- TF-IDF + LR: 2.56 s, peak process memory 351 MB
- GRU: [305.7, 235.5, 359.0] s per seed, best epochs [3, 2, 4], peak memory [952, 961, 967] MB, 1,140,049 parameters
