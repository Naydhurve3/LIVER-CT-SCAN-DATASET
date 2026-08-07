# Complete Consolidated Results

All values below come from saved JSON/CSV artifacts under `Practice/*_outputs`. Values close to zero are shown as `≈0`.

## Main modeling results

| Experiment | Best epoch | Global Dice | Mean patient Dice | V104 Dice | V116 Dice | Q1 detection | Positive empty | Empty FP | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16-slice overfit | 76 | `0.816306` | — | — | — | — | — | — | Pass |
| 5-epoch smoke | 5 | `0.461613` | — | — | — | — | `27.74%` | `14.42%` | Pass |
| Patient-aware baseline | 9 | `0.558357` | `0.383069` | `0.197613` | `≈0` | — | — | — | Revise |
| Lesion-balanced sampler | 7 | `0.524200` | `0.356189` | `0.152444` | `0.001856` | `53.99%` | `19.29%` | `17.83%` | Fail |
| Pure Focal-Tversky | 1 | `≈0` | `≈0` | `≈0` | `≈0` | `0%` | `100%` | `0%` | Collapsed |
| Composite loss | 5 | `0.568154` | `0.322939` | `0.005005` | `0.051481` | `29.66%` | `41.46%` | `3.03%` | Fail |
| Organ-assisted intensity | 8 | `0.546132` | `0.406915` | `0.586500` | `0.026782` | `60.08%` | `8.25%` | `47.42%` | Fail |
| Adjacent-slice 2.5D | 10 | `0.553527` | `0.355938` | `0.000636` | `0.000932` | `25.48%` | `42.99%` | `2.37%` | Fail |
| 3D post-processing | validation only | — | `0.406964` | `0.586555` | `0.026794` | `60.08%` | `8.25%` | `47.43%` | Fail |
| Multi-task liver/tumor | 8 | — | `0.332854` | `≈0` | `≈0` | `28.14%` | `46.16%` | `3.25%` | Fail |

## Multi-task epoch trajectory

| Epoch | Train loss | Val loss | Liver Dice | Mean patient Dice | V104 | V116 | Positive empty | Empty FP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `0.3895` | `0.4472` | `0.8461` | `0.2960` | `0.0159` | `0.0027` | `20.63%` | `15.64%` |
| 2 | `0.2065` | `0.4387` | `0.8669` | `0.3028` | `≈0` | `≈0` | `46.07%` | `3.40%` |
| 3 | `0.1287` | `0.4406` | `0.8801` | `0.2959` | `≈0` | `0.0026` | `49.42%` | `2.78%` |
| 4 | `0.1243` | `0.4352` | `0.8848` | `0.3138` | `≈0` | `≈0` | `50.00%` | `3.00%` |
| 5 | `0.1212` | `0.4352` | `0.8884` | `0.3133` | `≈0` | `0.0004` | `47.50%` | `3.09%` |
| 6 | `0.1182` | `0.4178` | `0.8785` | `0.3039` | `0.0286` | `≈0` | `51.15%` | `2.59%` |
| 7 | `0.1160` | `0.4236` | `0.8902` | `0.3180` | `0.00003` | `≈0` | `49.04%` | `2.24%` |
| 8 | `0.1126` | `0.4099` | `0.8863` | `0.3329` | `≈0` | `≈0` | `46.16%` | `3.25%` |
| 9 | `0.1120` | `0.4103` | `0.8796` | `0.3296` | `0.00014` | `0.00014` | `42.42%` | `3.94%` |
| 10 | `0.1102` | `0.4120` | `0.8881` | `0.3296` | `≈0` | `0.00012` | `45.87%` | `2.89%` |

Interpretation:

- Optimization was stable.
- Validation improvement plateaued after epoch 8.
- The model traded recall for specificity.
- Continuing to epoch 15 is not justified without a changed hypothesis.

## Best multi-task patient results

| Volume | Tumor pixels | Predicted pixels | Dice | Positive-slice recall | Positive empty | Empty FP |
|---:|---:|---:|---:|---:|---:|---:|
| 104 | 69,032 | 307 | `≈0` | `0%` | `97.54%` | `0.76%` |
| 116 | 152,763 | 534 | `≈0` | `0%` | `96.24%` | `1.42%` |
| 112 | 295 | 4,107 | `0.0672` | `38.46%` | `34.62%` | `4.97%` |
| 107 | 1,473 | 4,072 | `0.0775` | `21.13%` | `77.46%` | `8.71%` |
| 111 | 1,830 | 3,468 | `0.3103` | `40.22%` | `56.52%` | `3.59%` |
| 113 | 21,076 | 24,092 | `0.5811` | `74.81%` | `20.00%` | `2.28%` |
| 108 | 363,698 | 224,790 | `0.6005` | `71.29%` | `22.28%` | `0.46%` |
| 109 | 18,200 | 19,407 | `0.6573` | `77.10%` | `16.03%` | `3.36%` |
| 110 | 31,120 | 22,911 | `0.7017` | `80.77%` | `19.23%` | `0.15%` |

Volumes 105, 106, 114, and 115 have no tumor ground truth. At the best multi-task epoch, volume 114 had no predicted tumor, while the other empty-tumor volumes retained some false-positive predictions.

## Best multi-task size-quartile results

| Quartile | Slices | Mean Dice | Median Dice | Detection | Predicted empty |
|---|---:|---:|---:|---:|---:|
| Q1 smallest | 263 | `0.1556` | `≈0` | `28.14%` | `66.16%` |
| Q2 | 258 | `0.3273` | `0.3441` | `58.14%` | `36.43%` |
| Q3 | 260 | `0.4274` | `0.6255` | `61.15%` | `37.31%` |
| Q4 largest | 261 | `0.3250` | `≈0` | `49.81%` | `44.44%` |

The Q4 median near zero is important: large tumor burden did not guarantee detection. Volumes 104 and 116 dominate this failure.

## Appearance-shift diagnostic

| Measure | Value |
|---|---:|
| Sampled slices | 3,106 |
| Training P95 shift threshold | `6.4022` |
| Volume 104 shift score | `6.6768` |
| Volume 116 shift score | `3.7398` |

Volume 104 was an appearance outlier. Volume 116 was not explained by that specific aggregate shift score.

