# Phase 1 baselines

Generated 2026-10-01 on the **val** split. Each model's threshold was chosen on validation data to catch 90% of spoilers. Compare models by false-blur rate and PR-AUC.

> Data: ../data/processed/goodreads-5pct.jsonl (train 652935, val 71603, test 76768 sentences, split by title).

| Model | Sentences | Spoilers | Threshold | Recall | False-blur rate | Precision | PR-AUC | Calibration error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| keyword-rules | 71603 | 2104 | 0.000 | 1.000 | 1.000 | 0.029 | 0.033 | 0.043 |
| tfidf-logreg | 71603 | 2104 | 0.020 | 0.900 | 0.533 | 0.049 | 0.135 | 0.005 |

## Slices: keyword-rules

**By source**

| Group | Sentences | Spoilers | Recall | False-blur rate |
| --- | --- | --- | --- | --- |
| goodreads | 71603 | 2104 | 1.000 | 1.000 |

**By length**

| Group | Sentences | Spoilers | Recall | False-blur rate |
| --- | --- | --- | --- | --- |
| long (>25) | 11203 | 392 | 1.000 | 1.000 |
| medium (10-25) | 36574 | 987 | 1.000 | 1.000 |
| short (<10 words) | 23826 | 725 | 1.000 | 1.000 |

**By title mention**

| Group | Sentences | Spoilers | Recall | False-blur rate |
| --- | --- | --- | --- | --- |
| doesn't name it | 45808 | 1118 | 1.000 | 1.000 |
| names the title | 2113 | 22 | 1.000 | 1.000 |
| title unknown | 23682 | 964 | 1.000 | 1.000 |


## Slices: tfidf-logreg

**By source**

| Group | Sentences | Spoilers | Recall | False-blur rate |
| --- | --- | --- | --- | --- |
| goodreads | 71603 | 2104 | 0.900 | 0.533 |

**By length**

| Group | Sentences | Spoilers | Recall | False-blur rate |
| --- | --- | --- | --- | --- |
| long (>25) | 11203 | 392 | 0.921 | 0.568 |
| medium (10-25) | 36574 | 987 | 0.900 | 0.528 |
| short (<10 words) | 23826 | 725 | 0.890 | 0.525 |

**By title mention**

| Group | Sentences | Spoilers | Recall | False-blur rate |
| --- | --- | --- | --- | --- |
| doesn't name it | 45808 | 1118 | 0.910 | 0.524 |
| names the title | 2113 | 22 | 0.955 | 0.394 |
| title unknown | 23682 | 964 | 0.888 | 0.564 |
