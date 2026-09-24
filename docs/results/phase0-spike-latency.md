# Phase 0 spike: in-browser latency

**GO.** Multi-threaded (default) reaches a p95 of 7.34 ms per sentence, within the 15 ms budget.

> The spike model has **random weights**. These numbers measure the speed of the
> MiniLM-L6 architecture only. They say nothing about spoiler detection quality.

## Setup

| Item | Value |
| --- | --- |
| Model | BertForSequenceClassification, MiniLM-L6-H384 shape |
| Parameters | 22,713,986 |
| Model file, int8 | 22.91 MB |
| Runtime | transformers.js with ONNX Runtime Web, WASM backend |
| Mean tokens per sentence | 18.1, truncated at 64 |
| Iterations per batch size | 60, after 5 warm-up calls |
| Machine | 4 logical CPUs, headless Chromium in a shared cloud container |
| User agent | `Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/141.0.0.0 Safari/537.36` |

## Results

### Single thread

1 thread(s), cross-origin isolated: true, model load 1320 ms.

| Batch size | p50 ms per sentence | p95 ms per sentence | p50 ms per call | p95 ms per call |
| --- | --- | --- | --- | --- |
| 1 | 18.20 | 52.55 | 18.2 | 52.6 |
| 8 | 13.72 | 15.94 | 109.8 | 127.5 |
| 16 | 14.32 | 16.92 | 229.2 | 270.7 |

### Multi-threaded (default)

4 thread(s), cross-origin isolated: true, model load 1484 ms.

| Batch size | p50 ms per sentence | p95 ms per sentence | p50 ms per call | p95 ms per call |
| --- | --- | --- | --- | --- |
| 1 | 17.62 | 25.67 | 17.6 | 25.7 |
| 8 | 5.35 | 14.37 | 42.8 | 114.9 |
| 16 | 4.68 | 7.34 | 74.8 | 117.4 |

## How to reproduce

```bash
make bench
```

A shared cloud container is noisy, so treat these numbers as indicative. Rerun on a
typical laptop before launch.
