# Benchmark results

## 1. BM25 vs TF-IDF (sample corpus)

| Scorer | Precision@5 | MRR | Avg latency (ms) |
|---|---|---|---|
| BM25 | 0.600 | 1.000 | 0.044 |
| TFIDF | 0.600 | 1.000 | 0.036 |

## 2. Skip pointers vs naive intersection

| Long list | Short list | Naive comparisons | Skip comparisons | Saved |
|---|---|---|---|---|
| 1,000 | 10 | 917 | 262 | 71% |
| 10,000 | 10 | 9,941 | 1,041 | 90% |
| 10,000 | 100 | 9,818 | 6,473 | 34% |
| 100,000 | 50 | 95,016 | 15,361 | 84% |
