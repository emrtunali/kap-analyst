# Evaluation results

_Environment: Darwin arm64, Python 3.13.11, 948 chunks, 91 questions_

## Retrieval (page-level)

| Retriever | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 | nDCG@10 | ms/query |
|---|---|---|---|---|---|---|---|
| BM25 (no stemming) | 0.418 | 0.626 | 0.692 | 0.758 | 0.536 | 0.548 | 0.4 |
| BM25 + Turkish F5 stemming | 0.451 | 0.670 | 0.758 | 0.802 | 0.576 | 0.592 | 0.3 |
| Dense · multilingual-e5-small | 0.341 | 0.495 | 0.571 | 0.703 | 0.441 | 0.447 | 15.7 |
| Dense · bge-m3 | 0.527 | 0.703 | 0.791 | 0.879 | 0.639 | 0.652 | 30.8 |
| Hybrid RRF (BM25-F5 + bge-m3) | 0.549 | 0.703 | 0.780 | 0.846 | 0.647 | 0.654 | 26.9 |
| Hybrid + rerank · mmarco-mMiniLMv2-L12-H384-v1 | 0.615 | 0.802 | 0.890 | 0.934 | 0.722 | 0.714 | 308.2 |
| **Hybrid + rerank + company routing (default)** | 0.615 | 0.813 | 0.912 | 0.934 | 0.725 | 0.716 | 302.3 |
| Hybrid + rerank · bge-reranker-v2-m3 + routing | 0.681 | 0.890 | 0.901 | 0.923 | 0.782 | 0.770 | 2872.0 |


### By query language (Hit@5)

| Retriever | en | tr |
|---|---|---|
| BM25 (no stemming) | 0.200 (n=15) | 0.789 (n=76) |
| BM25 + Turkish F5 stemming | 0.200 (n=15) | 0.868 (n=76) |
| Dense · multilingual-e5-small | 0.333 (n=15) | 0.618 (n=76) |
| Dense · bge-m3 | 0.733 (n=15) | 0.803 (n=76) |
| Hybrid RRF (BM25-F5 + bge-m3) | 0.400 (n=15) | 0.855 (n=76) |
| Hybrid + rerank · mmarco-mMiniLMv2-L12-H384-v1 | 0.667 (n=15) | 0.934 (n=76) |
| **Hybrid + rerank + company routing (default)** | 0.667 (n=15) | 0.961 (n=76) |
| Hybrid + rerank · bge-reranker-v2-m3 + routing | 0.667 (n=15) | 0.947 (n=76) |

## Statement extraction

Exact-match accuracy: **90/90** values (100.0%); correct source page: 45/45 items.


## Ratios (computed from extracted values)

| Ratio | 2024 | Formula |
|---|---|---|
| gross_margin | 0.1917 | Brüt kâr / Hasılat |
| operating_margin | 0.1078 | Esas faaliyet kârı / Hasılat |
| net_margin | 0.1521 | Dönem kârı / Hasılat |
| roe | 0.1994 | Ana ortaklık payı net kâr / Ortalama ana ortaklık özkaynakları |
| roa | 0.0926 | Dönem kârı / Ortalama toplam varlıklar |
| current_ratio | 1.0070 | Dönen varlıklar / Kısa vadeli yükümlülükler |
| liabilities_to_equity | 1.0582 | (Kısa + uzun vadeli yükümlülükler) / Toplam özkaynaklar |
| revenue_growth | 0.4779 | Hasılat (t) / Hasılat (t-1) − 1 |
| capex_to_revenue | 0.0590 | &#124;Duran varlık alımları&#124; / Hasılat |
| cash_conversion | 1.1960 | İşletme faaliyetlerinden net nakit / Dönem kârı |
