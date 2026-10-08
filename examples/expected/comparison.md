# Paired comparison: `sample_predictions.jsonl` (A) vs `sample_baseline.jsonl` (B)

- Answers scored by both models: 397 from 40 questions
- A SHA-256: `fb239a1ab18fd121b6306c064cf6c0074ac6973a4a63a22cecd924d8ee5f53fc`
- B SHA-256: `05830eadc8402527e2a24514d9dd2f918cdd5ed672bdfff7c989a32edee738ac`
- Intervals and p-values: paired bootstrap over questions, 2000 resamples, seed 0
- Tool: asag-eval 0.1.0

| Metric | A | B | A - B | 95% CI | p |
| --- | --- | --- | --- | --- | --- |
| Quadratic weighted kappa (QWK) | 0.800 | 0.530 | 0.270 | [0.192, 0.351] | 0.0010 |
| Cohen's kappa | 0.812 | 0.550 | 0.261 | [0.195, 0.333] | 0.0010 |
| Macro-F1 | 0.876 | 0.701 | 0.174 | [0.129, 0.226] | 0.0010 |
| Accuracy | 0.877 | 0.705 | 0.171 | [0.127, 0.219] | 0.0010 |
| Mean absolute error | 0.181 | 0.428 | -0.247 | [-0.316, -0.184] | 0.0010 |
| Extreme error rate | 0.058 | 0.134 | -0.076 | [-0.105, -0.046] | 0.0010 |
