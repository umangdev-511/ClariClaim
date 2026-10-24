# Controlled evaluation

The prototype was evaluated on a local labelled development set containing 20 claims. The images are not published because their redistribution rights have not been established.

## Recorded results

| Field | Correct | Total | Exact match |
|---|---:|---:|---:|
| Claim status | 20 | 20 | 100% |
| Evidence standard met | 20 | 20 | 100% |
| Object part | 16 | 20 | 80% |
| Supporting image IDs | 15 | 20 | 75% |
| Valid image | 17 | 20 | 85% |
| Risk flags | 10 | 20 | 50% |
| Issue type | 9 | 20 | 45% |
| Severity | 8 | 20 | 40% |
| All compared fields | 135 | 180 | 75% |

All 20 output rows passed the schema validator.

The decision pipeline was also compared with a simple baseline that trusts the first machine-readable image. The baseline classified 13 of 20 claim statuses correctly. ClariClaim classified all 20 correctly on the same cached perception output.

## What the results mean

The result supports a controlled proof of concept. It shows that part matching, evidence checks and multi-image aggregation can improve final claim-status decisions on this small set.

It does not establish production accuracy. The set is small, it was used during development, the image distribution is narrow and the secondary fields remain inconsistent. A proper next evaluation needs held-out vehicle-only data, independent labelling, subgroup analysis, calibration and testing with real service advisors.

## Reproducing an evaluation

Prepare a labelled `dataset/sample_claims.csv` and corresponding permitted images, then run:

```bash
python3 -m evaluation.run_eval
```

The evaluator reports per-field exact match, a claim-status confusion matrix and schema violations. Perception output is cached so decision-rule changes can be tested without paying for the same model calls again.

