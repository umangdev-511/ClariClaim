# Data format

ClariClaim reads local CSV files and image paths. Development images are not included in this public repository.

## `claims.csv`

Required columns:

```text
user_id,image_paths,user_claim,claim_object
```

Use semicolons to separate multiple image paths. Supported object values in the current prototype are `car`, `laptop` and `package`.

Example:

```csv
user_id,image_paths,user_claim,claim_object
demo_001,dataset/images/demo_001/front.jpg;dataset/images/demo_001/close.jpg,The front bumper has a dent.,car
```

## `user_history.csv`

Optional history columns:

```text
user_id,past_claim_count,accept_claim,manual_review_claim,rejected_claim,last_90_days_claim_count,history_flags,history_summary
```

History can add a review flag. It does not override clear image evidence or change `claim_status` by itself.

## Labelled evaluation data

`evaluation/run_eval.py` expects `dataset/sample_claims.csv` with the input columns plus the expected output fields defined in `src/schema.py`. Use only images that you own or are permitted to process.

