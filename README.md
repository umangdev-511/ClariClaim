# ClariClaim

ClariClaim is a research prototype for reviewing damage claims from photographs, a short claim description and basic claim-history context. It turns that evidence into a structured decision with an explanation, supporting image IDs, quality warnings and a severity estimate.

The current build supports three object categories: cars, laptops and packages. The main product direction is vehicle damage inspection and claim documentation.

## What it does

For each claim, ClariClaim:

- extracts the claimed object, part and damage type from the description;
- inspects one or more submitted images;
- checks whether the claimed part is visible and usable for review;
- identifies mismatched, low-quality or possibly manipulated evidence;
- classifies the claim as `supported`, `contradicted` or `not_enough_information`;
- returns a short explanation and the image IDs that support the decision; and
- validates every result against a fixed output schema.

ClariClaim assists review. It does not approve claims or replace a surveyor, service advisor or insurer.

## Current stage

ClariClaim is at Technology Readiness Level 3: an experimental proof of concept demonstrated in a controlled environment.

Evidence for this assessment includes a working end-to-end pipeline, deterministic decision logic, schema validation, unit tests, cached repeat runs and evaluation on a small labelled development set. It has not been tested in a live service-centre or insurance workflow, so claims about production accuracy, fraud detection or commercial impact would be premature.

## Architecture

```text
Claim description + images + optional history
                     |
                     v
          Multimodal perception layer
        - claim and part extraction
        - image inspection
        - quality and authenticity signals
                     |
                     v
          Deterministic decision layer
        - evidence sufficiency
        - object and part matching
        - multi-image aggregation
        - risk flags and severity
                     |
                     v
            Schema-validated output
```

Only `src/perception.py` calls a model. The downstream decision and validation layers are ordinary Python, which makes the final policy easier to inspect and test.

## Repository structure

```text
src/
  perception.py       Model-backed claim and image reading
  decide.py           Deterministic evidence and decision rules
  schema.py           Output schema and validation
  main.py             Batch pipeline entry point
evaluation/
  run_eval.py         Labelled-set evaluation
  naive_baseline.py   First-image comparison baseline
tests/                Unit tests for decisions and schema validation
dataset/              Input templates and evidence requirements
docs/evaluation.md    Recorded controlled evaluation results
```

## Set up

ClariClaim requires Python 3.10 or newer.

```bash
git clone https://github.com/umangdev-511/ClariClaim.git
cd ClariClaim
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your Anthropic API key to `.env`:

```text
ANTHROPIC_API_KEY=your_key_here
```

You can also set `MODEL` to a compatible model name. Never commit the `.env` file.

## Add input data

The repository does not publish the private development images. Place data you are permitted to use under `dataset/` using the formats described in [dataset/README.md](dataset/README.md).

At minimum, the batch runner expects:

```text
dataset/claims.csv
dataset/user_history.csv
dataset/images/...
```

Run the pipeline from the repository root:

```bash
python3 -m src.main
```

The result is written to `output.csv`. Perception results are cached locally in `.cache/perception.json`, so unchanged inputs do not require repeated model calls.

## Tests

The unit tests do not call an external model:

```bash
pytest -q
```

The controlled evaluation results and their limits are recorded in [docs/evaluation.md](docs/evaluation.md).

## Privacy and safety

Damage-claim images may contain number plates, faces, addresses or location metadata. A production system should obtain consent, minimise stored data, remove unnecessary metadata, encrypt records, restrict access and define a deletion schedule.

Model output can be wrong. Low-confidence and high-risk cases should go to a human reviewer, and the final decision should retain the submitted evidence and the reasons used.

## Author

Umang Pandey

