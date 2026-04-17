# HLE Evaluation – Gemini 2.5 Flash graded by o3-mini

Adapted from [centerforaisafety/hle](https://github.com/centerforaisafety/hle).

## Files

| File | Purpose |
|---|---|
| `run_full_pipeline.py` | **Start here** – runs both steps end-to-end |
| `run_gemini_predictions.py` | Step 1: Gemini 2.5 Flash answers every HLE question |
| `run_judge_results.py` | Step 2: o3-mini judges each answer |
| `requirements.txt` | Python dependencies |

## Quick start

```bash
pip install -r requirements.txt

# Full 2 500-question eval (expect ~$30-60 in API costs)
python run_full_pipeline.py \
    --google_api_key  YOUR_GOOGLE_KEY \
    --openai_api_key  YOUR_OPENAI_KEY

# Quick smoke-test with 10 questions
python run_full_pipeline.py \
    --google_api_key  YOUR_GOOGLE_KEY \
    --openai_api_key  YOUR_OPENAI_KEY \
    --max_samples 10
```

## Run steps individually

```bash
# Step 1 – predictions (resumes automatically if interrupted)
python run_gemini_predictions.py \
    --google_api_key YOUR_GOOGLE_KEY \
    --dataset cais/hle \
    --max_completion_tokens 8192 \
    --num_workers 20

# Step 2 – judge
python run_judge_results.py \
    --openai_api_key YOUR_OPENAI_KEY \
    --dataset cais/hle \
    --predictions hle_gemini-2.5-flash-preview-04-17.json \
    --num_workers 50
```

## Output files

After running, you'll find:

- `hle_gemini-2.5-flash-preview-04-17.json` — raw model responses + token usage
- `judged_hle_gemini-2.5-flash-preview-04-17.json.json` — responses + judge verdicts
- `judged_hle_gemini-2.5-flash-preview-04-17.json_summary.json` — final accuracy + calibration error

## Sample output

```
*** Metrics ***
Accuracy: X.XX% +/- Y.YY% | n = 2500
Calibration Error: ZZ.Z
```

## Key design decisions

**Gemini via REST, not SDK** – `run_gemini_predictions.py` calls the
`generativelanguage.googleapis.com` REST endpoint directly with `httpx`,
avoiding SDK version conflicts and keeping concurrency control simple.

**Resume-safe** – both scripts skip questions already answered/judged,
so you can safely re-run after a rate-limit interruption.

**Identical judge logic** – `run_judge_results.py` is a line-for-line port
of the upstream HLE judge, so scores are directly comparable to published results.

**Image support** – multi-modal questions are handled: the image is fetched
and base64-encoded into the Gemini request automatically.

## Tuning concurrency

| Flag | Default | Guidance |
|---|---|---|
| `--num_workers_gemini` | 20 | Increase if you have high Gemini quota |
| `--num_workers_judge` | 50 | Increase if you have high OpenAI tier-4+ quota |

Both scripts handle rate-limit errors gracefully; just re-run and they'll pick up where they left off.
