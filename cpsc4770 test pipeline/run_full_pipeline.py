#!/usr/bin/env python3
"""
run_full_pipeline.py – one command to run the entire HLE eval for Gemini 2.5 Flash.

Steps:
  1. Fetch Gemini 2.5 Flash answers for every HLE question
  2. Grade with o3-mini
  3. Print final accuracy + calibration error

Usage:
    python run_full_pipeline.py \
        --google_api_key  YOUR_GOOGLE_KEY \
        --openai_api_key  YOUR_OPENAI_KEY \
        [--max_samples 20]          # quick smoke-test
        [--num_workers_gemini 20]   # tune to your Gemini quota
        [--num_workers_judge  50]   # tune to your OpenAI quota
        [--max_completion_tokens 8192]
        [--dataset cais/hle]
"""

import argparse
import subprocess
import sys
import os

MODEL_SLUG = "gemini-2.5-flash-lite"
PREDICTIONS_FILE = f"hle_{MODEL_SLUG}.json"


def run(cmd: list[str]):
    print("\n" + "=" * 70)
    print("Running:", " ".join(cmd))
    print("=" * 70)
    result = subprocess.run(cmd, check=True)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Full HLE pipeline: Gemini 2.5 Flash → o3-mini judge"
    )
    parser.add_argument("--google_api_key", required=True)
    parser.add_argument("--openai_api_key", required=True)
    parser.add_argument("--dataset", default="cais/hle")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit to N questions (omit for full 2500-question eval)")
    parser.add_argument("--max_completion_tokens", type=int, default=8192)
    parser.add_argument("--num_workers_gemini", type=int, default=20,
                        help="Concurrent Gemini requests")
    parser.add_argument("--num_workers_judge", type=int, default=50,
                        help="Concurrent o3-mini judge requests")
    parser.add_argument("--judge", default="o3-mini-2025-01-31")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ------------------------------------------------------------------
    # Step 1: Generate Gemini predictions
    # ------------------------------------------------------------------
    predict_cmd = [
        sys.executable,
        os.path.join(script_dir, "run_gemini_predictions.py"),
        "--google_api_key", args.google_api_key,
        "--dataset", args.dataset,
        "--max_completion_tokens", str(args.max_completion_tokens),
        "--num_workers", str(args.num_workers_gemini),
    ]
    if args.max_samples:
        predict_cmd += ["--max_samples", str(args.max_samples)]

    run(predict_cmd)

    # ------------------------------------------------------------------
    # Step 2: Judge with o3-mini
    # ------------------------------------------------------------------
    judge_cmd = [
        sys.executable,
        os.path.join(script_dir, "run_judge_results.py"),
        "--openai_api_key", args.openai_api_key,
        "--dataset", args.dataset,
        "--predictions", PREDICTIONS_FILE,
        "--num_workers", str(args.num_workers_judge),
        "--judge", args.judge,
    ]

    run(judge_cmd)
    print("\n✅  Pipeline complete.")


if __name__ == "__main__":
    main()
