#!/usr/bin/env python3
"""
fetch_questions.py — Download the real HLE dataset from HuggingFace.

Writes hle_dataset.json, which run_eval.py loads with --questions hle_dataset.json.

Usage:
  pip install datasets huggingface_hub
  HF_TOKEN=<token> python3 fetch_questions.py
  HF_TOKEN=<token> python3 fetch_questions.py --samples 100 --text-only

Get a free HF token at: https://huggingface.co/settings/tokens
"""

import argparse
import json
import os
import random
import sys
from collections import Counter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples",    type=int, default=50)
    p.add_argument("--output",     default="hle_dataset.json")
    p.add_argument("--text-only",  action="store_true", default=True,
                   help="Skip image questions (default: on)")
    p.add_argument("--no-text-only", dest="text_only", action="store_false")
    p.add_argument("--subject",    default=None, help="Filter to one subject")
    p.add_argument("--seed",       type=int, default=42)
    args = p.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets huggingface_hub")
        sys.exit(1)

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("WARNING: HF_TOKEN not set — dataset may be gated.")
        print("  export HF_TOKEN=hf_...")

    print("Downloading cais/hle from HuggingFace...")
    try:
        ds = load_dataset("cais/hle", split="test", token=token, streaming=True)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    rows, seen = [], set()
    for i, row in enumerate(ds):
        if len(rows) >= args.samples * 4:
            break
        if args.text_only:
            img = row.get("image")
            if img is not None and img not in ("", b""):
                continue
        if args.subject and args.subject.lower() not in row.get("subject", "").lower():
            continue
        qid = row.get("id", f"hle_{i:05d}")
        if qid in seen:
            continue
        seen.add(qid)
        rows.append({
            "id":          qid,
            "subject":     row.get("subject", "Unknown"),
            "question":    row.get("question", ""),
            "answer":      row.get("answer", ""),
            "answer_type": row.get("answer_type", "exact"),
        })

    random.seed(args.seed)
    random.shuffle(rows)
    rows = rows[:args.samples]

    with open(args.output, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"Saved {len(rows)} questions -> {args.output}")
    print("\nSubject breakdown:")
    for subj, n in Counter(r["subject"] for r in rows).most_common():
        print(f"  {subj:<35} {n}")
    print(f"\nNext: python3 run_eval.py --questions {args.output}")


if __name__ == "__main__":
    main()
