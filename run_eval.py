#!/usr/bin/env python3
"""
run_eval.py — HLE Benchmark Evaluation Pipeline
================================================

QUICKSTART
----------
  # Gemini only (set your API key first):
  export GOOGLE_API_KEY=your_key_from_aistudio.google.com
  python3 run_eval.py

  # Local Gemma only (needs Ollama running — see README.md):
  python3 run_eval.py --provider gemma

  # Compare both side-by-side:
  python3 run_eval.py --provider both --samples 20

  # Use the real HLE dataset (after running fetch_questions.py):
  python3 run_eval.py --questions hle_dataset.json --samples 50

ALL OPTIONS
-----------
  --provider   gemini | gemma | both          (default: gemini)
  --samples    N                              (default: 10)
  --workers    N  parallel threads            (default: 3)
  --output     results.json                   (default: hle_results.json)
  --questions  path/to/questions.json         (default: built-in samples)
  --gemini-model  gemini-2.5-pro-...          (default: gemini-2.5-pro-preview-06-05)
  --gemma-model   gemma3:27b                  (default: gemma3:27b)
  --ollama-url    http://127.0.0.1:11434      (default: http://127.0.0.1:11434)
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from cpsc4770final.grader import score_response
from cpsc4770final.providers import GeminiProvider, OllamaProvider
from cpsc4770final.questions import SAMPLE_QUESTIONS


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="HLE Benchmark — Gemini vs Gemma")
    p.add_argument("--provider",      default="gemini",
                   choices=["gemini", "gemma", "both"])
    p.add_argument("--samples",       type=int, default=10)
    p.add_argument("--workers",       type=int, default=3,
                   help="Parallel threads per provider")
    p.add_argument("--output",        default="hle_results.json")
    p.add_argument("--questions",     default=None,
                   help="Path to hle_dataset.json from fetch_questions.py")
    p.add_argument("--gemini-model",  default="gemini-2.5-pro-preview-06-05")
    p.add_argument("--gemma-model",   default="gemma3:27b")
    p.add_argument("--ollama-url",    default="http://127.0.0.1:11434")
    return p.parse_args()


# ── Load questions ─────────────────────────────────────────────────────────────

def load_questions(path: str | None, n: int) -> list[dict]:
    if path:
        raw = json.loads(open(path).read())
        # Support both plain list and promptfoo test-case format
        questions = [r.get("vars", r) for r in raw]
        print(f"  Loaded {len(questions)} questions from {path}")
    else:
        questions = SAMPLE_QUESTIONS
        print("  Using built-in sample questions.")
        print("  (Run fetch_questions.py for the real 2,500-question dataset.)")
    return questions[:n]


# ── Run one question against one provider ──────────────────────────────────────

def run_one(provider, question: dict) -> dict:
    """Called in a thread. Returns a result dict for one question."""
    try:
        output, latency = provider.generate(question["question"])
        result = score_response(output, question["answer"], question.get("answer_type", "exact"))
        return {
            "id":        question["id"],
            "subject":   question["subject"],
            "question":  question["question"],
            "answer":    question["answer"],
            "output":    output,
            "extracted": result["extracted"],
            "pass":      result["pass"],
            "score":     result["score"],
            "note":      result["note"],
            "latency":   round(latency, 2),
            "error":     None,
        }
    except Exception as e:
        return {
            "id":        question["id"],
            "subject":   question["subject"],
            "question":  question["question"],
            "answer":    question["answer"],
            "output":    None,
            "extracted": None,
            "pass":      False,
            "score":     0.0,
            "note":      f"ERROR: {e}",
            "latency":   0.0,
            "error":     str(e),
        }


# ── Run all questions against one provider ─────────────────────────────────────

def run_provider(provider, questions: list[dict], workers: int) -> list[dict]:
    results = [None] * len(questions)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_one, provider, q): i
            for i, q in enumerate(questions)
        }
        done = 0
        for future in as_completed(futures):
            i = futures[future]
            results[i] = future.result()
            done += 1
            r = results[i]
            icon = "✓" if r["pass"] else "✗"
            print(
                f"  [{done:2d}/{len(questions)}] {icon} "
                f"{r['id']:<10} {r['subject']:<22} "
                f"{r['latency']:.1f}s  {r['note']}"
            )

    return results


# ── Summarise results ──────────────────────────────────────────────────────────

def summarise(results: list[dict]) -> dict:
    passed  = sum(1 for r in results if r["pass"])
    total   = len(results)
    lats    = [r["latency"] for r in results if r["latency"]]

    by_subject = defaultdict(lambda: {"passed": 0, "total": 0})
    for r in results:
        s = r["subject"]
        by_subject[s]["total"] += 1
        if r["pass"]:
            by_subject[s]["passed"] += 1

    return {
        "accuracy":       f"{100 * passed / total:.1f}%" if total else "0.0%",
        "passed":         passed,
        "total":          total,
        "avg_latency_s":  round(sum(lats) / len(lats), 2) if lats else None,
        "by_subject":     {
            s: f"{v['passed']}/{v['total']} ({100*v['passed']//v['total']}%)"
            for s, v in sorted(by_subject.items())
        },
        "questions":      results,
    }


# ── Print results table ────────────────────────────────────────────────────────

def print_report(report: dict):
    LINE = "═" * 68
    THIN = "─" * 68

    print(f"\n{LINE}")
    print(f"  HLE RESULTS  ·  {report['timestamp']}")
    print(f"  {report['num_questions']} questions evaluated")
    print(LINE)

    for label, data in report["providers"].items():
        print(f"\n  ▶ {label}")
        print(f"    Accuracy : {data['accuracy']}  ({data['passed']}/{data['total']})")
        if data["avg_latency_s"]:
            print(f"    Latency  : {data['avg_latency_s']}s avg")

        print(f"\n    By subject:")
        for subject, score in data["by_subject"].items():
            print(f"      {subject:<28} {score}")

        print(f"\n    {'ID':<10} {'Subject':<22} {'✓/✗':<5} {'Extracted':<20} Expected")
        print(f"    {THIN[:65]}")
        for q in data["questions"]:
            icon = "✓" if q["pass"] else "✗"
            extracted = (q["extracted"] or "(none)")[:18]
            print(
                f"    {q['id']:<10} {q['subject']:<22} {icon:<5} "
                f"{extracted:<20} {q['answer']}"
            )

    if len(report["providers"]) > 1:
        print(f"\n{LINE}")
        print("  COMPARISON")
        for label, data in sorted(
            report["providers"].items(), key=lambda x: -x[1]["passed"]
        ):
            print(f"    {label:<42} {data['accuracy']}")

    print(f"\n{LINE}")
    print(f"  Saved → {report['output_file']}")
    print(LINE + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("\n🔬 HLE Evaluation Pipeline (Python)")
    print(f"   provider     : {args.provider}")
    print(f"   gemini model : {args.gemini_model}")
    print(f"   gemma model  : {args.gemma_model}")
    print(f"   samples      : {args.samples}")
    print(f"   workers      : {args.workers}")

    # Load questions
    print()
    questions = load_questions(args.questions, args.samples)
    print(f"  Running {len(questions)} questions\n")

    # Build provider list
    providers = []
    if args.provider in ("gemini", "both"):
        try:
            providers.append(GeminiProvider(model=args.gemini_model))
        except EnvironmentError as e:
            print(f"❌ {e}")
            sys.exit(1)

    if args.provider in ("gemma", "both"):
        try:
            providers.append(OllamaProvider(model=args.gemma_model, base_url=args.ollama_url))
        except ConnectionError as e:
            print(f"❌ {e}")
            sys.exit(1)

    # Run evaluations
    provider_results = {}
    for provider in providers:
        print(f"── {provider.label} {'─' * (50 - len(provider.label))}")
        results = run_provider(provider, questions, workers=args.workers)
        provider_results[provider.label] = summarise(results)
        print()

    # Build and save report
    report = {
        "timestamp":     datetime.now().isoformat(timespec="seconds"),
        "benchmark":     "HLE (Humanity's Last Exam)",
        "num_questions": len(questions),
        "output_file":   args.output,
        "providers":     provider_results,
    }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print_report(report)


if __name__ == "__main__":
    main()
