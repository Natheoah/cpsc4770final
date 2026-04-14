# grader.py
#
# Deterministic answer grader. Extracts the "Answer: ..." line from model
# output and scores it against ground truth.
#
# Handles:
#   - Exact string match (case-insensitive, punctuation-stripped)
#   - Numeric equivalence within 1%  (e.g. "0" vs "0.00")
#   - LaTeX normalization            (e.g. "\frac{pi}{2}" vs "pi/2")
#   - Multiple-choice letter extraction  (e.g. "(A)" -> "A")

import math
import re


# ── Answer extraction ──────────────────────────────────────────────────────────

def extract_answer(output: str) -> str | None:
    """Pull the text after 'Answer:' from a structured model response."""
    match = re.search(r"^Answer:\s*(.+?)(?:\n|$)", output, re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else None


# ── Normalization helpers ──────────────────────────────────────────────────────

def normalize(s: str) -> str:
    return (
        s.strip()
         .lower()
         .rstrip(".,;:!?")
         .strip()
         .replace("  ", " ")
    )


def normalize_latex(s: str) -> str:
    s = normalize(s)
    s = re.sub(r"\s*", "", s)                          # remove all spaces
    s = re.sub(r"\\frac\{(.+?)\}\{(.+?)\}", r"(\1)/(\2)", s)  # \frac{a}{b} -> (a)/(b)
    s = s.replace("\\pi", "pi").replace("π", "pi")
    return s


def extract_mc_letter(s: str) -> str | None:
    """Return the first A-E letter found, ignoring parentheses."""
    m = re.search(r"\b([A-Ea-e])\b", s)
    return m.group(1).upper() if m else None


# ── Scoring ────────────────────────────────────────────────────────────────────

def grade(extracted: str, correct: str, answer_type: str) -> dict:
    """
    Returns dict with keys:
      pass   (bool)
      score  (float 0.0-1.0)
      note   (str, reason for pass/fail)
    """
    if answer_type == "multiple_choice":
        got = extract_mc_letter(extracted)
        exp = extract_mc_letter(correct)
        passed = got is not None and got == exp
        return {
            "pass": passed,
            "score": 1.0 if passed else 0.0,
            "note": f"extracted letter '{got}' vs expected '{exp}'",
        }

    # --- Exact / numeric / LaTeX matching ---

    if normalize(extracted) == normalize(correct):
        return {"pass": True, "score": 1.0, "note": "exact match"}

    # Numeric equivalence — substitute known constants then compare as floats
    _SUBS = [("pi", str(math.pi)), ("e", str(math.e))]

    def to_float(s: str) -> float | None:
        s = normalize(s)
        for sym, val in _SUBS:
            s = s.replace(sym, val)
        # Handle simple a/b fractions
        if "/" in s:
            parts = s.split("/", 1)
            try:
                return float(parts[0]) / float(parts[1])
            except (ValueError, ZeroDivisionError):
                pass
        try:
            return float(s)
        except ValueError:
            return None

    a, b = to_float(extracted), to_float(correct)
    if a is not None and b is not None:
        rel_err = abs(a - b) / (abs(b) + 1e-10)
        if rel_err < 0.01:
            return {"pass": True, "score": 1.0, "note": f"numeric match (rel err {rel_err:.2e})"}


    # LaTeX / symbol normalization
    if normalize_latex(extracted) == normalize_latex(correct):
        return {"pass": True, "score": 1.0, "note": "LaTeX-normalized match"}

    return {"pass": False, "score": 0.0, "note": f"got '{extracted}' expected '{correct}'"}


def score_response(output: str, correct_answer: str, answer_type: str) -> dict:
    """
    Top-level function called per question.
    Returns full result dict including the extracted answer.
    """
    extracted = extract_answer(output)

    if extracted is None:
        return {
            "pass": False,
            "score": 0.0,
            "extracted": None,
            "note": "No 'Answer:' line found in model output",
        }

    result = grade(extracted, correct_answer, answer_type)
    result["extracted"] = extracted
    return result
