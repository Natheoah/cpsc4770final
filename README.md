# HLE Benchmark — Gemini vs Gemma (Python)

Evaluates models on [Humanity's Last Exam](https://agi.safe.ai/) (HLE), a benchmark of
2,500 expert-level questions across math, physics, biology, CS, and humanities.

## File Map

```
hle-eval/
├── run_eval.py          ← entry point  (python3 run_eval.py)
├── providers.py         ← GeminiProvider + OllamaProvider (Gemma)
├── grader.py            ← answer extraction and scoring
├── questions.py         ← 10 built-in sample questions (fallback)
├── fetch_questions.py   ← downloads real HLE dataset from HuggingFace
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install google-genai requests
# For fetching the real dataset:
pip install datasets huggingface_hub
```

### 2. Get a Google API key

Go to [aistudio.google.com](https://aistudio.google.com), click **Get API key**, then:

```bash
export GOOGLE_API_KEY=your_key_here
```

### 3. (For Gemma) Install and start Ollama

```bash
# Install: https://ollama.com
ollama serve

ollama pull gemma3:4b    # CPU-friendly, ~4GB
ollama pull gemma3:12b   # needs ~8GB VRAM
ollama pull gemma3:27b   # best accuracy, needs ~16GB VRAM
```

### 4. (Optional) Download the real HLE dataset

The pipeline ships with 10 sample questions. For real benchmarking:

```bash
# Get a free token at huggingface.co/settings/tokens
HF_TOKEN=hf_... python3 fetch_questions.py --samples 50
# writes hle_dataset.json
```

---

## Running Evaluations

```bash
# Gemini only
python3 run_eval.py

# Local Gemma only
python3 run_eval.py --provider gemma

# Compare both side-by-side
python3 run_eval.py --provider both --samples 20

# Use real HLE dataset
python3 run_eval.py --questions hle_dataset.json --samples 50
```

### All options

```
--provider    gemini | gemma | both     (default: gemini)
--samples     N                         (default: 10)
--workers     N  parallel threads       (default: 3)
--output      results.json              (default: hle_results.json)
--questions   path/to/questions.json    (default: built-in samples)
--gemini-model  gemini-2.5-pro-...      (default: gemini-2.5-pro-preview-06-05)
--gemma-model   gemma3:27b              (default: gemma3:27b)
--ollama-url    http://127.0.0.1:11434  (default: http://127.0.0.1:11434)
```

---

## How It Works

```
run_eval.py
    │
    ├── loads questions  ←  questions.py (samples) or hle_dataset.json (real)
    │
    ├── runs questions in parallel (ThreadPoolExecutor)
    │     ├── GeminiProvider  →  Google AI Studio API  (google-genai SDK)
    │     └── OllamaProvider  →  local Ollama server   (HTTP /api/chat)
    │
    └── grades each response via grader.py
          ├── extracts "Answer: ..." line from model output
          ├── exact string match
          ├── numeric equivalence within 1%  (1.5708 matches pi/2)
          ├── LaTeX normalization            (\frac{pi}{2} matches pi/2)
          └── multiple-choice letter extraction  ((A) matches A)
```

---

## Expected Scores

From the [Scale AI HLE leaderboard](https://labs.scale.com/leaderboard/humanitys_last_exam):

| Model | HLE Accuracy |
|---|---|
| Gemini 3 Pro Preview | 37.5% |
| Claude Opus 4.6 (Thinking) | 34.4% |
| GPT-5 Pro | 31.6% |
| Gemma 3 27B | ~5–10% (estimated) |

*Official scores use all 2,500 questions. Sample runs will vary.*