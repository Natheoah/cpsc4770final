# providers.py
#
# Thin wrappers around the two model APIs.
#
# GeminiProvider  — Google AI Studio via google-genai SDK
# OllamaProvider  — local Ollama server (any model, including Gemma)
#
# Both implement the same interface:
#   provider.generate(question: str) -> str   (the raw model response text)

import os
import time
import requests


SYSTEM_PROMPT = """\
You are being evaluated on Humanity's Last Exam — a benchmark of extremely \
difficult expert-level academic questions. Answer with precision.

Your response MUST use this exact format (three lines, no extra text):

Explanation: <your step-by-step reasoning>
Answer: <your final answer — letter only for multiple-choice, exact value for others>
Confidence: <0%–100%>\
"""


# ── Gemini ─────────────────────────────────────────────────────────────────────

class GeminiProvider:
    """
    Calls the Google AI Studio API via the official google-genai SDK.

    Requires:  GOOGLE_API_KEY environment variable
    Get a key: https://aistudio.google.com/
    """

    def __init__(self, model: str = "gemini-2.5-pro-preview-06-05", temperature: float = 0.0):
        import google.genai as genai
        from google.genai import types

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY is not set.\n"
                "  export GOOGLE_API_KEY=your_key_from_aistudio.google.com"
            )

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=temperature,
            max_output_tokens=8192,
        )
        self.label = f"Gemini ({model})"

    def generate(self, question: str) -> tuple[str, float]:
        """Returns (response_text, latency_seconds)."""
        import google.genai as genai

        t0 = time.perf_counter()
        response = self.client.models.generate_content(
            model=self.model,
            contents=question,
            config=self.config,
        )
        latency = time.perf_counter() - t0
        return response.text, latency


# ── Ollama (Gemma local) ───────────────────────────────────────────────────────

class OllamaProvider:
    """
    Calls a locally-running Ollama server.
    Works with any model pulled into Ollama, including all Gemma variants.

    Setup:
      1. Install Ollama:  https://ollama.com
      2. Start server:    ollama serve
      3. Pull a model:    ollama pull gemma3:27b
                          ollama pull gemma3:12b   (needs less VRAM)
                          ollama pull gemma3:4b    (CPU-friendly)

    No API key needed — runs entirely on your machine.
    """

    def __init__(
        self,
        model: str = "gemma3:27b",
        base_url: str = "http://127.0.0.1:11434",
        temperature: float = 0.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.label = f"Gemma ({model} via Ollama)"

        self._check_server()

    def _check_server(self):
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            if not any(self.model in m for m in models):
                print(f"  ⚠  Model '{self.model}' not found in Ollama.")
                print(f"     Run:  ollama pull {self.model}")
        except requests.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}.\n"
                "  Make sure Ollama is running:  ollama serve"
            )

    def generate(self, question: str) -> tuple[str, float]:
        """Returns (response_text, latency_seconds)."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ]
        payload = {
            "model":   self.model,
            "messages": messages,
            "stream":  False,
            "options": {
                "temperature": self.temperature,
                "num_predict": 8192,
            },
        }

        t0 = time.perf_counter()
        r = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=300,
        )
        latency = time.perf_counter() - t0
        r.raise_for_status()

        content = r.json()["message"]["content"]
        return content, latency
