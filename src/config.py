import os

# Anchor all paths to the repo root so scripts work from any directory
# (src/, a stage folder like src/call_models/, or the repo root).
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)

# One designated output folder per pipeline stage:
# 1. src/gather_data/       -> data/gather_data/
# 2. src/local_qwen/        -> (model weights only, no pipeline outputs)
# 3. src/call_models/       -> data/call_models/
# 4. src/analyze_results/   -> results/analyze_results/
# 5. src/ground_truth/      -> results/ground_truth/
PATH_TO_DATA = os.path.join(ROOT_DIR, "data", "gather_data") + os.sep
PATH_TO_SAMPLES = os.path.join(ROOT_DIR, "data", "gather_data", "samples") + os.sep
PATH_TO_MODEL_RESULTS = os.path.join(ROOT_DIR, "data", "call_models") + os.sep
PATH_TO_RESULTS = os.path.join(ROOT_DIR, "results", "analyze_results") + os.sep
PATH_TO_GROUND_TRUTH = os.path.join(ROOT_DIR, "results", "ground_truth") + os.sep

# llama.cpp server + GGUF weights for the local models
LOCAL_QWEN_DIR = os.path.join(SRC_DIR, "local_qwen")

NUM_ITERATIONS = 300  # legacy: repeated calls on the one full-dataset summary

# --- Sampling design ---
# N distinct datasets are drawn from the cleaned loan-level data. Each sample
# gets its own summary table (what the model sees) and its own ground-truth
# bias label (logistic regression on the sample's raw rows). Every model x
# prompt condition is run once per sample, so flips on EXACTLY the same data
# can be measured pairwise against the control prompt.
N_SAMPLES = 500  # samples per model/prompt setup; scale down via cost estimate
SAMPLE_SIZE = 2000  # rows (X) per sample; see results/ground_truth/calibration
SAMPLE_SEED = 42  # base RNG seed; sample i uses SAMPLE_SEED + i

USE_SUMMARY = False  # True: embed per-sample summary.txt in prompts.
                      # False: embed the sample's raw CSV rows instead.

GPT_MODELS = ["gpt-5-nano", "gpt-4o-mini"]  # gpt-3.5-turbo swapped for gpt-5-nano: its 16K context can't fit raw-mode prompts (~82K tokens/call)

CLAUDE_MODELS = [
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5-20250929",
]  # claude-sonnet-4-20250514 gives non json responses

GEMINI_MODELS = ["gemini-2.5-flash-lite", "gemini-2.5-pro"]

QWEN_MODELS = [
#    "qwen2.5-7b-instruct",
    "qwen3-8b",
]

LLAMA_MODELS = [
    "llama-3.1-8b-instruct",
    "llama-3.2-3b-instruct",
]

GEMMA_MODELS = [
    "gemma-2-9b-it",
    "gemma-3-12b-it",
]

# All models served locally with llama.cpp (see call_qwen.py / local_qwen)
LOCAL_MODELS = QWEN_MODELS + LLAMA_MODELS + GEMMA_MODELS

# prompt_jobs_config.py

PROMPT_TYPES = [
    "control_prompt",
    "emotional_identity_prompt",
    "emotional_prompt",
    "emotional_extreme_prompt",
    "emotional_suicidal_prompt",
    "identity_prompt",
    "identity_hypothetical_prompt",
]
