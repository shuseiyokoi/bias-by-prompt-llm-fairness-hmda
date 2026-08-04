## Extended Research

This project started as a class project and is now being extended into a research project on LLM prompt sensitivity and fairness analysis.

The original class project is kept on the `main` branch.  
Ongoing research experiments are developed in the `research-extension` branch.


# Bias by Prompt LLM Fairness

The goal of this project is to investigate how a given persona or role affects LLM-based data analysis. [Martin Bertran (2026)](https://arxiv.org/pdf/2602.18710) pointed out that LLM-powered agentic data scientists can reach different conclusions depending on the prompt framing or assigned persona, even when they use the same dataset and analytical methods. Based on this study, I wonder whether a model generates different outputs when it is given a sensitive background context and is encouraged to respond with empathy.

## Introduction

This project investigates how sensitive framing affects large language model conclusions in data analysis. In particular, it examines whether emotionally framed statements about bias influence how LLMs interpret loan approval outcomes.

### Research Question

"How does emotional or bias-related framing in prompts affect LLM conclusions when analyzing loan approval data?"

## Data Source

This project uses publicly available HMDA loan application data.

- **Dataset:** 2024 California HMDA loan application data
- **Source:** FFIEC HMDA Data Browser API
- **Outcomes analyzed:**
  - Loan originated
  - Application approved but not accepted
  - Application denied

### Models Used

**Cloud APIs**
- OpenAI (GPT)
- Google (Gemini)
- Anthropic (Claude)

**Local open-weight models** (served with llama.cpp, OpenAI-compatible API)
- Qwen2.5-7B-Instruct
- Qwen3-8B
- Llama-3.1-8B-Instruct
- Llama-3.2-3B-Instruct
- Gemma-2-9B-it
- Gemma-3-12B-it

Which models and prompt types run is controlled in [`src/config.py`](src/config.py) (`GPT_MODELS`, `CLAUDE_MODELS`, `GEMINI_MODELS`, `QWEN_MODELS`, `LLAMA_MODELS`, `GEMMA_MODELS`, `PROMPT_TYPES`, `NUM_ITERATIONS`).

## Analysis

**Chi-Square & Fisher Test**
For categorical variable testing 2 x 2 table for example

  
||NO|YES|
|---|---|---|
|Control|236|66|
|Emotional|300|0|

## Summary of the results 


- Haiku 4.5 : Changed answer 180 degree with emotional input
- Gemini 2.5: Guardrail triggered by “bias” ? → More safer Output
- GPT 3.5: Same output, but lower confidence level

### Percentages by Model-Prompt (n=300 each)
<img width="1200" height="600" title = "Percentages by Model-Prompt (n=300 each)" alt="percentages-by-model-and-prompt" src="https://github.com/user-attachments/assets/9e55ce10-3441-42a5-a541-af4a79930a69" />

### Confidence by Model-Prompt (n=300 each)
<img width="1200" height="600" title = "Confidence by Model-Prompt (n=300 each)" alt="confidence-by-model-and-prompt-small" src="https://github.com/user-attachments/assets/acefc6bb-ca47-4624-8766-8cff8ec86c24" />

### Chi-Square & Fisher Test

<img width="1016" height="337" alt="image (12)" src="https://github.com/user-attachments/assets/77876b4e-c171-4c4b-af4d-db0ccec44fec" />


## Project layout

The pipeline is split into 5 stages. Each stage lives in its own folder under
`src/` and writes its outputs to a designated folder under `data/` or `results/`:

| Stage | Code | Outputs |
|---|---|---|
| 1. Gather data | `src/gather_data/` | `data/gather_data/` |
| 2. Local model server | `src/local_qwen/` | (GGUF weights only) |
| 3. Call models | `src/call_models/` | `data/call_models/` |
| 4. Analyze results | `src/analyze_results/` | `results/analyze_results/` |
| 5. Ground truth regression | `src/ground_truth/` | `results/ground_truth/` |

All paths are defined in `src/config.py` and anchored to the repo root, so every
script can be run from any directory.

## Sampling design (per-sample ground truth)

Instead of sending the same full-dataset summary `NUM_ITERATIONS` times, the
sampling design draws `N_SAMPLES` distinct datasets of `SAMPLE_SIZE` rows each
(`src/config.py`), gives every sample its own ground-truth bias label, and runs
each model x prompt condition once per sample. Because every prompt condition
sees exactly the same N samples, decision flips caused purely by prompt framing
can be measured pairwise (McNemar test), and model conclusions can be scored
against ground truth (accuracy / TPR / FPR).

```sh
cd src
python main.py --sample-data      # data/gather_data/samples/sample_*.{csv,txt} + manifest
python main.py --label-samples    # results/ground_truth/sample_labels.csv
# run the call scripts in src/call_models/ (outputs: data/call_models/sample_results_*.jsonl)
python main.py --compare          # results/analyze_results/gt_metrics_*.csv, gt_flips_*.csv
```

Ground truth per sample is automated: the same logistic regression used on the
full dataset (`denied ~ race + sex + ethnicity + financial controls`) is fit on
the sample's raw rows; a sensitive term that is significant and adverse marks
the sample `BIAS`. General prompts are scored against `bias_any`; the
female-Latino identity prompts against `bias_latino_female`. `SAMPLE_SIZE=2000`
was calibrated (`results/ground_truth/calibration_label_rates.csv`) so that
`bias_any` is true for ~46% of samples — near-maximal label variance — while the
per-sample summary (~4k tokens) still fits the 8k context of the local models.

### Prompt data mode: summary vs. raw rows

`USE_SUMMARY` in `src/config.py` controls what each sample embeds under the
`Data:` section of the prompt:

- `USE_SUMMARY = True` — the per-sample aggregate stats table
  (`sample_{id}_summary.txt`, grouped by race/sex/ethnicity), a few KB.
- `USE_SUMMARY = False` (**current default**) — the sample's raw
  `sample_{id}.csv` rows (all `SAMPLE_SIZE` of them) embedded verbatim.

`sample_datasets.py` only writes `_summary.txt` files when `USE_SUMMARY` is
`True`; the raw CSV is always written either way. Note that raw mode sends a
much larger prompt than the calibrated ~4k-token summary above, so it may not
fit the 8k context window of the local models — check `USE_SUMMARY` before
running `call_qwen.py` or other local-model calls.

## How to run

### 1. Set up a virtual environment
From root repo

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Populate API keys in .env
Only needed for the cloud models (skip for local Qwen).

```sh
cd src
touch .env
```
Add your own API keys
```.env
export OPENAI_API_KEY = ""
export GEMINI_API_KEY = ""
export ANTHROPIC_API_KEY = ""
```

### 3. Gather data from HMDA and summarize

```sh
cd src/gather_data
python gather_data.py
```

1. Retrieve HMDA loan application data from the API `load_data.py`
2. Filter the dataset to selected decision outcomes `preprocess_data.py`
3. Summarize the preprocessed data by race, sex, and ethnicity `summarize_data.py`
4. Create 3 files in `data/gather_data/`
  - `hmda_CA_2024.csv`
  - `preprocessed_data.csv`
  - `summary.txt`

### 4. Call Large Language Models 

> [!WARNING]
> Skip, if you have data already. Set `NUM_ITERATIONS` in `src/config.py` before running — each iteration is one API call per model and prompt type.

**Cloud models** (from `src/call_models/`)

```sh
cd src/call_models
python call_chatGPT.py
python call_claude.py
python call_gemini.py
```

**Local Qwen** (no API key needed)

Terminal 1 — start the llama.cpp server (expects the GGUF model path set in `src/local_qwen/Makefile`):

```sh
cd src/local_qwen
make serve
```

Terminal 2 — run the calls against the local server:

```sh
cd src/call_models
python call_qwen.py
```

Each script loops over its models in `config.py` and all `PROMPT_TYPES`, and appends one JSON line per run to `data/call_models/results_{prompt_type}_{model}.jsonl`.

### 5. Analyze Results

```sh
cd src/analyze_results
python analyze_results.py
```

Or interactively with the notebook `src/analyze_results/analyze_results.ipynb`.

Models are auto-discovered from the `results_*.jsonl` files in `data/call_models/`, so this works regardless of which models are active in `config.py`. It prints a coverage table (usable responses per model and prompt — rows that came back as API errors or non-JSON are skipped and counted), then saves to `results/analyze_results/`:

- `conclusion-count-by-prompt-{model}.png`
- `percentages-by-prompt-{model}.png` (with per-bar sample sizes)
- `confidence-by-prompt-{model}.png`
- `confidence-by-model-and-conclusion.png`
- `tableresults.csv` — conclusion counts per model and prompt
- `stats_vs_control.csv` — chi-square / Fisher exact tests of each prompt type against the control prompt, per model

### 6. Ground truth regression

```sh
cd src/ground_truth
python run_regression.py
```

Fits a logistic regression (`denied ~ race + sex + ethnicity + financial controls`)
on the loan-level data from `data/gather_data/preprocessed_data.csv` and saves to
`results/ground_truth/`:

- `ground_truth_labels.csv` — BIAS / FAVORED / NO_BIAS label per sensitive group
- `full_regression_coefficients.csv`
- `regression_summary.txt`

### 7. Statistical bias tests

```sh
cd src/ground_truth
python run_statistical_tests.py
```

Tests whether the application decision depends on race, sex, and ethnicity in the
raw loan-level data. Runs marginal chi-square tests of independence (decision vs
each attribute, no controls) plus the same controlled logistic regression as
`run_regression.py` (reused via import), and saves to `results/ground_truth/`:

- `chi_square_tests.csv` — chi-square statistic, dof, and p-value per attribute
- `bias_test_labels.csv` — BIAS / FAVORED / NO_BIAS label per sensitive group
- `statistical_tests_summary.txt` — full report: chi-square table, regression summary, and labels

Key result: race and ethnicity are strongly associated with denial both marginally
and after financial controls (Black or African American OR ≈ 1.71, Hispanic or
Latino OR ≈ 1.15 vs the White / Male / Not Hispanic reference). Sex shows no
marginal association, but with financial controls female applicants have slightly
*lower* denial odds (OR ≈ 0.90).

> [!NOTE]
> `src/main.py` is a thin dispatcher over the same stages: `python main.py --gather-data | --call-models | --analyze` (cloud models only; local models still run via `call_qwen.py`).
