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

This project currently uses multiple LLM APIs, including:
- OpenAI
- Gemini
- Anthropic

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


## How to run

### 1. Create data results directories 
From root repo

```sh
mkdir data
mkdir results
```

### 2. Populate API keys on .env
From root repo

```sh
cd src
touch .env
````
Add your own API KEYs
```.env
export OPENAI_API_KEY = ""
export GEMINI_API_KEY = ""
export ANTHROPIC_API_KEY = ""
```
### 3. Run `gather_data.py`

```sh
python gather_data.py
```

1. Retrieve HMDA loan application data from the API `load_data.py`
2. Filter the dataset to selected decision outcomes and summarize the data to `summarize_raw_data.py`
3. Create 2 files in data directory
  - `hmda_CA_2024.csv`
  - `summary.txt`

### 4. Run `call_models.py`

```sh
python call_models.py
```

1. Call LLMs with given prompt in `prompts.py` and `summary.txt`
2. Save jsonls file under data directory

 
### 5. Run `python analyze_results.py`

```sh
python analyze_results.py
```

It will save vidualizations under results directry and show statistical tests' results
