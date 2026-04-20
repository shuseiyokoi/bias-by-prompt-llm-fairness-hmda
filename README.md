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


## How to run

1. Retrieve HMDA loan application data from the API `load_data.py`
2. Filter the dataset to selected decision outcomes and summarize the data to `summarize_raw_data.py`
3. Send the same statistical evidence to multiple LLMs with different framing styles


1. Compare how conclusions change depending on prompt wording
