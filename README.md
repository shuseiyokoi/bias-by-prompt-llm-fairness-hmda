# Bias by Prompt LLM Fairness

The goal of this project is to investigate how a given persona or role affects LLM-based data analysis. [Martin Bertran (2026)](https://arxiv.org/pdf/2602.18710) pointed out that LLM-powered agentic data scientists can reach different conclusions depending on the prompt framing or assigned persona, even when they use the same dataset and analytical methods. Based on this study, I wonder whether a model generates different outputs when it is given a sensitive background context and is encouraged to respond with empathy.

## Overview

This project investigates how sensitive framing affects large language model conclusions in data analysis. In particular, it examines whether emotionally framed statements about bias influence how LLMs interpret loan approval outcomes.

## Research Question

How does emotional or bias-related framing in prompts affect LLM conclusions when analyzing loan approval data?

## Motivation

Large language models are increasingly used for summarization, reasoning, and decision support. However, their conclusions may shift depending on how a prompt is written. This project studies whether framing a prompt with emotional language or implied unfairness changes how models interpret the same statistical evidence.

## Project Scope

Rather than sending raw loan-level data directly to LLMs, this project uses pre-analyzed statistical summaries to reduce token usage and improve response speed.

Initial testing showed that sending a 1,000-row CSV required about 53,000 tokens and 5 to 6 minutes per request. After switching to compact summary tables, the input size dropped to about 1,500 tokens and response time decreased to about 5 to 6 seconds.

This change allows the project to focus more clearly on framing sensitivity rather than raw data interpretation.

## Data Source

This project uses publicly available HMDA loan application data.

- **Dataset:** 2024 California HMDA loan application data
- **Source:** FFIEC HMDA Data Browser API
- **Outcomes analyzed:**
  - Loan originated
  - Application approved but not accepted
  - Application denied

## Methodology

1. Retrieve HMDA loan application data from the API `load_data.py`
2. Filter the dataset to selected decision outcomes and summarize the data by key financial variables `summarize_raw_data.py`
5. Send the same statistical evidence to multiple LLMs with different framing styles 
6. Compare how conclusions change depending on prompt wording

## Models Used

This project currently uses multiple LLM APIs, including:

- OpenAI
- Gemini
- Anthropic
- Qwen
