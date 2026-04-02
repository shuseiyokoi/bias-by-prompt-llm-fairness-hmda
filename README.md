# Bias by Prompt LLM Fairness

A short one- to two-sentence description of what this project does and why it matters.

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

1. Retrieve HMDA loan application data from the API
2. Filter the dataset to selected decision outcomes
3. Summarize the raw data by sensitive attributes and key financial variables
4. Convert summarized results into prompt-ready text
5. Send the same statistical evidence to multiple LLMs with different framing styles
6. Compare how conclusions change depending on prompt wording

## Prompt Framing Conditions

Examples of framing styles tested in this project include:

- Neutral statistical review
- Emotionally concerned framing
- Identity- or fairness-focused framing
- Strong suspicion of unfair treatment

The goal is to measure whether framing changes:
- Whether disparity is detected
- Whether bias is attributed
- Confidence level
- Final conclusion wording

## Models Used

This project currently uses multiple LLM APIs, including:

- OpenAI
- Gemini
- Anthropic
- Qwen
