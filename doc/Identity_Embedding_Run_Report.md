# Identity Embedding Run: Status & Cost Report

**Prepared:** 2026-08-31 · **Scope:** `src/config.py` as committed — `N_SAMPLES=3` x `SAMPLE_SIZE=2000` rows, 16-identity matrix (race x ethnicity x sex x age), raw-CSV mode. Results archived in `data/call_models/0831_identity_embedding_test/`.

**Bottom line.** All 6 cloud models have completed the identity embedding run — 155–156 of 156 calls each, **935/936 calls, 99.9%**. Estimated spend across the 6 cloud models is **~$72**, dominated by `claude-sonnet-4-5` (54%) and `gemini-3.1-pro-preview` (23%), the same cost profile as the discovery-phase projection. The 4 local models have **not** been run yet; projected serial runtime is **3–7 hours per model** using discovery-phase latencies. One `gemini-3.1-pro-preview` call is still missing after a transient `503` (not a config/model problem — see §3).

## 1. Run design

The identity matrix crosses `IDENTITY_RACES` (White, Black or African American) x `IDENTITY_ETHNICITIES` (Hispanic, Not Hispanic) x `IDENTITY_SEXES` (Female, Male) x `IDENTITY_AGES` (35, 55) = **16 identities**. Three of the seven `PROMPT_TYPES` (`identity_prompt`, `identity_hypothetical_prompt`, `emotional_identity_prompt`) are run once per identity; the other four (`control_prompt`, `emotional_prompt`, `emotional_extreme_prompt`, `emotional_suicidal_prompt`) run once, unconditioned. That's **52 (prompt_type, identity) jobs x `N_SAMPLES`=3 = 156 calls per model**.

## 2. Completion status (156 calls/model expected)

| Model | Calls done | Missing | Status |
|---|---:|---:|---|
| `gpt-5-nano` | 156/156 | 0 | Complete |
| `gpt-4o-mini` | 156/156 | 0 | Complete |
| `claude-haiku-4-5` | 156/156 | 0 | Complete |
| `claude-sonnet-4-5` | 156/156 | 0 | Complete |
| `gemini-3.5-flash-lite` | 156/156 | 0 | Complete |
| `gemini-3.1-pro-preview` | 155/156 | 1 | 1 sample failed on transient `503 UNAVAILABLE` |
| `qwen2.5-7b-instruct` | 0/156 | 156 | Not started |
| `llama-3.1-8b-instruct` | 0/156 | 156 | Not started |
| `llama-3.2-3b-instruct` | 0/156 | 156 | Not started |
| `gemma-3-12b-it` | 0/156 | 156 | Not started |

No error entries currently sit in any output file — all API/quota/deprecated-model failures encountered mid-run (`gemini-2.5-*` 404s, one 429 quota, one 503) were stripped as they occurred, so file contents reflect only successful parses.

## 3. Cloud cost — 156 calls/model, ~82,400 in / 200 out tokens per call

*Same token/call estimate as the discovery-phase benchmark (`Discovery_Benchmark_and_Cost_Report.md`) — `SAMPLE_SIZE`/`USE_SUMMARY` are unchanged, so the ~82,401-token raw-CSV prompt still applies. Gemini pricing is carried over from the retired `gemini-2.5-*` tier as a placeholder — **re-verify `gemini-3.5-flash-lite` / `gemini-3.1-pro-preview` pricing before treating this as final.***

| Model | $/1M in · out | $/call | **Run cost (156 calls)** |
|---|---:|---:|---:|
| `claude-sonnet-4-5` | $3.00 · $15.00 | $0.2502 | $39.03 |
| `gemini-3.1-pro-preview`* | $1.25 · $10.00 | $0.1050 | $16.38 |
| `claude-haiku-4-5` | $1.00 · $5.00 | $0.0834 | $13.01 |
| `gpt-4o-mini` | $0.15 · $0.60 | $0.0125 | $1.95 |
| `gemini-3.5-flash-lite`* | $0.10 · $0.40 | $0.0083 | $1.30 |
| `gpt-5-nano` | $0.05 · $0.40 | $0.0042 | $0.66 |
| **Total (cloud, complete)** | | | **~$72.32** |

## 4. Local models — not yet run

No local-model output exists for this identity run. Projected serial runtime at discovery-phase per-call latency:

| Model | Est. runtime (156 calls) |
|---|---:|
| `qwen2.5-7b-instruct` | 6.4 h |
| `gemma-3-12b-it` | 5.0 h |
| `llama-3.1-8b-instruct` | 3.4 h |
| `llama-3.2-3b-instruct` | 3.1 h |

**Carry-forward risk:** the discovery benchmark found `llama-3.2-3b-instruct` at 0/10 valid JSON and `llama-3.1-8b-instruct` at 5/10 — unresolved. Running either as-is on this 156-call identity set will likely reproduce the same failure rate; fix formatting/add a repair step before running, or drop the model.

## 5. Data-quality notes

- **38 empty leftover files** remain in `0831_identity_embedding_test/` under the retired `gemini-2.5-flash-lite` / `gemini-2.5-pro` names (all 0 bytes — their only content was the 404 errors already stripped). Safe to delete; superseded by the `gemini-3.5-flash-lite` / `gemini-3.1-pro-preview` files, which hold the real data.
- **1 missing sample** (`identity_hypothetical_prompt`, `black_nonhispanic_female_age35`, `gemini-3.1-pro-preview`, `sample_0001`) — dropped for a transient `503`, not re-run yet.

## 6. Simulation: `SAMPLE_SIZE=300` (vs. current 2,000)

*Run isolated in scratch — 3 samples (matching current `N_SAMPLES=3`), same seeds as production (`SAMPLE_SEED=42`, seeds 42/43/44), drawn from the same 53,202 eligible rows, labeled with the same regression as `label_samples.py`. No production data was modified for this test.*

**Label balance.** All 3 samples came back `NO_BIAS` (`bias_any` 0/3), and one of the three failed to converge on the logistic regression, falling back to marginal Fisher tests — consistent with the existing calibration curve (`calibration_label_rates.csv`), which shows label positivity dropping sharply below 1,000 rows:

| | X=300 (n=3, this sim) | X=1000 (n=150, calib.) | X=2000 (n=150, current) |
|---|---:|---:|---:|
| `bias_any` | **0/3 (0%)** | 37.3% | 56.0% |
| `bias_latino_female` | 0/3 (0%) | 7.3% | 12.7% |

This is a 3-draw result, not a stable rate estimate — but at `N_SAMPLES=3`, X=300 risks **zero label variance**, which breaks the pairwise-flip / accuracy analysis the pipeline depends on.

**Cost.** Full prompt re-tokenized with the actual `get_prompt()` template (`o200k_base`): **~12,465 tokens/call** at X=300 vs. ~82,401 at X=2000 — near-linear scaling, fixed template overhead is only 101 tokens.

| Scope | X=2000 cost | X=300 cost |
|---|---:|---:|
| 156 calls/model (this run's scale) | ~$72.32 | **~$11.77** |
| 3,500 calls/model (discovery-doc full-study scale) | ~$1,622.62 | **~$264.11** |

**Verdict:** X=300 is ~6.6x cheaper but, at only 3 samples, is likely too small to produce usable ground-truth label variance. Cutting cost by raising `N_SAMPLES` at a mid-range `SAMPLE_SIZE` (e.g. keeping 2,000) is probably a better lever than shrinking `SAMPLE_SIZE` down to 300.
