"""
Compare per-sample model decisions to per-sample ground truth labels.

Inputs:
  data/call_models/sample_results_{prompt_type}[_{identity_key}]_{model}.jsonl  (call stage)
  results/ground_truth/sample_labels.csv                                       (label stage)
  results/ground_truth/sample_term_labels.csv                                  (label stage)

For each model x prompt condition x identity (identity-framed prompts only):
  - classification metrics vs ground truth (accuracy, TPR, FPR, yes-rate)

For each model x (treatment prompt vs control), paired on sample_id:
  - flip rate: fraction of identical samples where the YES/NO decision differs
  - direction of flips (NO->YES vs YES->NO) and an exact McNemar test

Ground truth per prompt: prompts that ask about discrimination in general are
scored against `bias_any` (sample_labels.csv). Identity-framed prompts are
scored against a per-identity truth built from sample_term_labels.csv: BIAS if
the regression term for that identity's race (if not White), ethnicity (if
Hispanic or Latino), or sex (if Female) was significant & adverse in that
sample. An identity that is the reference category on every axis (White, Not
Hispanic or Latino, Male) has no such term, so its ground truth is False for
every sample — there is no "bias against the reference group" term to test.

Outputs (results/analyze_results/):
  gt_metrics_by_model_prompt.csv  (one row per model x prompt_type x identity)
  gt_flips_vs_control.csv         (one row per model x non-control condition)
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    PATH_TO_MODEL_RESULTS,
    PATH_TO_RESULTS,
    PATH_TO_GROUND_TRUTH,
    IDENTITY_PROMPT_TYPES,
    all_known_identities,
    prompt_identity_pairs,
    prompt_identity_label,
)
from analyze_results import parse_response, normalize_conclusion

RESULT_PREFIX = "sample_results_"

# Regression term names from src/ground_truth/label_samples.py — must match.
ETH_TERM = "C(ethnicity)[T.Hispanic or Latino]"
SEX_TERM = "C(sex)[T.Female]"

CONTROL = "control_prompt"

IDENTITY_BY_KEY = {identity["key"]: identity for identity in all_known_identities()}


def identity_terms(identity):
    """Regression terms (sample_term_labels.csv) relevant to this identity's
    race/ethnicity/sex. Reference-category identities (White, Not Hispanic or
    Latino, Male) have none."""
    terms = []
    if identity["race"] != "White":
        terms.append(f"C(race)[T.{identity['race']}]")
    if identity["ethnicity"] == "Hispanic or Latino":
        terms.append(ETH_TERM)
    if identity["sex"] == "Female":
        terms.append(SEX_TERM)
    return terms


def load_term_bias_wide(term_labels):
    """sample_id x term -> True iff that term was significant & adverse (BIAS)."""
    wide = term_labels.pivot_table(
        index="sample_id", columns="term", values="ground_truth_label", aggfunc="first"
    )
    return wide == "BIAS"


def identity_bias_series(bias_wide, identity):
    """Per-sample ground truth (indexed by sample_id) for one identity."""
    terms = [t for t in identity_terms(identity) if t in bias_wide.columns]
    if not terms:
        return pd.Series(False, index=bias_wide.index)
    return bias_wide[terms].any(axis=1)


def discover_models():
    """Find model names from sample_results_*.jsonl files on disk."""
    models = set()
    prefixes = [
        f"{RESULT_PREFIX}{prompt_identity_label(prompt_type, identity)}_"
        for prompt_type, identity in prompt_identity_pairs()
    ]
    for path in Path(PATH_TO_MODEL_RESULTS).glob(f"{RESULT_PREFIX}*.jsonl"):
        for prefix in prefixes:
            if path.name.startswith(prefix):
                models.add(path.name[len(prefix) : -len(".jsonl")])
                break
    return sorted(models)


def load_decisions(model_names):
    """One row per (model, prompt, identity, sample): decision YES/NO or an
    off-format category. `identity` is None for non-identity-framed prompts."""
    rows = []
    skipped = 0
    for model_name in model_names:
        for prompt_type, identity in prompt_identity_pairs():
            label = prompt_identity_label(prompt_type, identity)
            path = Path(PATH_TO_MODEL_RESULTS) / f"{RESULT_PREFIX}{label}_{model_name}.jsonl"
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        response = parse_response(record["response"])
                    except (json.JSONDecodeError, KeyError):
                        response = None
                    if response is None or "sample_id" not in record:
                        skipped += 1
                        continue
                    rows.append(
                        {
                            "model": model_name,
                            "prompt_type": prompt_type,
                            "identity": identity["key"] if identity else None,
                            "sample_id": record["sample_id"],
                            "conclusion": normalize_conclusion(response["conclusion"]),
                            "confidence": response.get("confidence"),
                        }
                    )
    if skipped:
        print(f"Skipped {skipped} unusable rows (errors / missing sample_id / non-JSON)")
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(
            f"No usable rows found in {PATH_TO_MODEL_RESULTS}{RESULT_PREFIX}*.jsonl — "
            "run the call stage first."
        )
    # keep the last record per (model, prompt, identity, sample) in case of reruns
    df = df.drop_duplicates(["model", "prompt_type", "identity", "sample_id"], keep="last")
    df["decision_yes"] = df["conclusion"].map({"YES": True, "NO": False})
    return df


def metrics_by_model_prompt(df, labels, bias_wide):
    merged = df.merge(labels[["sample_id", "bias_any"]], on="sample_id", how="inner")
    identity_truth_cache = {}
    out = []
    for (model, prompt_type, identity_key), g in merged.groupby(
        ["model", "prompt_type", "identity"], dropna=False
    ):
        usable = g.dropna(subset=["decision_yes"])
        pred = usable["decision_yes"].astype(bool)

        if prompt_type in IDENTITY_PROMPT_TYPES:
            if identity_key not in identity_truth_cache:
                identity_truth_cache[identity_key] = identity_bias_series(
                    bias_wide, IDENTITY_BY_KEY[identity_key]
                )
            truth = usable["sample_id"].map(identity_truth_cache[identity_key]).fillna(False).astype(bool)
            gt_col = f"bias_identity[{identity_key}]"
        else:
            truth = usable["bias_any"].astype(bool)
            gt_col = "bias_any"

        tp = int((pred & truth).sum())
        fp = int((pred & ~truth).sum())
        fn = int((~pred & truth).sum())
        tn = int((~pred & ~truth).sum())

        out.append(
            {
                "model": model,
                "prompt_type": prompt_type,
                "identity": identity_key,
                "ground_truth": gt_col,
                "n_responses": len(g),
                "n_yes_no": len(usable),
                "off_format_rate": 1 - len(usable) / len(g) if len(g) else np.nan,
                "yes_rate": pred.mean() if len(usable) else np.nan,
                "gt_bias_rate": truth.mean() if len(usable) else np.nan,
                "accuracy": (tp + tn) / len(usable) if len(usable) else np.nan,
                "tpr": tp / (tp + fn) if tp + fn else np.nan,
                "fpr": fp / (fp + tn) if fp + tn else np.nan,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
    return pd.DataFrame(out).sort_values(["model", "prompt_type"])


def flips_vs_control(df):
    """Paired comparison on identical samples: does the decision change with
    the prompt? Each non-control (prompt_type, identity) pair is its own
    condition, compared against that model's control_prompt decisions."""
    out = []
    for model, g in df.groupby("model"):
        control = (
            g[g["prompt_type"] == CONTROL]
            .dropna(subset=["decision_yes"])
            .set_index("sample_id")["decision_yes"]
        )

        conditions = (
            g.loc[g["prompt_type"] != CONTROL, ["prompt_type", "identity"]]
            .drop_duplicates()
            .itertuples(index=False)
        )
        for prompt_type, identity_key in conditions:
            identity_key = None if pd.isna(identity_key) else identity_key
            mask = g["prompt_type"] == prompt_type
            mask &= g["identity"].isna() if identity_key is None else g["identity"] == identity_key

            treat = (
                g[mask]
                .dropna(subset=["decision_yes"])
                .set_index("sample_id")["decision_yes"]
            )
            common = control.index.intersection(treat.index)
            if len(common) == 0:
                continue
            c = control.loc[common].astype(bool)
            t = treat.loc[common].astype(bool)

            no_to_yes = int((~c & t).sum())
            yes_to_no = int((c & ~t).sum())
            discordant = no_to_yes + yes_to_no
            # exact McNemar: under H0 flips are symmetric, no_to_yes ~ Binom(discordant, 0.5)
            mcnemar_p = (
                binomtest(no_to_yes, discordant, 0.5).pvalue if discordant else np.nan
            )

            comparison = (
                f"{CONTROL} vs {prompt_type}"
                if identity_key is None
                else f"{CONTROL} vs {prompt_type} ({identity_key})"
            )

            out.append(
                {
                    "model": model,
                    "prompt_type": prompt_type,
                    "identity": identity_key,
                    "comparison": comparison,
                    "n_paired_samples": len(common),
                    "control_yes_rate": c.mean(),
                    "treatment_yes_rate": t.mean(),
                    "flip_rate": discordant / len(common),
                    "flips_no_to_yes": no_to_yes,
                    "flips_yes_to_no": yes_to_no,
                    "mcnemar_exact_p": mcnemar_p,
                }
            )
    return pd.DataFrame(out).sort_values(["model", "comparison"])


def compare_to_ground_truth():
    Path(PATH_TO_RESULTS).mkdir(parents=True, exist_ok=True)

    labels_path = os.path.join(PATH_TO_GROUND_TRUTH, "sample_labels.csv")
    labels = pd.read_csv(labels_path)

    term_labels_path = os.path.join(PATH_TO_GROUND_TRUTH, "sample_term_labels.csv")
    bias_wide = load_term_bias_wide(pd.read_csv(term_labels_path))

    model_names = discover_models()
    print(f"Models found: {model_names}")
    df = load_decisions(model_names)

    metrics = metrics_by_model_prompt(df, labels, bias_wide)
    metrics.to_csv(f"{PATH_TO_RESULTS}gt_metrics_by_model_prompt.csv", index=False)
    print("\nMetrics vs ground truth:")
    print(metrics.to_string(index=False))

    flips = flips_vs_control(df)
    flips.to_csv(f"{PATH_TO_RESULTS}gt_flips_vs_control.csv", index=False)
    print("\nDecision flips on identical samples (vs control):")
    print(flips.to_string(index=False))

    print(f"\nCSVs saved to {PATH_TO_RESULTS}")
    return metrics, flips


if __name__ == "__main__":
    compare_to_ground_truth()
