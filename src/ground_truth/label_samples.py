"""
Automated ground-truth bias labels, one per sample.

For every sample CSV in data/gather_data/samples/, fit the same logistic
regression used for the full-dataset ground truth
(denied ~ race + sex + ethnicity + dti + ltv + income + loan_amount
 + property_value) on that sample's raw rows and derive:

  bias_any            any sensitive term significant & adverse (p < alpha, coef > 0)
  bias_latino_female  the Hispanic-or-Latino ethnicity term OR the Female sex
                      term is significant & adverse (matches the identity prompts)

This is independent of any model output — it is the label model conclusions
are compared against. Rare race categories are collapsed into one bucket
before fitting to avoid perfect separation; samples where the fit still fails
fall back to marginal Fisher exact tests (flagged in the `method` column).

Outputs:
  results/ground_truth/sample_labels.csv       one row per sample
  results/ground_truth/sample_term_labels.csv  one row per sample x term
"""

import argparse
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import PATH_TO_SAMPLES, PATH_TO_GROUND_TRUTH
from ground_truth.run_regression import (
    load_and_clean,
    run_regression,
    extract_ground_truth_labels,
)

ETH_TERM = "C(ethnicity)[T.Hispanic or Latino]"
SEX_TERM = "C(sex)[T.Female]"

MIN_CATEGORY_ROWS = 25  # race categories smaller than this are pooled


def collapse_rare_races(df, min_rows=MIN_CATEGORY_ROWS):
    counts = df["race"].value_counts()
    rare = counts[counts < min_rows].index
    if len(rare) > 0:
        df = df.copy()
        df["race"] = df["race"].where(
            ~df["race"].isin(rare), "Other or multiple minority races"
        )
    return df


def fisher_term(df, mask_a, mask_b, term):
    """2x2 denied-vs-approved Fisher test between two applicant groups."""
    a_denied = int(df.loc[mask_a, "denied"].sum())
    a_ok = int(mask_a.sum()) - a_denied
    b_denied = int(df.loc[mask_b, "denied"].sum())
    b_ok = int(mask_b.sum()) - b_denied
    if min(a_denied + a_ok, b_denied + b_ok) == 0:
        return {"term": term, "Coef.": np.nan, "P>|z|": np.nan, "odds_ratio": np.nan}
    odds_ratio, p = fisher_exact([[a_denied, a_ok], [b_denied, b_ok]])
    return {
        "term": term,
        "Coef.": np.log(odds_ratio) if odds_ratio > 0 else np.nan,
        "P>|z|": p,
        "odds_ratio": odds_ratio,
    }


def marginal_fallback_labels(df, alpha):
    """Marginal Fisher tests when the regression cannot be fit."""
    rows = [
        fisher_term(
            df,
            df["ethnicity"] == "Hispanic or Latino",
            df["ethnicity"] == "Not Hispanic or Latino",
            ETH_TERM,
        ),
        fisher_term(df, df["sex"] == "Female", df["sex"] == "Male", SEX_TERM),
    ]
    for race in df["race"].unique():
        if race == "White":
            continue
        rows.append(
            fisher_term(df, df["race"] == race, df["race"] == "White", f"C(race)[T.{race}]")
        )
    labels = pd.DataFrame(rows)
    labels["significant"] = labels["P>|z|"] < alpha
    labels["adverse"] = labels["Coef."] > 0
    labels["ground_truth_label"] = np.where(
        labels["significant"] & labels["adverse"],
        "BIAS",
        np.where(labels["significant"] & ~labels["adverse"], "FAVORED", "NO_BIAS"),
    )
    return labels


def label_one_sample(csv_path, alpha=0.05):
    df = load_and_clean(csv_path)
    df = collapse_rare_races(df)

    method = "logit"
    labels = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = run_regression(df)
            if model.mle_retvals.get("converged", False):
                labels = extract_ground_truth_labels(model, alpha=alpha)
            else:
                method = "fisher_marginal"
        except Exception:
            method = "fisher_marginal"

    if labels is None:
        labels = marginal_fallback_labels(df, alpha)

    labels = labels.copy()
    labels["method"] = method

    def term_row(term):
        row = labels[labels["term"] == term]
        if row.empty:
            return {"p": np.nan, "or": np.nan, "bias": False}
        r = row.iloc[0]
        return {
            "p": r["P>|z|"],
            "or": r["odds_ratio"],
            "bias": bool(r["ground_truth_label"] == "BIAS"),
        }

    eth = term_row(ETH_TERM)
    sex = term_row(SEX_TERM)
    bias_any = bool((labels["ground_truth_label"] == "BIAS").any())

    summary = {
        "n_rows": len(df),
        "method": method,
        "bias_any": bias_any,
        "bias_latino_female": eth["bias"] or sex["bias"],
        "bias_ethnicity_hispanic": eth["bias"],
        "bias_sex_female": sex["bias"],
        "p_ethnicity_hispanic": eth["p"],
        "or_ethnicity_hispanic": eth["or"],
        "p_sex_female": sex["p"],
        "or_sex_female": sex["or"],
        "n_bias_terms": int((labels["ground_truth_label"] == "BIAS").sum()),
    }
    return summary, labels


def label_samples(samples_dir=PATH_TO_SAMPLES, alpha=0.05, out_suffix=""):
    csv_paths = sorted(glob.glob(os.path.join(samples_dir, "sample_*.csv")))
    if not csv_paths:
        raise FileNotFoundError(f"No sample_*.csv files in {samples_dir}")

    os.makedirs(PATH_TO_GROUND_TRUTH, exist_ok=True)

    summaries, term_frames = [], []
    for k, path in enumerate(csv_paths):
        sample_id = os.path.splitext(os.path.basename(path))[0]
        summary, labels = label_one_sample(path, alpha=alpha)
        summary["sample_id"] = sample_id
        summaries.append(summary)
        labels.insert(0, "sample_id", sample_id)
        term_frames.append(labels)
        if (k + 1) % 50 == 0 or k + 1 == len(csv_paths):
            print(f"  labeled {k + 1}/{len(csv_paths)} samples")

    front = ["sample_id", "n_rows", "method"]
    summary_df = pd.DataFrame(summaries)
    summary_df = summary_df[front + [c for c in summary_df.columns if c not in front]]
    term_df = pd.concat(term_frames, ignore_index=True)

    summary_path = os.path.join(PATH_TO_GROUND_TRUTH, f"sample_labels{out_suffix}.csv")
    term_path = os.path.join(PATH_TO_GROUND_TRUTH, f"sample_term_labels{out_suffix}.csv")
    summary_df.to_csv(summary_path, index=False)
    term_df.to_csv(term_path, index=False)

    print(f"\nLabels written to {summary_path}")
    print(f"Per-term detail written to {term_path}")
    print("\nLabel distribution:")
    print(
        summary_df[["bias_any", "bias_latino_female", "bias_ethnicity_hispanic", "bias_sex_female"]]
        .mean()
        .rename("fraction_true")
        .to_string()
    )
    print(f"\nRegression failures (Fisher fallback): {(summary_df['method'] != 'logit').sum()}")
    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", default=PATH_TO_SAMPLES)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out-suffix", default="", help="suffix for output filenames")
    args = parser.parse_args()

    label_samples(args.samples_dir, args.alpha, args.out_suffix)
