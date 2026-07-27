"""
Draw N samples of X loan-level rows and render a per-sample summary table.

Each sample i produces:
  data/gather_data/samples/sample_{i:04d}.csv          raw rows (ground truth input)
  data/gather_data/samples/sample_{i:04d}_summary.txt  summary table (model input)
plus a manifest.csv describing all samples.

Rows are drawn without replacement within a sample (seed = SAMPLE_SEED + i),
from the subset of preprocessed_data.csv that survives the ground-truth
cleaning step — so the model and the regression always see the same rows.
"""

import argparse
import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import PATH_TO_DATA, PATH_TO_SAMPLES, N_SAMPLES, SAMPLE_SIZE, SAMPLE_SEED
from summarize_data import build_summary

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ground_truth"))
from run_regression import load_and_clean


def load_eligible_rows(data_path):
    """Rows of preprocessed_data.csv (original columns) that survive cleaning."""
    raw = pd.read_csv(data_path, na_values=["NA", "None", "Exempt", "", " "])
    clean = load_and_clean(data_path)  # index is preserved through cleaning
    return raw.loc[clean.index]


def sample_datasets(n_samples=N_SAMPLES, sample_size=SAMPLE_SIZE, seed=SAMPLE_SEED):
    os.makedirs(PATH_TO_SAMPLES, exist_ok=True)

    eligible = load_eligible_rows(f"{PATH_TO_DATA}preprocessed_data.csv")
    print(f"Eligible rows after cleaning: {len(eligible):,}")

    if sample_size > len(eligible):
        raise ValueError(
            f"sample_size {sample_size} exceeds eligible rows {len(eligible)}"
        )

    manifest = []
    for i in range(n_samples):
        sample = eligible.sample(n=sample_size, random_state=seed + i)

        csv_path = os.path.join(PATH_TO_SAMPLES, f"sample_{i:04d}.csv")
        txt_path = os.path.join(PATH_TO_SAMPLES, f"sample_{i:04d}_summary.txt")

        sample.to_csv(csv_path, index=False)

        summary_text = build_summary(sample).to_string(index=False)
        with open(txt_path, "w") as f:
            f.write(summary_text)

        manifest.append(
            {
                "sample_id": f"sample_{i:04d}",
                "seed": seed + i,
                "n_rows": sample_size,
                "csv": os.path.basename(csv_path),
                "summary": os.path.basename(txt_path),
                "summary_chars": len(summary_text),
            }
        )

        if (i + 1) % 50 == 0 or i + 1 == n_samples:
            print(f"  wrote {i + 1}/{n_samples} samples")

    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(os.path.join(PATH_TO_SAMPLES, "manifest.csv"), index=False)
    print(f"Manifest written to {PATH_TO_SAMPLES}manifest.csv")
    return manifest_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    args = parser.parse_args()

    sample_datasets(args.n_samples, args.sample_size, args.seed)
