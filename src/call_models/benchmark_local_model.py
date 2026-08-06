"""
Standalone latency/token/memory benchmark for one local llama.cpp model x
one prompt type, over N samples. Does not touch the real pipeline output
(data/call_models/) or any existing config/call_* files.

Usage:
    python benchmark_local_model.py --model qwen2.5-7b-instruct \
        --prompt-type control_prompt --n-samples 100

Writes results/benchmark/benchmark_{model}_{prompt_type}_n{N}_{run_id}.csv
(one row per call) and a matching ....summary.json (aggregate stats).
"""

import argparse
import csv
import json
import os
import resource
import statistics
import sys
import time
from datetime import datetime, timezone

from openai import OpenAI

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import ROOT_DIR

from call_qwen import LOCAL_SERVER_CONFIG, BASE_URL, start_server, stop_server
from prompts import list_sample_ids, get_sample_prompt, USER_STATEMENTS
from sample_runner import parse_json_response

PATH_TO_BENCHMARK_RESULTS = os.path.join(ROOT_DIR, "results", "benchmark") + os.sep

CSV_FIELDNAMES = [
    "iteration", "sample_id", "latency_s", "prompt_tokens",
    "completion_tokens", "total_tokens", "parsed_ok", "error", "timestamp",
]


def peak_rss_mb():
    """Process's own peak RSS, normalized to MB (macOS reports bytes, Linux reports KB)."""
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return max_rss / (1024 ** 2) if sys.platform == "darwin" else max_rss / 1024


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def select_samples(n_samples):
    """First N sample ids, same convention as call_qwen.py's --smoke."""
    ids = list_sample_ids()
    if n_samples > len(ids):
        raise ValueError(f"--n-samples {n_samples} exceeds available samples ({len(ids)})")
    return ids[:n_samples]


def run_one_call(client, model_name, prompt_type, sample_id, iteration):
    """One timed API call. Never raises -- catches its own exceptions and
    returns a row with error info instead, so one bad call can't kill the
    other iterations."""
    prompt_text = get_sample_prompt(prompt_type, sample_id)
    t0 = time.perf_counter()
    try:
        completion = client.chat.completions.create(
            model=model_name,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt_text}],
        )
        latency_s = time.perf_counter() - t0
        usage = completion.usage
        content = completion.choices[0].message.content
        parsed = parse_json_response(content.strip())
        return {
            "iteration": iteration,
            "sample_id": sample_id,
            "latency_s": latency_s,
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "total_tokens": usage.total_tokens if usage else None,
            "parsed_ok": "raw_text" not in parsed,
            "error": "",
            "timestamp": now_iso(),
        }
    except Exception as e:
        return {
            "iteration": iteration,
            "sample_id": sample_id,
            "latency_s": time.perf_counter() - t0,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "parsed_ok": False,
            "error": str(e),
            "timestamp": now_iso(),
        }


def write_summary(json_path, model_name, prompt_type, n_samples, rows,
                   wall_time_s, start_ts, end_ts, csv_path):
    ok_rows = [r for r in rows if not r["error"]]
    latencies = [r["latency_s"] for r in ok_rows]
    summary = {
        "model": model_name,
        "prompt_type": prompt_type,
        "n_samples": n_samples,
        "n_errors": n_samples - len(ok_rows),
        "n_parsed_ok": sum(1 for r in rows if r["parsed_ok"]),
        "total_wall_time_s": wall_time_s,
        "latency_s": {
            "mean": statistics.mean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "tokens": {
            "total_prompt_tokens": sum(r["prompt_tokens"] for r in ok_rows),
            "total_completion_tokens": sum(r["completion_tokens"] for r in ok_rows),
            "total_tokens": sum(r["total_tokens"] for r in ok_rows),
            "mean_tokens_per_call": (
                sum(r["total_tokens"] for r in ok_rows) / len(ok_rows) if ok_rows else None
            ),
        },
        "process_peak_rss_mb": peak_rss_mb(),
        "start_time_iso": start_ts,
        "end_time_iso": end_ts,
        "csv_file": os.path.basename(csv_path),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def run_benchmark(model_name, prompt_type, n_samples):
    os.makedirs(PATH_TO_BENCHMARK_RESULTS, exist_ok=True)
    sample_ids = select_samples(n_samples)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    base = f"benchmark_{model_name}_{prompt_type}_n{n_samples}_{run_id}"
    csv_path = f"{PATH_TO_BENCHMARK_RESULTS}{base}.csv"
    json_path = f"{PATH_TO_BENCHMARK_RESULTS}{base}.summary.json"

    client = OpenAI(base_url=BASE_URL, api_key="local")
    proc = start_server(model_name)
    rows = []
    start_ts = now_iso()
    wall_t0 = time.perf_counter()
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            for i, sample_id in enumerate(sample_ids):
                row = run_one_call(client, model_name, prompt_type, sample_id, i)
                rows.append(row)
                writer.writerow(row)
                f.flush()
                print(
                    f"[{i + 1}/{n_samples}] {sample_id}: {row['latency_s']:.2f}s "
                    f"ok={row['parsed_ok']} err={row['error'] or '-'}"
                )
    finally:
        stop_server(proc)

    wall_time_s = time.perf_counter() - wall_t0
    end_ts = now_iso()
    write_summary(json_path, model_name, prompt_type, n_samples, rows,
                  wall_time_s, start_ts, end_ts, csv_path)
    print(f"\nWrote {csv_path}\nWrote {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Latency/token/memory benchmark for one local llama.cpp model x one prompt type."
    )
    parser.add_argument("--model", default="qwen2.5-7b-instruct", choices=sorted(LOCAL_SERVER_CONFIG.keys()))
    parser.add_argument("--prompt-type", default="control_prompt", choices=sorted(USER_STATEMENTS.keys()))
    parser.add_argument("--n-samples", type=int, default=100)
    args = parser.parse_args()

    run_benchmark(args.model, args.prompt_type, args.n_samples)
