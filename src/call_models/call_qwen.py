import argparse
import os
import subprocess
import sys
import time
import urllib.request

from openai import OpenAI

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import PATH_TO_MODEL_RESULTS, LOCAL_QWEN_DIR, LOCAL_MODELS, PROMPT_TYPES
from prompts import list_sample_ids
from sample_runner import run_sample_set, completed_sample_ids

LLAMA_SERVER = os.getenv("LLAMA_SERVER", "llama-server")
PORT = int(os.getenv("QWEN_PORT", "8080"))
BASE_URL = f"http://localhost:{PORT}/v1"

MODELS_DIR = os.path.join(LOCAL_QWEN_DIR, "models")

# GGUF file and extra llama-server flags for each model in config.LOCAL_MODELS
LOCAL_SERVER_CONFIG = {
    "qwen2.5-7b-instruct": {
        "gguf": os.path.join(MODELS_DIR, "Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
        # this GGUF's baked-in context_length (32768) makes llama-server cap
        # the slot to 32768 regardless of -c; override the metadata so the
        # cap isn't applied, and scale RoPE via YaRN so the model still
        # attends sanely out to the full -c (raw-mode prompts run ~111.6k)
        "extra_args": [
            "--override-kv", "qwen2.context_length=int:131072",
            "--rope-scaling", "yarn",
            "--rope-scale", "4",
            "--yarn-orig-ctx", "32768",
        ],
    },
    "qwen3-8b": {
        "gguf": os.path.join(MODELS_DIR, "Qwen3-8B-Q4_K_M.gguf"),
        # disable thinking so the output is only the strict JSON answer
        "extra_args": ["--reasoning-budget", "0"],
    },
    "llama-3.1-8b-instruct": {
        "gguf": os.path.join(MODELS_DIR, "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
        "extra_args": [],
    },
    "llama-3.2-3b-instruct": {
        "gguf": os.path.join(MODELS_DIR, "Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
        "extra_args": [],
    },
    "gemma-2-9b-it": {
        "gguf": os.path.join(MODELS_DIR, "gemma-2-9b-it-Q4_K_M.gguf"),
        "extra_args": [],
    },
    "gemma-3-12b-it": {
        "gguf": os.path.join(MODELS_DIR, "google_gemma-3-12b-it-Q4_K_M.gguf"),
        "extra_args": [],
    },
}


def server_is_up():
    try:
        with urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def start_server(model_name):
    if server_is_up():
        raise RuntimeError(
            f"A server is already running on port {PORT}. Stop it first so "
            "results are labeled with the model this script loads."
        )

    server_config = LOCAL_SERVER_CONFIG[model_name]
    if not os.path.exists(server_config["gguf"]):
        raise FileNotFoundError(
            f"{server_config['gguf']} not found. Run `make download` in local_qwen first."
        )

    proc = subprocess.Popen(
        [
            LLAMA_SERVER,
            "-m", server_config["gguf"],
            "--port", str(PORT),
            "-c", "120000",  # raw-mode prompts run up to ~111.6k tokens + completion; 100000 (old value) was too tight
            "-ngl", "99",
            "--parallel", "1",  # avoid dividing -c across llama.cpp's default 4 slots
            *server_config["extra_args"],
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for _ in range(60):  # model load can take a few minutes
        if server_is_up():
            print(f"llama.cpp server is up with {model_name}")
            return proc
        if proc.poll() is not None:
            raise RuntimeError(
                f"llama-server exited while loading {model_name} "
                f"(exit code {proc.returncode})"
            )
        time.sleep(5)

    proc.terminate()
    raise RuntimeError(f"Server for {model_name} was not healthy after 5 minutes")


def stop_server(proc):
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


def call_qwen(prompt_types=PROMPT_TYPES, sample_ids=None, output_prefix="sample_results"):
    os.makedirs(PATH_TO_MODEL_RESULTS, exist_ok=True)

    if sample_ids is None:
        sample_ids = list_sample_ids()

    client = OpenAI(base_url=BASE_URL, api_key="local")

    for model_name in LOCAL_MODELS:
        # resume support: skip (model, prompt) pairs whose output file already
        # covers every sample; run_sample_set fills in partial ones
        pending = []
        for prompt_type in prompt_types:
            output_file = f"{PATH_TO_MODEL_RESULTS}{output_prefix}_{prompt_type}_{model_name}.jsonl"
            done = completed_sample_ids(output_file)
            if all(s in done for s in sample_ids):
                print(f"Skipping {model_name} | {prompt_type}: all samples recorded")
            else:
                pending.append((prompt_type, output_file))

        if not pending:
            print(f"Skipping {model_name}: all prompt types complete")
            continue

        proc = start_server(model_name)

        def send_fn(prompt_text):
            completion = client.chat.completions.create(
                model=model_name,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt_text}],
            )
            return completion.choices[0].message.content

        try:
            for prompt_type, output_file in pending:
                print(f"\nStarting: {model_name} | {prompt_type}")
                run_sample_set(
                    send_fn,
                    model_name,
                    prompt_type,
                    output_file,
                    sample_ids=sample_ids,
                    sleep_s=0,
                )
                print(f"Finished: {model_name} | {prompt_type}")
        finally:
            stop_server(proc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="quick test: control_prompt only, 1 sample per model, writes to smoke_*.jsonl",
    )
    args = parser.parse_args()

    if args.smoke:
        call_qwen(
            prompt_types=["control_prompt"],
            sample_ids=list_sample_ids()[:1],
            output_prefix="smoke",
        )
    else:
        call_qwen()
