from dotenv import load_dotenv
import os
import sys
from openai import OpenAI

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import PATH_TO_MODEL_RESULTS, GPT_MODELS, prompt_identity_pairs, prompt_identity_label
from sample_runner import run_sample_set


def call_chatGPT():
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")

    os.makedirs(PATH_TO_MODEL_RESULTS, exist_ok=True)

    client = OpenAI(api_key=api_key)

    for model_name in GPT_MODELS:

        def send_fn(prompt_text):
            response = client.responses.create(
                model=model_name,
                input=[{"role": "user", "content": prompt_text}],
            )
            return response.output_text

        for prompt_type, identity in prompt_identity_pairs():
            label = prompt_identity_label(prompt_type, identity)
            output_file = f"{PATH_TO_MODEL_RESULTS}sample_results_{label}_{model_name}.jsonl"

            print(f"\nStarting: {model_name} | {label}")
            run_sample_set(send_fn, model_name, prompt_type, output_file, identity=identity)
            print(f"Finished: {model_name} | {label}")


if __name__ == "__main__":
    call_chatGPT()
