from dotenv import load_dotenv
from google import genai
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import PATH_TO_MODEL_RESULTS, GEMINI_MODELS, prompt_identity_pairs, prompt_identity_label
from sample_runner import run_sample_set


def call_gemini():
    load_dotenv()

    os.makedirs(PATH_TO_MODEL_RESULTS, exist_ok=True)

    client = genai.Client()

    for model_name in GEMINI_MODELS:

        def send_fn(prompt_text):
            response = client.models.generate_content(
                model=model_name,
                contents=[prompt_text],
            )
            return response.text

        for prompt_type, identity in prompt_identity_pairs():
            label = prompt_identity_label(prompt_type, identity)
            output_file = f"{PATH_TO_MODEL_RESULTS}sample_results_{label}_{model_name}.jsonl"

            print(f"\nStarting: {model_name} | {label}")
            run_sample_set(send_fn, model_name, prompt_type, output_file, identity=identity)
            print(f"Finished: {model_name} | {label}")


if __name__ == "__main__":
    call_gemini()
