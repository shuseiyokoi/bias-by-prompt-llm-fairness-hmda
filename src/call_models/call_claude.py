import os
import sys
import anthropic
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import PATH_TO_MODEL_RESULTS, CLAUDE_MODELS, PROMPT_TYPES
from sample_runner import run_sample_set


def call_claude():
    load_dotenv()

    os.makedirs(PATH_TO_MODEL_RESULTS, exist_ok=True)

    client = anthropic.Anthropic()

    for model_name in CLAUDE_MODELS:

        def send_fn(prompt_text):
            message = client.messages.create(
                model=model_name,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt_text}],
            )
            return message.content[0].text

        for prompt_type in PROMPT_TYPES:
            output_file = f"{PATH_TO_MODEL_RESULTS}sample_results_{prompt_type}_{model_name}.jsonl"

            print(f"\nStarting: {model_name} | {prompt_type}")
            run_sample_set(send_fn, model_name, prompt_type, output_file)
            print(f"Finished: {model_name} | {prompt_type}")


if __name__ == "__main__":
    call_claude()
