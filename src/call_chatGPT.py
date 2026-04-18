from dotenv import load_dotenv
import os
from openai import OpenAI
from append_results import append_jsonl
import json
from config import (
    PATH_TO_DATA,
    GPT_MODEL,
    CONTROL_PROMPT,
    EMOTIONAL_PROMPT,
    NUM_ITERATIONS,
)


def main():
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)

    file = client.files.create(
        file=open(f"{PATH_TO_DATA}/summary.txt", "rb"), purpose="user_data"
    )

    for i in range(NUM_ITERATIONS):
        response = client.responses.create(
            model=GPT_MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "file_id": file.id,
                        },
                        {
                            "type": "input_text",
                            "text": CONTROL_PROMPT,
                        },
                    ],
                }
            ],
        )

        raw_text = response.output_text.strip()

        if raw_text.startswith("```json"):
            raw_text = raw_text[len("```json") :].strip()

        if raw_text.startswith("```"):
            raw_text = raw_text[len("```") :].strip()

        if raw_text.endswith("```"):
            raw_text = raw_text[:-3].strip()

        parsed = json.loads(raw_text)

        result = {"run": i + 1, "response": parsed}

        with open(
            f"{PATH_TO_DATA}/results_control_prompt_ChatGPT.jsonl",
            "a",
            encoding="utf-8",
        ) as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"Run {i + 1}: {parsed}")

    for i in range(NUM_ITERATIONS):
        response = client.responses.create(
            model=GPT_MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "file_id": file.id,
                        },
                        {
                            "type": "input_text",
                            "text": EMOTIONAL_PROMPT,
                        },
                    ],
                }
            ],
        )

        raw_text = response.output_text.strip()

        if raw_text.startswith("```json"):
            raw_text = raw_text[len("```json") :].strip()

        if raw_text.startswith("```"):
            raw_text = raw_text[len("```") :].strip()

        if raw_text.endswith("```"):
            raw_text = raw_text[:-3].strip()

        parsed = json.loads(raw_text)

        result = {"run": i + 1, "response": parsed}

        with open(
            f"{PATH_TO_DATA}/results_emotional_prompt_ChatGPT.jsonl",
            "a",
            encoding="utf-8",
        ) as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"Run {i + 1}: {parsed}")


if __name__ == "__main__":
    main()
