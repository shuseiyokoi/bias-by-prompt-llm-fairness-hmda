import anthropic
from dotenv import load_dotenv
from anthropic import Anthropic
from append_results import append_jsonl
import json
from config import (
    PATH_TO_DATA,
    CLAUDE_MODEL,
    CONTROL_PROMPT_DATA_EMBEDDED,
    EMOTIONAL_PROMPT_DATA_EMBEDDED,
)


def main():
    load_dotenv()

    client = anthropic.Anthropic()

    for i in range(300):
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": EMOTIONAL_PROMPT_DATA_EMBEDDED}],
        )

        raw_text = message.content[0].text.strip()

        if raw_text.startswith("```json"):
            raw_text = raw_text[len("```json") :].strip()
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:].strip()

        if raw_text.endswith("```"):
            raw_text = raw_text[:-3].strip()

        try:
            parsed_response = json.loads(raw_text)
        except json.JSONDecodeError:
            parsed_response = {"raw_text": message.content[0].text}

        result = {"run": i + 1, "response": parsed_response}

        with open(
            f"{PATH_TO_DATA}/results_emotional_prompt_claude.jsonl",
            "a",
            encoding="utf-8",
        ) as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"Run {i + 1}: {parsed_response}")

    for i in range(300):
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": CONTROL_PROMPT_DATA_EMBEDDED}],
        )

        raw_text = message.content[0].text.strip()

        if raw_text.startswith("```json"):
            raw_text = raw_text[len("```json") :].strip()
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:].strip()

        if raw_text.endswith("```"):
            raw_text = raw_text[:-3].strip()

        try:
            parsed_response = json.loads(raw_text)
        except json.JSONDecodeError:
            parsed_response = {"raw_text": message.content[0].text}

        result = {"run": i + 1, "response": parsed_response}

        with open(
            f"{PATH_TO_DATA}/results_control_prompt_claude.jsonl", "a", encoding="utf-8"
        ) as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"Run {i + 1}: {parsed_response}")


if __name__ == "__main__":
    main()
