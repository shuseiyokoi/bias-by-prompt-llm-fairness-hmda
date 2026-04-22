import json
import anthropic
from dotenv import load_dotenv

from config import PATH_TO_DATA, CLAUDE_MODEL, NUM_ITERATIONS
from prompts import get_control_prompt_embedded, get_emotional_prompt_embedded


def parse_json_response(raw_text):
    raw_text = raw_text.strip()

    if raw_text.startswith("```json"):
        raw_text = raw_text[len("```json") :].strip()
    elif raw_text.startswith("```"):
        raw_text = raw_text[3:].strip()

    if raw_text.endswith("```"):
        raw_text = raw_text[:-3].strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw_text[start : end + 1])
            except json.JSONDecodeError:
                return {"raw_text": raw_text}
        return {"raw_text": raw_text}


def run_prompt_set(client, prompt_text, output_file):
    for i in range(NUM_ITERATIONS):
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt_text}],
        )

        raw_text = message.content[0].text.strip()
        parsed_response = parse_json_response(raw_text)

        result = {"run": i + 1, "response": parsed_response}

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"Run {i + 1}: {parsed_response}")


def call_claude():
    load_dotenv()

    client = anthropic.Anthropic()

    emotional_prompt = get_emotional_prompt_embedded()
    control_prompt = get_control_prompt_embedded()

    run_prompt_set(
        client=client,
        prompt_text=emotional_prompt,
        output_file=f"{PATH_TO_DATA}results_emotional_prompt_claude.jsonl",
    )

    run_prompt_set(
        client=client,
        prompt_text=control_prompt,
        output_file=f"{PATH_TO_DATA}results_control_prompt_claude.jsonl",
    )


if __name__ == "__main__":
    call_claude()
