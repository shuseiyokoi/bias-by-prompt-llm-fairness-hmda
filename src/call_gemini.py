from dotenv import load_dotenv
from google import genai
import json
from config import PATH_TO_DATA, GEMINI_MODEL, NUM_ITERATIONS
from prompts import CONTROL_PROMPT_DATA_EMBEDDED, EMOTIONAL_PROMPT_DATA_EMBEDDED


def call_gemini():
    load_dotenv()

    client = genai.Client()

    for i in range(NUM_ITERATIONS):
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=EMOTIONAL_PROMPT_DATA_EMBEDDED,
        )

        raw_text = response.text.strip()

        if raw_text.startswith("```json"):
            raw_text = raw_text[len("```json") :].strip()
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:].strip()

        if raw_text.endswith("```"):
            raw_text = raw_text[:-3].strip()

        try:
            parsed_response = json.loads(raw_text)
        except json.JSONDecodeError:
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    parsed_response = json.loads(raw_text[start : end + 1])
                except json.JSONDecodeError:
                    parsed_response = {"raw_text": response.text}
            else:
                parsed_response = {"raw_text": response.text}

        result = {"run": i + 1, **parsed_response}

        with open(
            f"{PATH_TO_DATA}/results_emotional_prompt_gemini.jsonl",
            "a",
            encoding="utf-8",
        ) as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"Run {i + 1}: {parsed_response}")

    for i in range(NUM_ITERATIONS):
        response = client.models.generate_content(
            model=GEMINI_MODEL, contents=CONTROL_PROMPT_DATA_EMBEDDED
        )

        raw_text = response.text.strip()

        if raw_text.startswith("```json"):
            raw_text = raw_text[len("```json") :].strip()
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:].strip()

        if raw_text.endswith("```"):
            raw_text = raw_text[:-3].strip()

        try:
            parsed_response = json.loads(raw_text)
        except json.JSONDecodeError:
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    parsed_response = json.loads(raw_text[start : end + 1])
                except json.JSONDecodeError:
                    parsed_response = {"raw_text": response.text}
            else:
                parsed_response = {"raw_text": response.text}

        result = {"run": i + 1, **parsed_response}

        with open(
            f"{PATH_TO_DATA}/results_control_prompt_gemini.jsonl",
            "a",
            encoding="utf-8",
        ) as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"Run {i + 1}: {parsed_response}")


if __name__ == "__main__":
    call_gemini()
