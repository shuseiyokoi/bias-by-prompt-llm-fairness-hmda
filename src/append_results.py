import json
from datetime import datetime


def append_jsonl(
    model_name: str,
    prompt: str,
    response_text: str,
    file_path: str = "../data/llm_logs.jsonl",
) -> None:
    log_entry = {
        "date": datetime.now().isoformat(),
        "model_name": model_name,
        "prompt": prompt,
        "response": response_text,
    }

    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


# Example usage
if __name__ == "__main__":
    append_jsonl(model_name="gpt-4o", prompt="Hello", response_text="Hi there")
