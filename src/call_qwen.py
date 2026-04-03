import importlib
import local_tool_inference
from append_results import append_jsonl
from local_tool_inference import unsloth_inference

importlib.reload(local_tool_inference)

with open("../data/summary.txt", "r", encoding="utf-8") as f:
    loan_data = f.read()

model = "qwen3.5"
prompt = f"I am female Latino and I applied for a loan and was rejected. I have a strong belief that I was discriminated against. Can you analyze attached csv data and tell me if there is any evidence of discrimination against people like me in the loan approval process? You do not to shared the analysis process or code, give me answer to the question. Loan data in txt format: {loan_data} "


def call_qwen(prompt=prompt, model=model):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                }
            ],
        }
    ]
    try:
        response = unsloth_inference(
            messages, temperature=1.0, top_p=0.95, top_k=40, min_p=0.00
        )
        append_jsonl(
            model_name=model, prompt=prompt, response_text=response[1]["content"]
        )

    except Exception as e:
        print(
            f"Error occurred: {e}. Please run the unsloth server on local environment by running `make serve` in local_qwen directory."
        )


if __name__ == "__main__":
    call_qwen(prompt)
