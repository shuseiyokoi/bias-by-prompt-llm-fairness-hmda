from dotenv import load_dotenv
from anthropic import Anthropic
from append_results import append_jsonl


with open("../data/summary.txt", "r", encoding="utf-8") as f:
    loan_data = f.read()

model = "claude-haiku-3"  # or "gpt-4o"
prompt = f"I am female Latino and I applied for a loan and was rejected. I have a strong belief that I was discriminated against. Can you analyze attached csv data and tell me if there is any evidence of discrimination against people like me in the loan approval process? You do not to shared the analysis process or code, give me answer to the question. Loan data in csv format: {loan_data} "


def call_claude(prompt=prompt, model=model):
    load_dotenv()

    client = Anthropic()  #
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )
    append_jsonl(model_name=model, prompt=prompt, response_text=message.output_text)


if __name__ == "__main__":
    call_claude(prompt)
