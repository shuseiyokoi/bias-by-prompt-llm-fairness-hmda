from dotenv import load_dotenv
from google import genai
from google.genai import types
from append_results import append_jsonl
from config import PATH_TO_DATA

with open(f"{PATH_TO_DATA}/summary.txt", "r", encoding="utf-8") as f:
    loan_data = f.read()
model = "gemini-2.5-flash-lite"  # or "gpt-4o"
prompt = f"I am female Latino and I applied for a loan and was rejected. I have a strong belief that I was discriminated against. Can you analyze attached csv data and tell me if there is any evidence of discrimination against people like me in the loan approval process? You do not to shared the analysis process or code, give me answer to the question. Loan data in txt format: {loan_data} "


def call_gemini(prompt=prompt, model=model):
    load_dotenv()

    client = genai.Client()

    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    append_jsonl(model_name=model, prompt=prompt, response_text=response.text)


if __name__ == "__main__":
    call_gemini(prompt)
