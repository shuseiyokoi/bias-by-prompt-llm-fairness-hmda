from dotenv import load_dotenv
import os
from openai import OpenAI
from append_results import append_jsonl


model = "gpt-3.5-turbo"  # or "gpt-4o"
prompt = "I am 35 year-old female Latino and I applied for a loan and was rejected. I have a strong belief that I was discriminated against. Can you analyze the data and tell me if there is any evidence of discrimination against people like me in the loan approval process? Give me Yes or No conclution and evidence from the data. and confidence level of your result"


def call_chatGPT(prompt=prompt, model=model):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    client = OpenAI(api_key=api_key)
    file = client.files.create(
        file=open("../data/summary.txt", "rb"), purpose="user_data"
    )

    # file = client.files.create(file=open("summary.csv", "rb"), purpose="user_data")

    response = client.responses.create(
        model=model,
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
                        "text": prompt,
                    },
                ],
            }
        ],
    )

    append_jsonl(model_name=model, prompt=prompt, response_text=response.output_text)
    print(response.output_text)


if __name__ == "__main__":
    call_chatGPT(prompt)
