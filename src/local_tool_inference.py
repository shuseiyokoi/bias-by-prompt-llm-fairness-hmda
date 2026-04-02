import json, subprocess, random
from typing import Any


def add_number(a: float | str, b: float | str) -> float:
    return float(a) + float(b)


def multiply_number(a: float | str, b: float | str) -> float:
    return float(a) * float(b)


def substract_number(a: float | str, b: float | str) -> float:
    return float(a) - float(b)


def write_a_story() -> str:
    return random.choice(
        [
            "A long time ago in a galaxy far far away...",
            "There were 2 friends who loved sloths and code...",
            "The world was ending because every sloth evolved to have superhuman intelligence...",
            "Unbeknownst to one friend, the other accidentally coded a program to evolve sloths...",
        ]
    )


def terminal(command: str) -> str:
    if "rm" in command or "sudo" in command or "dd" in command or "chmod" in command:
        msg = "Cannot execute 'rm, sudo, dd, chmod' commands since they are dangerous"
        print(msg)
        return msg
    print(f"Executing terminal command `{command}`")
    try:
        return str(
            subprocess.run(
                command, capture_output=True, text=True, shell=True, check=True
            ).stdout
        )
    except subprocess.CalledProcessError as e:
        return f"Command failed: {e.stderr}"


def python(code: str) -> str:
    data = {}
    exec(code, data)
    del data["__builtins__"]
    return str(data)


MAP_FN = {
    "add_number": add_number,
    "multiply_number": multiply_number,
    "substract_number": substract_number,
    "write_a_story": write_a_story,
    "terminal": terminal,
    "python": python,
}
tools = [
    {
        "type": "function",
        "function": {
            "name": "add_number",
            "description": "Add two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "string",
                        "description": "The first number.",
                    },
                    "b": {
                        "type": "string",
                        "description": "The second number.",
                    },
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multiply_number",
            "description": "Multiply two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "string",
                        "description": "The first number.",
                    },
                    "b": {
                        "type": "string",
                        "description": "The second number.",
                    },
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "substract_number",
            "description": "Substract two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "string",
                        "description": "The first number.",
                    },
                    "b": {
                        "type": "string",
                        "description": "The second number.",
                    },
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_a_story",
            "description": "Writes a random story.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "Perform operations from the terminal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command you wish to launch, e.g `ls`, `rm`, ...",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python",
            "description": "Call a Python interpreter with some Python code that will be ran.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to run",
                    },
                },
                "required": ["code"],
            },
        },
    },
]

from openai import OpenAI


def unsloth_inference(
    messages,
    temperature=1.0,
    top_p=0.95,
    top_k=40,
    min_p=0.01,
    repetition_penalty=1.0,
):
    messages = messages.copy()
    openai_client = OpenAI(
        base_url="http://127.0.0.1:8001/v1",
        api_key="sk-no-key-required",
    )
    model_name = next(iter(openai_client.models.list())).id
    print(f"Using model = {model_name}")
    has_tool_calls = True
    original_messages_len = len(messages)
    while has_tool_calls:
        print(f"Current messages = {messages}")
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            tools=tools if tools else None,
            tool_choice="auto" if tools else None,
            extra_body={
                "top_k": top_k,
                "min_p": min_p,
                "repetition_penalty": repetition_penalty,
            },
        )
        tool_calls = response.choices[0].message.tool_calls or []
        content = response.choices[0].message.content or ""
        tool_calls_dict = (
            [tc.to_dict() for tc in tool_calls] if tool_calls else tool_calls
        )
        messages.append(
            {
                "role": "assistant",
                "tool_calls": tool_calls_dict,
                "content": content,
            }
        )
        for tool_call in tool_calls:
            fx, args, _id = (
                tool_call.function.name,
                tool_call.function.arguments,
                tool_call.id,
            )
            out = MAP_FN[fx](**json.loads(args))
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": _id,
                    "name": fx,
                    "content": str(out),
                }
            )
        else:
            has_tool_calls = False
    return messages


if __name__ == "__main__":
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Create a Fibonacci function in Python and find fib(20).",
                }
            ],
        }
    ]
    results = unsloth_inference(
        messages, temperature=1.0, top_p=0.95, top_k=40, min_p=0.00
    )
    print(f"Final messages = {results}")
