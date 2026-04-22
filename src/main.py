import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, fisher_exact
import matplotlib.ticker as mtick
import os

from load_data import load_data
from summarize_raw_data import summarize_data
from call_chatGPT import call_chatGPT
from call_gemini import call_gemini
from call_claude import call_claude
from config import PATH_TO_DATA, GPT_MODEL, CLAUDE_MODEL, GEMINI_MODEL, PATH_TO_RESULTS


def read_multiple_json_objects(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    decoder = json.JSONDecoder()
    pos = 0
    objects = []

    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            break
        obj, pos = decoder.raw_decode(text, pos)
        objects.append(obj)

    return objects


def load_json_file(filepath, model_name, prompt_name):
    rows = []
    data = read_multiple_json_objects(filepath)

    for row in data:
        if "response" not in row:
            conclusion = row.get("conclusion")
            confidence = row.get("confidence")
        else:
            response = row["response"]

            if isinstance(response, str):
                response = response.strip()
                if response.startswith("```json"):
                    response = (
                        response.replace("```json", "").replace("```", "").strip()
                    )
                response = json.loads(response)

            conclusion = response.get("conclusion")
            confidence = response.get("confidence")

        rows.append(
            {
                "model": model_name,
                "prompt": prompt_name,
                "conclusion": conclusion,
                "confidence": confidence,
            }
        )

    return pd.DataFrame(rows)


def gather_data():
    load_data()
    summarize_data()
    print("Data gathering pipeline started")


def call_models():
    try:
        call_chatGPT()
    except Exception as e:
        print(f"ChatGPT API call failed: {e}")

    try:
        call_gemini()
    except Exception as e:
        print(f"Gemini API call failed: {e}")

    try:
        call_claude()
    except Exception as e:
        print(f"Claude API call failed: {e}")


def analyze_results():
    os.makedirs(PATH_TO_RESULTS, exist_ok=True)
    control_chatgpt_file = f"{PATH_TO_DATA}results_control_prompt_ChatGPT.jsonl"
    emotional_chatgpt_file = f"{PATH_TO_DATA}results_emotional_prompt_ChatGPT.jsonl"

    control_claude_file = f"{PATH_TO_DATA}results_control_prompt_claude.jsonl"
    emotional_claude_file = f"{PATH_TO_DATA}results_emotional_prompt_claude.jsonl"

    control_gemini_file = f"{PATH_TO_DATA}results_control_prompt_gemini.jsonl"
    emotional_gemini_file = f"{PATH_TO_DATA}results_emotional_prompt_gemini.jsonl"

    df = pd.concat(
        [
            load_json_file(control_chatgpt_file, GPT_MODEL[:16], "Control"),
            load_json_file(emotional_chatgpt_file, GPT_MODEL[:16], "Emotional"),
            load_json_file(control_claude_file, CLAUDE_MODEL[:16], "Control"),
            load_json_file(emotional_claude_file, CLAUDE_MODEL[:16], "Emotional"),
            load_json_file(control_gemini_file, GEMINI_MODEL[:16], "Control"),
            load_json_file(emotional_gemini_file, GEMINI_MODEL[:16], "Emotional"),
        ],
        ignore_index=True,
    )

    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    count_table = (
        df.groupby(["model", "prompt", "conclusion"]).size().unstack(fill_value=0)
    )
    count_table.plot(kind="bar", figsize=(12, 6))
    plt.title("Conclusion Counts by Model and Prompt (n=300)")
    plt.ylabel("Count")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(f"{PATH_TO_RESULTS}conclusion-count-by-model-and-prompt.png")
    plt.close()

    percent_table = count_table.div(count_table.sum(axis=1), axis=0) * 100
    ax = percent_table.plot(kind="bar", stacked=True, figsize=(12, 6))

    for container in ax.containers:
        labels = [
            f"{bar.get_height():.1f}%" if bar.get_height() > 0 else ""
            for bar in container
        ]
        ax.bar_label(container, labels=labels, label_type="center")

    ax.set_title("Conclusion Percentages by Model and Prompt (n=300 each)")
    ax.set_ylabel("Percentage")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.xticks(rotation=45)
    ax.legend(title="Conclusion", bbox_to_anchor=(1, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{PATH_TO_RESULTS}percentages-by-model-and-prompt.png")
    plt.close()

    df["group"] = df["model"] + " - " + df["prompt"]

    fig, ax = plt.subplots(figsize=(12, 6))
    df.boxplot(column="confidence", by="group", ax=ax)
    ax.set_title("Confidence by Model and Prompt (n=300 each)")
    ax.set_ylabel("Confidence")
    ax.tick_params(axis="x", rotation=45)
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(f"{PATH_TO_RESULTS}confidence-by-model-and-prompt.png")
    plt.close(fig)

    df["group"] = df["model"] + " - " + df["conclusion"]

    fig, ax = plt.subplots(figsize=(6, 6))
    df.boxplot(column="confidence", by="group", ax=ax)
    ax.set_title("Confidence by Model and Conclusion")
    ax.set_ylabel("Confidence")
    ax.tick_params(axis="x", rotation=45)
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(f"{PATH_TO_RESULTS}confidence-by-model-and-conclusion-small.png")
    plt.close(fig)

    temp = df.groupby(["model", "prompt", "conclusion"]).size()
    temp.to_csv(f"{PATH_TO_RESULTS}tableresults.csv")

    sub = df[df["model"] == "gemini-2.5-flash"].copy()
    sub["prompt"] = pd.Categorical(sub["prompt"], categories=["Control", "Emotional"])
    table = pd.crosstab(sub["prompt"], sub["conclusion"])
    print("\nGemini:")
    print(table)
    print(f"Chi-square p = {chi2_contingency(table)[1]:.3g}")
    print(f"Fisher p     = {fisher_exact(table)[1]:.3g}")

    sub = df[df["model"] == "gpt-3.5-turbo"].copy()
    sub["prompt"] = pd.Categorical(sub["prompt"], categories=["Control", "Emotional"])
    table = pd.crosstab(sub["prompt"], sub["conclusion"])
    print("\nChatGPT:")
    print(table)
    print(f"Chi-square p = {chi2_contingency(table)[1]:.3g}")
    print(f"Fisher p     = {fisher_exact(table)[1]:.3g}")

    print("Analysis completed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gather-data", action="store_true", help="Load and summarize data"
    )
    parser.add_argument(
        "--call-models", action="store_true", help="Run model API calls"
    )
    parser.add_argument(
        "--analyze", action="store_true", help="Analyze saved model results"
    )
    args = parser.parse_args()

    if not args.gather_data and not args.call_models and not args.analyze:
        print("No mode selected")
        return

    if args.gather_data:
        try:
            gather_data()
        except Exception as e:
            print(f"Error occurred while gathering data: {e}")

    if args.call_models:
        call_models()

    if args.analyze:
        try:
            analyze_results()
        except Exception as e:
            print(f"Error occurred while analyzing results: {e}")


if __name__ == "__main__":
    main()
