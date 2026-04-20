from load_data import load_data
from summarize_raw_data import summarize_data

try:
    load_data()
    summarize_data()
except Exception as e:
    print(f"Error occurred while loading or summarizing data: {e}")
