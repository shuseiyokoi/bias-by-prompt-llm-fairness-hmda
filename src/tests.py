from load_data import load_data
from summarize_raw_data import summarize_data
from call_chatGPT import call_chatGPT
from call_gemini import call_gemini
from call_claude import call_claude
from call_qwen import call_qwen

load_data()
summarize_data()

try:
    call_chatGPT()
except:
    print("ChatGPT API call failed. Please check your API key and network connection.")
    pass

try:
    call_gemini()
except:
    print("Gemini API call failed. Please check your API key and network connection.")
    pass
try:
    call_claude()
except:
    print("Claude API call failed. Please check your API key and network connection.")
    pass
try:
    call_qwen()
except:
    print(
        "Qwen API call failed. Please check your local server and network connection."
    )
    pass
