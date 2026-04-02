MODEL=/Users/shusei/models/qwen2.5-7b-instruct-q4_k_m.gguf
PORT=8001
SERVER=./llama.cpp/build/bin/llama-server

.PHONY: serve test

serve:
	$(SERVER) -m "$(MODEL)" --port $(PORT)

test:
	curl http://127.0.0.1:$(PORT)/v1/models