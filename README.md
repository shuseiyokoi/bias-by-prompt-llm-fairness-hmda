This project scope changed because sending raw loan-level HMDA data to LLMs used too many tokens and took too long. In testing, a 1,000-row CSV used about 53,000 tokens and took 5 to 6 minutes per request.

To improve efficiency, the study now uses pre-analyzed statistical summaries instead of raw data. This allows the project to focus more clearly on how emotional and identity-based framing affects LLM conclusions. It also reduces the cost dramatically, bringing usage down to about 1,500 tokens and response time to about 5 to 6 seconds.

The project uses 2024 California HMDA loan application data and focuses on three outcomes: loan originated, application approved but not accepted, and application denied. After downloading the data, summarize_raw_data.py groups records by sensitive attributes and outcome while keeping important fields such as ethnicity, race, sex, loan amount, income, and applicant age.

The project uses one API to retrieve HMDA data and three model APIs: OpenAI, Gemini, and Anthropic. Current challenges include converting CSV data into text for model input and making model outputs more consistent by requesting JSON-formatted responses.
