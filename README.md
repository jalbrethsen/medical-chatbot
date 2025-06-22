# Installation
## create venv using UV
uv init
uv add -r requirements.txt
# Usage (assuming you have ollama running locally with llama3.2:3b downloaded)
## use uv to run everything in the python environment
uv run streamlit run brave_med_chatbot.py

