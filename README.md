# Introduction
This is an easy to use chatbot I built to help answer medical questions.
I create use langchain's ollama integration and I provide it 3 tools.
The first tool is a search engine (I used brave), the next tool is wikipedia, and finally pubmed.
The main point is for specific medical questions I have the LLM do what I would do,
which is to use a search engine with some reputable source like mayo clinic or cdc.
If it is something more in depth but related to common knowledge then wikipedia may be more helpful.
Finally if it is related to cutting edge medical research the pubmed tool may be of use. The LLM
shoudl choose the best tool and then intepret the tool's output to give the user an answer.
# Installation
## create venv using UV
git clone https://github.com/jalbrethsen/medical-chatbot.git
cd medical-chatbot
uv venv
uv pip install -r requirements.txt
## set .env file
```
cp .env.example .env
```
Add in your own brave search key and point to your local OLLAMA instance
# Usage 
## use uv to run everything in the python environment
uv run streamlit run brave_med_chatbot.py

