import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import (
    WikipediaAPIWrapper,
)  # Needed for WikipediaQueryRun
from langchain_community.tools.brave_search.tool import BraveSearch

load_dotenv()
# --- Configuration ---
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL")
OLLAMA_MODEL = os.environ.get(
    "OLLAMA_MODEL"
)  # Ensure this model is available in your Ollama instance
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY")
# --- LLM Setup ---
try:
    model = ChatOllama(
        model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, extract_reasoning=True
    )
except Exception as e:
    st.error(
        f"Failed to initialize LLM: {e}. Ensure Ollama is running and the model '{OLLAMA_MODEL}' is available at {OLLAMA_BASE_URL}."
    )
    st.stop()  # Stop the script if LLM can't be loaded

# --- Tool Definitions ---


@tool
def brave_search(query: str) -> str:
    """
    Searches BraveSearch for general medical information, prioritizing results from expert medical sources.
    Use this for specific medical questions that require up-to-date external knowledge from reputable sources.
    The input should be the user's medical query prepended with mayoclinic, nih.gov, cdc.gov, or another authoritative medical source.
    """
    st.info(f"🛠️ Performing Brave Search for: {query}")
    try:
        search_tool = BraveSearch.from_api_key(
            api_key=BRAVE_API_KEY, search_kwargs={"count": 5}
        )
        # Prioritize Mayo Clinic but also allow for broader search if no specific hit
        # This could be further refined, e.g., try "mayoclinic {query}" then just "{query}"
        results = search_tool.invoke(query)

        if not results:
            return (
                "I couldn't find relevant information using BraveSearch for your query."
            )

        prompt_template = ChatPromptTemplate.from_template(
            "You are an expert medical information summarizer. "
            "Based on the following Brave search results, answer the user's query: '{user_query}'. "
            "Provide a concise summary of no more than 150 words with references from the search results. "
            "Focus on the most relevant information and present it clearly. "
            "Do not add any information not present in the search results. "
            "If the information seems to offer medical advice, reiterate that the user should consult a healthcare professional.\n\n"
            "Search Results:\n{search_results}"
        )
        chain = prompt_template | model | StrOutputParser()
        return chain.invoke({"user_query": query, "search_results": results})
    except Exception as e:
        print(f"Error in brave_search: {e}")
        return "Sorry, I encountered an error while searching with Brave. Please try again."


@tool
def pubmed_search(query: str) -> str:
    """
    Searches PubMed for medical research papers, abstracts, and scientific studies.
    Use this for in-depth medical questions, inquiries about specific research, or when looking for scientific evidence.
    The input should be a specific medical topic or research question.
    """
    st.info(f"🛠️ Performing PubMed Search for: {query}")
    try:
        pubmed_tool = PubmedQueryRun()
        results = pubmed_tool.invoke(query)

        if (
            not results or "No results found" in results
        ):  # PubmedQueryRun can return "No results found."
            return "No relevant articles were found on PubMed for your query."

        prompt_template = ChatPromptTemplate.from_template(
            "You are an expert at summarizing medical research from PubMed. "
            "Based on the following PubMed abstracts/results, answer the user's query: '{user_query}'. "
            "Provide a concise summary highlighting key findings and conclusions, in no more than 200 words. "
            "Mention that this information is from research articles and may require expert interpretation. "
            "Do not add any information not present in the search results.\n\n"
            "PubMed Results:\n{pubmed_results}"
        )
        chain = prompt_template | model | StrOutputParser()
        # Truncate results if too long for the prompt context window
        return chain.invoke({"user_query": query, "pubmed_results": results[:4000]})
    except Exception as e:
        print(f"Error in pubmed_search: {e}")
        return "Sorry, I encountered an error while searching PubMed. Please try again."


@tool
def wikipedia_search(query: str) -> str:
    """
    Searches Wikipedia for general information on medical topics, conditions, drugs, or concepts.
    Useful for broader understanding or definitions.
    The input should be the medical topic or term to search for.
    """
    st.info(f"🛠️ Performing Wikipedia Search for: {query}")
    try:
        api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=2000)
        wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
        results = wiki_tool.invoke(query)

        if not results or "No good Wikipedia Search Result was found" in results:
            return "I couldn't find a relevant Wikipedia page for your query."

        prompt_template = ChatPromptTemplate.from_template(
            "You are an expert at summarizing Wikipedia articles for medical context. "
            "Based on the following Wikipedia content, answer the user's query: '{user_query}'. "
            "Provide a concise summary of the key information, in no more than 150 words. "
            "Highlight definitions, causes, and general overview if applicable. "
            "Remind the user that Wikipedia is a general knowledge source and not a substitute for professional medical advice.\n\n"
            "Wikipedia Content:\n{wikipedia_content}"
        )
        chain = prompt_template | model | StrOutputParser()
        return chain.invoke({"user_query": query, "wikipedia_content": results})
    except Exception as e:
        print(f"Error in wikipedia_search: {e}")
        return (
            "Sorry, I encountered an error while searching Wikipedia. Please try again."
        )


tools = [brave_search, pubmed_search, wikipedia_search]


# --- Tool Calling Logic ---
# This function processes tool calls from the LLM.
# If the LLM makes multiple tool_calls in one response, this will try to chain them if 'input' is a common arg.
def tool_chain_executor(model_output_with_tool_calls):
    if (
        not hasattr(model_output_with_tool_calls, "tool_calls")
        or not model_output_with_tool_calls.tool_calls
    ):
        # If no tool calls, it might be a direct answer from the LLM (if the main chain is structured to allow this)
        # However, with .bind_tools, it usually forces a tool call if it can.
        # If it's just a string response, this function might not even be called by Langchain's default tool binding.
        # This path needs careful testing based on how the LLM is expected to behave.
        # For now, we assume if this function is called, there are tool_calls.
        return "No tool was called by the model."

    tool_map = {tool.name: tool for tool in tools}
    # For simplicity, we'll process the first valid tool call.
    # The original script's chaining logic (if out: tool_call['args']['input'] = out)
    # implies a more complex scenario where the LLM plans sequential tools.
    # LangChain's agent executors usually handle multi-step tool use more explicitly.
    # Here, we'll just execute the first one found.
    # If you need sequential tool execution within one turn based on LLM output,
    # the original logic would need to be carefully managed.

    response_from_tool = ""
    for tool_call in model_output_with_tool_calls.tool_calls:
        chosen_tool_name = tool_call["name"]
        if chosen_tool_name in tool_map:
            chosen_tool = tool_map[chosen_tool_name]
            tool_input_args = tool_call["args"]
            try:
                # Langchain tools expect a single string or a dict.
                # We assume the LLM provides args in the correct format.
                # If the tool expects a single 'query' or 'input' string:
                if len(tool_input_args) == 1 and (
                    "query" in tool_input_args or "input" in tool_input_args
                ):
                    input_val = tool_input_args.get("query") or tool_input_args.get(
                        "input"
                    )
                    response_from_tool = chosen_tool.invoke(input_val)
                else:  # Tool might expect a dictionary of arguments
                    response_from_tool = chosen_tool.invoke(tool_input_args)
                break  # Process first valid tool call and exit loop
            except Exception as e:
                print(f"Error invoking tool {chosen_tool_name}: {e}")
                response_from_tool = f"Error using tool {chosen_tool_name}."
                break
        else:
            response_from_tool = f"Error: Tool '{chosen_tool_name}' not found."
            break  # Unknown tool

    return (
        response_from_tool
        if response_from_tool
        else "A tool was called, but no response was generated."
    )


# --- Main Chain Setup ---
# Bind tools to the model. The model will decide which tool to call (if any).
# If the model decides to use a tool, its output will include 'tool_calls'.
# If it decides to answer directly, it will output the message content.
model_with_tools = model.bind_tools(tools)

# The main chain:
# 1. model_with_tools: LLM decides if to use a tool or respond directly.
#    - If tool: output contains tool_calls.
#    - If direct: output is AIMessage content.
# 2. tool_chain_executor: If tool_calls exist, execute them.
# 3. StrOutputParser: Ensures the final output is a string.

# We need a way to conditionally route to tool_chain_executor or pass direct LLM response.
# Langchain's LCEL offers ways for this (e.g., RunnableBranch).
# For simplicity, let's adjust the chain slightly. The LLM bound with tools will produce an AIMessage.
# This AIMessage may or may not have tool_calls.

# If AIMessage has tool_calls, tool_chain_executor runs them and its string output is final.
# If AIMessage has no tool_calls (direct LLM answer), its `content` is the final string.


def get_final_response(model_output):
    if model_output.tool_calls:
        return tool_chain_executor(model_output)  # This will return a string
    return model_output.content  # This is already a string


chain = model_with_tools | get_final_response

# --- Streamlit UI ---
st.set_page_config(page_title="Medical Chatbot Deluxe", layout="wide")
st.title("🩺 Medical Chatbot Deluxe ⚕️")

# Initialize chat history
msgs = StreamlitChatMessageHistory(key="langchain_medical_messages")

if len(msgs.messages) == 0:
    initial_message = (
        "Hello! I am your AI Medical Information Assistant. I can help you search for medical topics using "
        "BraveSearch, PubMed, and Wikipedia.\n\n"
        "**Medical Disclaimer:**\n"
        "I am an AI assistant and **cannot provide medical advice.** "
        "The information I provide is for general informational and educational purposes only, and "
        "**does not constitute medical advice, diagnosis, or treatment.** "
        "Always seek the advice of your physician or other qualified health provider with any "
        "questions you may have regarding a medical condition. Never disregard professional "
        "medical advice or delay in seeking it because of something you have read or heard from me."
    )
    msgs.add_ai_message(initial_message)

# Render chat history
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# React to user input
if user_input := st.chat_input("Ask a medical question..."):
    st.chat_message("user").write(user_input)
    msgs.add_user_message(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking and researching..."):
            try:
                response = chain.invoke(user_input)  # Pass as dict for LLM input
                # Ensure response is a string, sometimes tools might return complex objects if not parsed.
                if not isinstance(response, str):
                    response = str(response)

            except Exception as e:
                print(f"Error in main chain invocation: {e}")
                response = (
                    "I'm sorry, I encountered an unexpected error. Please try again."
                )
                st.error(response)  # Show error in UI as well

        st.write(response)
        msgs.add_ai_message(response)
