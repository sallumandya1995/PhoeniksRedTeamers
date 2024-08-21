 


import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import base64

load_dotenv()
 
def get_together_models():
    return [
        "meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3-70B-Instruct-Lite",
        "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "google/gemma-2-9b-it",
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "deepseek-ai/deepseek-coder-33b-instruct",
        "meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
         "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
        "meta-llama/Meta-Llama-3-70B-Instruct-Lite",
        "google/gemma-2-27b-it",
        "allenai/OLMo-7B-Instruct",
        "zero-one-ai/Yi-34B-Chat",
        "allenai/OLMo-7B-Twin-2T",
        "allenai/OLMo-7B",
        "Austism/chronos-hermes-13b",
        "cognitivecomputations/dolphin-2.5-mixtral-8x7b",
        "databricks/dbrx-instruct",
        
        "deepseek-ai/deepseek-llm-67b-chat",
        "garage-bAInd/Platypus2-70B-instruct",
        "google/gemma-2b-it",
        "google/gemma-7b-it",
        "Gryphe/MythoMax-L2-13b",
        "lmsys/vicuna-13b-v1.5",
        "lmsys/vicuna-7b-v1.5",
        "codellama/CodeLlama-13b-Instruct-hf",
        "codellama/CodeLlama-34b-Instruct-hf",
        "codellama/CodeLlama-70b-Instruct-hf",
        "codellama/CodeLlama-7b-Instruct-hf",
        "meta-llama/Llama-2-70b-chat-hf",
        "meta-llama/Llama-2-13b-chat-hf",
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-3-8b-chat-hf",
        "meta-llama/Llama-3-70b-chat-hf",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "NousResearch/Nous-Capybara-7B-V1p9",
        "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
        "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        "NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",
        "NousResearch/Nous-Hermes-llama-2-7b",
        "NousResearch/Nous-Hermes-Llama2-13b",
        "NousResearch/Nous-Hermes-2-Yi-34B",
        "openchat/openchat-3.5-1210",
        "Open-Orca/Mistral-7B-OpenOrca",
        "Qwen/Qwen1.5-0.5B-Chat",
        "Qwen/Qwen1.5-1.8B-Chat",
        "Qwen/Qwen1.5-4B-Chat",
        "Qwen/Qwen1.5-7B-Chat",
        "Qwen/Qwen1.5-14B-Chat",
        "Qwen/Qwen1.5-32B-Chat",
        "Qwen/Qwen1.5-72B-Chat",
        "Qwen/Qwen1.5-110B-Chat",
        "Qwen/Qwen2-72B-Instruct",
        "snorkelai/Snorkel-Mistral-PairRM-DPO",
        "Snowflake/snowflake-arctic-instruct",
        "togethercomputer/alpaca-7b",
        "teknium/OpenHermes-2-Mistral-7B",
        "teknium/OpenHermes-2p5-Mistral-7B",
        "togethercomputer/Llama-2-7B-32K-Instruct",
        "togethercomputer/RedPajama-INCITE-Chat-3B-v1",
        "togethercomputer/RedPajama-INCITE-7B-Chat",
        "togethercomputer/StripedHyena-Nous-7B",
        "Undi95/ReMM-SLERP-L2-13B",
        "Undi95/Toppy-M-7B",
        "WizardLM/WizardLM-13B-V1.2",
        "upstage/SOLAR-10.7B-Instruct-v1.0"
    ]


# Function to get Groq chat models
def get_groq_models():
    return [
        # "llama-3.1-405b-reasoning",
        "llama-3.1-70b-versatile",
        "mixtral-8x7b-32768",
        "llama3-groq-70b-8192-tool-use-preview",
        "llama-3.1-8b-instant",
        "llama3-groq-8b-8192-tool-use-preview",
        "llama-guard-3-8b",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "gemma-7b-it",
        "gemma2-9b-it",
        "whisper-large-v3"
    ]
 
def get_openairouter_models():
    return [
        "meta-llama/llama-3.1-8b-instruct:free",
        "nousresearch/hermes-3-llama-3.1-405b",
        "claude-3-5-sonnet",
        "gpt-4-turbo-128k-france",
        "gemini-1.0-pro",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "Llama-3-70B-Instruct",
        "Mixtral-8x7B-Instruct-v0.1",
        "CodeLlama-2",
        "jina-embeddings-v2-base-de",
        "jina-embeddings-v2-base-code",
        "text-embedding-bge-m3",
        "llava-v1.6-34b",
        "llava-v1.6-vicuna-13b",
        "gpt-35-turbo",
        "text-embedding-ada-002",
        "gpt-4-32k-1",
        "gpt-4-32k-canada",
        "gpt-4-32k-france",
        "text-embedding-ada-002-france",
        "mistral-large-32k-france",
        "Llama-3.1-405B-Instruct-US",
        "Mistral-Large-2407",
        "Mistral-Nemo-2407"
    ]

# Function to get OpenAI-like models
def get_openai_like_models():
    return [
        "claude-3-5-sonnet",
        "gpt-4-turbo-128k-france",
        "gemini-1.0-pro",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "Llama-3-70B-Instruct",
        "Mixtral-8x7B-Instruct-v0.1",
        "CodeLlama-2",
        "jina-embeddings-v2-base-de",
        "jina-embeddings-v2-base-code",
        "text-embedding-bge-m3",
        "llava-v1.6-34b",
        "llava-v1.6-vicuna-13b",
        "gpt-35-turbo",
        "text-embedding-ada-002",
        "gpt-4-32k-1",
        "gpt-4-32k-canada",
        "gpt-4-32k-france",
        "text-embedding-ada-002-france",
        "mistral-large-32k-france",
        "Llama-3.1-405B-Instruct-US",
        "Mistral-Large-2407",
        "Mistral-Nemo-2407"
    ]


 
def to_leetspeak(text):
    leet_dict = {
        'a': '4', 'e': '3', 'g': '6', 'i': '1', 'o': '0', 's': '5', 't': '7',
        'A': '4', 'E': '3', 'G': '6', 'I': '1', 'O': '0', 'S': '5', 'T': '7'
    }
    return ''.join(leet_dict.get(char, char) for char in text)

def to_base64(text):
    return base64.b64encode(text.encode()).decode()

def to_binary(text):
    return ' '.join(format(ord(char), '08b') for char in text)

def to_emoji(text):
    emoji_dict = {
        'a': '🅰', 'b': '🅱', 'c': '🅲', 'd': '🅳', 'e': '🅴', 'f': '🅵', 'g': '🅶', 'h': '🅷', 'i': '🅸', 'j': '🅹',
        'k': '🅺', 'l': '🅻', 'm': '🅼', 'n': '🅽', 'o': '🅾', 'p': '🅿', 'q': '🆀', 'r': '🆁', 's': '🆂', 't': '🆃',
        'u': '🆄', 'v': '🆅', 'w': '🆆', 'x': '🆇', 'y': '🆈', 'z': '🆉',
        'A': '🅰', 'B': '🅱', 'C': '🅲', 'D': '🅳', 'E': '🅴', 'F': '🅵', 'G': '🅶', 'H': '🅷', 'I': '🅸', 'J': '🅹',
        'K': '🅺', 'L': '🅻', 'M': '🅼', 'N': '🅽', 'O': '🅾', 'P': '🅿', 'Q': '🆀', 'R': '🆁', 'S': '🆂', 'T': '🆃',
        'U': '🆄', 'V': '🆅', 'W': '🆆', 'X': '🆇', 'Y': '🆈', 'Z': '🆉'
    }
    return ''.join(emoji_dict.get(char, char) for char in text)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

if 'selected_model' not in st.session_state:
    st.session_state.selected_model = ""

if 'selected_service' not in st.session_state:
    st.session_state.selected_service = ""

if 'base_url' not in st.session_state:
    st.session_state.base_url = ""

# Sidebar
st.sidebar.title("Chat Settings")

 
# Service selection
service = st.sidebar.radio("Select a service:", ("Together AI","OpenAI-like", "Groq","Openrouter"))
st.session_state.selected_service = service

# Model selection based on the chosen service
if service == "OpenAI-like":
    openai_like_models = get_openai_like_models()
    selected_model = st.sidebar.selectbox("Select an OpenAI-like model:", openai_like_models)
    base_url = st.sidebar.text_input("Enter the base URL for the OpenAI-like API:",type="password",value=os.getenv('API_BASE'))
    api_key = st.sidebar.text_input("Enter your API Key:", type="password",value=os.getenv('API_KEY'))
    if api_key:
        st.session_state.api_key = api_key
    if base_url:
        st.session_state.base_url = base_url

elif service == "Openrouter":
    openai_like_models = get_openairouter_models()
    selected_model = st.sidebar.selectbox("Select an OpenAI-like model:", openai_like_models)
    base_url = st.sidebar.text_input("Enter the base URL for the OpenAI-like API:",type="password",value=os.getenv('API_BASE_OPENROUTER'))
    api_key = st.sidebar.text_input("Enter your API Key:", type="password",value=os.getenv('API_KEY_OPENROUTER'))
    if api_key:
        st.session_state.api_key = api_key
    if base_url:
        st.session_state.base_url = base_url
elif service == "Groq":
    groq_models = get_groq_models()
    api_key = st.sidebar.text_input("Enter your API Key:", type="password",value=os.getenv('GROQ_API_KEY'))
    if api_key:
        st.session_state.api_key = api_key
    selected_model = st.sidebar.selectbox("Select a Groq model:", groq_models)
    base_url = "https://api.groq.com/openai/v1"
else:  # OpenAI-like
    together_models = get_together_models()
    api_key = st.sidebar.text_input("Enter your API Key:", type="password",value=os.getenv('TOGETHER_API_KEY'))
    if api_key:
        st.session_state.api_key = api_key
    selected_model = st.sidebar.selectbox("Select a Together AI model:", together_models)
    base_url = "https://api.together.xyz/v1"

if selected_model:
    st.session_state.selected_model = selected_model

# Main chat interface
st.title("AI Chat Application")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("You:"):
    if not st.session_state.api_key:
        st.error("Please enter an API key.")
    elif not st.session_state.selected_model:
        st.error("Please select a model.")
    elif service == "OpenAI-like" and not st.session_state.base_url:
        st.error("Please enter the base URL for the OpenAI-like API.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate AI response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                if service == "OpenAI-like":
                    client = OpenAI(
                        api_key=st.session_state.api_key,
                        base_url=st.session_state.base_url + '/v2',
                    )
                else:
                    client = OpenAI(api_key=st.session_state.api_key, base_url=base_url)
                
                for response in client.chat.completions.create(
                    model=st.session_state.selected_model,
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    stream=True,
                    max_tokens=1000,
                    temperature=0.7
                ):
                    full_response += (response.choices[0].delta.content or "")
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                full_response = "I apologize, but an error occurred while generating the response."
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

# Clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []

    st.rerun()

 
 

with st.sidebar:

    st.title("Text Conversion")
    input_text = st.text_area("Enter text to convert:")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("To Leetspeak"):
            if input_text:
                converted_text = to_leetspeak(input_text)
                st.text_area("Leetspeak Result:", converted_text, height=100)
                st.code(converted_text, language="text")

        if st.button("To Base64"):
            if input_text:
                converted_text = to_base64(input_text)
                st.text_area("Base64 Result:", converted_text, height=100)
                st.code(converted_text, language="text")

    with col2:
        if st.button("To Binary"):
            if input_text:
                converted_text = to_binary(input_text)
                st.text_area("Binary Result:", converted_text, height=100)
                st.code(converted_text, language="text")

        if st.button("To Emoji"):
            if input_text:
                converted_text = to_emoji(input_text)
                st.text_area("Emoji Result:", converted_text, height=100)
                st.code(converted_text, language="text")


with st.sidebar:
    import pandas as pd
    prompts_csv = pd.read_csv("Prompts.csv")
    st.title("Prompt break")
    st.write("prompt breaker")
    st.write("https://github.com/elder-plinius/L1B3RT45")
    st.dataframe(prompts_csv)



 

 


