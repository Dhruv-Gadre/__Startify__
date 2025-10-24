import requests
import json
import streamlit as st

# Define the Ollama API endpoint
OLLAMA_ENDPOINT = "http://127.0.0.1:11434/api/generate"

# IMPORTANT: Specify the Ollama model you have downloaded
# (e.g., "llama3", "mistral", "phi3")
MODEL_NAME = "llama3.1:8b" 

@st.cache_data(show_spinner=False)
def is_ollama_running():
    """Checks if the Ollama server is running."""
    try:
        response = requests.get("http://127.0.0.1:11434")
        return response.status_code == 200
    except requests.ConnectionError:
        return False

def get_similarity_analysis(user_idea, top_match_description, top_match_company):
    """
    Asks the local LLM to analyze the similarity between two startup ideas.
    """

    prompt = f"""
    You are a Venture Capital (VC) analyst. Your task is to compare two startup descriptions for similarity.
    
    Startup Idea 1 (User): "{user_idea}"
    Startup Idea 2 (Existing): "{top_match_description}" (Company: {top_match_company})

    Analyze the similarity and determine if Idea 1 is a direct "copy" of Idea 2.
    Provide your analysis in the following JSON format:
    
    {{
      "similarity_score": <a number between 0 and 100>,
      "is_copy": <true or false>,
      "reasoning": "<Your 1-2 sentence explanation>"
    }}
    """
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,  # We want the full response at once
        "format": "json"  # Ask Ollama to *only* output JSON
    }
    
    try:
        # Make the API call
        response = requests.post(OLLAMA_ENDPOINT, json=payload)
        response.raise_for_status() # Raise an error for bad responses (4xx, 5xx)
        
        # Parse the JSON response from Ollama
        response_data = response.json()
        
        # The actual content is in the 'response' key,
        # which is *another* JSON string.
        llm_json_output = json.loads(response_data.get("response", "{}"))
        
        return llm_json_output

    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with Ollama: {e}")
        return None
    except json.JSONDecodeError as e:
        st.error(f"Error decoding LLM's JSON response: {e}")
        st.write("LLM output was:", response_data.get("response", "No response"))
        return None