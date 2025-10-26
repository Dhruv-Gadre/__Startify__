import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import datetime
import logging


try:
    from model_client import predict_funding
except ImportError:
    print("Error: 'model_clients.py' not found.")
    def predict_funding(*args, **kwargs):
        return {"predicted_amount": None, "confidence": 0.0, "error": "model_clients.py not found"}
    
# Import your existing Ollama functions
# Make sure ollama_client.py is in the same folder
try:
    from ollama_client import is_ollama_running, get_similarity_analysis
except ImportError:
    print("Error: 'ollama_client.py' not found.")
    print("Please make sure the file is in the same directory.")
    # Define dummy functions if not found, so the app can at least start
    def is_ollama_running(): return False
    def get_similarity_analysis(*args): return None

# --- Globals for models and data ---
df = None
model = None
index = None
DATA_FILE = 'Startups1.csv'

# --- 1. Pydantic Model for Input Validation ---
# This defines what data your API endpoint expects
class StartupInput(BaseModel):
    user_idea: str
    user_city: str
    founding_year: int

# --- 2. Data/Model Loading Functions (from your Streamlit app) ---
# We removed all `st.` calls and replaced them with `logging`
def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        df['Description'] = df['Description'].fillna("No description provided")
        df['Industries'] = df['Industries'].fillna("Not specified")
        df['Funding Amount in $'] = df['Funding Amount in $'].astype(str).str.replace(r'[$,]', '', regex=True)
        df['Funding Amount in $'] = pd.to_numeric(df['Funding Amount in $'], errors='coerce').fillna(0)
        df['id'] = df.index
        return df
    except FileNotFoundError:
        logging.error(f"Error: The file '{filepath}' was not found.")
        logging.error(f"Please make sure '{DATA_FILE}' is in the same folder.")
        return None

def load_model_and_index(_df):
    if _df is None:
        return None, None
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    descriptions = _df['Description'].tolist()
    print("Creating vector index... This may take a moment.")
    embeddings = model.encode(descriptions, show_progress_bar=True)
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index_map = faiss.IndexIDMap(index)
    index_map.add_with_ids(embeddings.astype('float32'), _df['id'].values.astype('int64'))
    print("Vector index created successfully!")
    return model, index_map

# --- 3. FastAPI App & Startup Event ---
app = FastAPI(
    title="Startify AI API",
    description="API for the Startify Startup Idea Analyzer"
)

# Add CORS middleware to allow your HTML file to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """
    This function runs once when the server starts.
    It loads the data and models into memory.
    """
    global df, model, index
    print("Server starting up... Loading data...")
    df = load_data(DATA_FILE)
    if df is not None:
        print("Data loaded. Loading model and index...")
        model, index = load_model_and_index(df)
        print("--- Server is ready to accept requests ---")
    else:
        print("--- FATAL ERROR: Could not load data. Server started but /analyze will fail. ---")

# --- 4. The API Endpoint ---
@app.post("/analyze")
async def analyze_startup(input_data: StartupInput):
    """
    This is the main endpoint that performs all the analysis.
    """
    # --- A. Check if server is ready ---
    if df is None or model is None or index is None:
        raise HTTPException(status_code=503, detail="Server is not ready. Model or data not loaded.")

    # --- B. Check if Ollama is running ---
    if not is_ollama_running():
        raise HTTPException(status_code=503, detail="Ollama server is not running. Please start Ollama to use AI analysis.")
    
    try:
        # --- C. Find Similar Startups ---
        query_vector = model.encode([input_data.user_idea]).astype('float32')
        k = 5
        D, I = index.search(query_vector, k)
        matched_ids = I[0]
        results_df = df[df['id'].isin(matched_ids)]
        results_df = results_df.set_index('id').loc[matched_ids].reset_index()

        # --- D. Classify Industry ---
        industry_mode = results_df['Industries'].mode()
        predicted_industry = industry_mode[0] if not industry_mode.empty else "Could not determine"

        # --- E. LLM Similarity Analysis ---
        top_match = results_df.iloc[0]
        analysis = get_similarity_analysis(
            input_data.user_idea,
            top_match['Description'],
            top_match['Company']
        )
        
        ai_analysis_result = {
            "top_match_name": top_match['Company'],
            "score": 0,
            "is_copy": False,
            "reasoning": "Could not get AI analysis."
        }
        if analysis:
            ai_analysis_result.update({
                "score": analysis.get("similarity_score", 0),
                "is_copy": analysis.get("is_copy", False),
                "reasoning": analysis.get("reasoning", "No reasoning provided.")
            })

        # --- F. Funding Analysis ---
        funded_count = (results_df['Funding Amount in $'] > 0).sum()
        avg_funding = results_df[results_df['Funding Amount in $'] > 0]['Funding Amount in $'].mean()
        
        funding_analysis_result = {
            "rate": f"{funded_count} out of {k}",
            "avg_funding_str": f"${avg_funding:,.0f}" if funded_count > 0 else "N/A"
        }
        
        print("Predicting funding...")
        funding_prediction_result = predict_funding(
            description=input_data.user_idea,
            industry=predicted_industry,  # Use the industry we just predicted
            city=input_data.user_city,
            founding_year=input_data.founding_year
        )
        print(f"Funding prediction result: {funding_prediction_result}")
        # --- (END PLACEHOLDER REPLACEMENT) ---

        # --- G. Format and Return Results ---
        return {
            "profile": {
                "predicted_industry": predicted_industry,
                "user_city": input_data.user_city,
                "founding_year": input_data.founding_year
            },
            "ai_analysis": ai_analysis_result,
            "similar_startups": results_df.to_dict('records'),
            "funding_analysis": funding_analysis_result,
            
            # --- MODIFIED LINE ---
            # "extra_prediction": extra_prediction  <-- DELETE THIS LINE
            "funding_prediction": funding_prediction_result # <-- ADD THIS LINE
            # --- END MODIFIED LINE ---
        }

    except Exception as e:
        logging.error(f"Error during analysis: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

# --- 5. Run the Server ---
if __name__ == "__main__":
    print("Starting FastAPI server on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)