import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import datetime
# NEW: Import our Ollama functions
from ollama_client import is_ollama_running, get_similarity_analysis 

# --- Page Configuration ---
st.set_page_config(
    page_title="Startup Idea Analyzer",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Startup Idea Analyzer")
st.write("Enter your startup idea, city, and founding year to see similar companies and analyze your potential.")

# --- File Path ---
DATA_FILE = 'Startups1.csv'

# --- 1. Load Data (Cache to speed up app) ---
@st.cache_data
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
        st.error(f"Error: The file '{filepath}' was not found.")
        st.error(f"Please make sure '{DATA_FILE}' is in the same folder as 'app.py'")
        return None

# --- 2. Load Model & Create Vector Index (Cache to speed up app) ---
@st.cache_resource
def load_model_and_index(_df):
    if _df is None:
        return None, None
    model = SentenceTransformer('all-MiniLM-L6-v2')
    descriptions = _df['Description'].tolist()
    with st.spinner("Creating vector index... This may take a moment on first run."):
        embeddings = model.encode(descriptions, show_progress_bar=True)
    d = embeddings.shape[1] 
    index = faiss.IndexFlatL2(d)
    index_map = faiss.IndexIDMap(index)
    index_map.add_with_ids(embeddings.astype('float32'), _df['id'].values.astype('int64'))
    st.success("Vector index created successfully!")
    return model, index_map

# --- Run the loading functions ---
df = load_data(DATA_FILE)
if df is not None:
    model, index = load_model_and_index(df)
else:
    model, index = None, None

# --- NEW: Check for Ollama ---
if not is_ollama_running():
    st.error("🚨 Ollama server is not running!")
    st.warning("Please start your local Ollama server to use the AI analysis features.")
    st.stop() # Stop the app from running further if Ollama isn't on

# --- 3. User Input Section (Tasks 1 & 3) ---
st.header("1. Tell Us About Your Startup")
user_idea = st.text_area("A. Describe your startup idea:", height=150, 
                         placeholder="e.g., 'A mobile app that uses AI to detect plant diseases for farmers'")
col1, col2 = st.columns(2)
with col1:
    user_city = st.text_input("B. What city are you based in?", placeholder="e.g., New York")
with col2:
    current_year = datetime.date.today().year
    founding_year = st.number_input("C. What is your planned founding year?", 
                                    min_value=current_year - 5, 
                                    max_value=current_year + 5, 
                                    value=current_year, 
                                    step=1,
                                    format="%d")

search_button = st.button("Analyze My Idea")

# --- 4. Analysis and Results Section ---
if search_button and user_idea and user_city and founding_year and df is not None and index is not None:
    
    st.header("2. Analysis Results")
    st.markdown("---")
    
    with st.spinner("Finding similar startups..."):
        # --- A. Find Similar Startups (Core of Task 4) ---
        query_vector = model.encode([user_idea]).astype('float32')
        k = 5  
        D, I = index.search(query_vector, k)
        matched_ids = I[0]
        results_df = df[df['id'].isin(matched_ids)]
        results_df = results_df.set_index('id').loc[matched_ids].reset_index()
        
        # --- B. Classify Industry (Task 2) ---
        industry_mode = results_df['Industries'].mode()
        print(industry_mode[0])
        x = (industry_mode[0].split(","))
        print(" ".join(x[:2]))
        predicted_industry = industry_mode[0] if not industry_mode.empty else "Could not determine"

    # --- C. Display Profile (Tasks 1, 2, 3) ---
    st.subheader("Your Startup Profile")
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Industry", predicted_industry)
    col2.metric("Your City", user_city)
    col3.metric("Founding Year", founding_year)

    # --- D. NEW: LLM Similarity Analysis (The new Task 4) ---
    st.subheader("🤖 AI 'Copycat' Analysis")
    with st.spinner("Asking local AI to analyze similarity..."):
        top_match = results_df.iloc[0] # Get the #1 match
        
        # Call our new Ollama function
        analysis = get_similarity_analysis(user_idea, 
                                           top_match['Description'], 
                                           top_match['Company'])
        
        if analysis:
            score = analysis.get("similarity_score", 0)
            is_copy = analysis.get("is_copy", False)
            reasoning = analysis.get("reasoning", "No reasoning provided.")
            
            # Display the score in a progress bar
            st.write(f"**Similarity to closest match ({top_match['Company']}):**")
            st.progress(score / 100, text=f"{score}% Similar")
            
            if is_copy:
                st.error(f"**AI Verdict: High 'Copycat' Risk.**")
                st.error(f"**Reasoning:** {reasoning}")
            else:
                st.success(f"**AI Verdict: Looks Unique!**")
                st.success(f"**Reasoning:** {reasoning}")
        else:
            st.warning("Could not get AI analysis. Proceeding without it.")

    # --- E. Display Similar Startups (Existing) ---
    st.subheader(f"Top {k} Most Similar Startups Found:")
    for i, row in results_df.iterrows():
        st.markdown("---")
        st.markdown(f"**{i+1}. {row['Company']}** (Industry: {row['Industries']})")
        st.markdown(f"**Description:** {row['Description']}")
        st.markdown(f"**Funding:** ${row['Funding Amount in $']:,.0f}  (Round: {row['Funding Round']})")
        st.markdown(f"**Location:** {row['City']}")

    # --- F. Display Funding Analysis (Existing) ---
    st.subheader("3. Initial Funding Analysis")
    funded_count = (results_df['Funding Amount in $'] > 0).sum()
    avg_funding = results_df[results_df['Funding Amount in $'] > 0]['Funding Amount in $'].mean()
    col1, col2 = st.columns(2)
    col1.metric("Funding Rate (in sample)", 
                f"{funded_count} out of {k}", 
                "startups were funded")
    if funded_count > 0:
        col2.metric(f"Average Funding (for funded)", 
                    f"${avg_funding:,.0f}")
    else:
        col2.info("No funding data found for these specific matches.")

elif search_button:
    st.warning("Please fill out all fields: Idea Description, City, and Year.")