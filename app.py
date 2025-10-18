import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Startup Idea Analyzer",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Startup Idea Analyzer")
st.write("Enter your startup idea to see similar companies and analyze their funding potential.")


DATA_FILE = 'Startups1.csv'


@st.cache_data
def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        df['Description'] = df['Description'].fillna("No description provided")
        df['Funding Amount in $'] = df['Funding Amount in $'].astype(str).str.replace(r'[$,]', '', regex=True)
        df['Funding Amount in $'] = pd.to_numeric(df['Funding Amount in $'], errors='coerce').fillna(0)
        df['id'] = df.index
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{filepath}' was not found.")
        st.error("Please make sure 'startup_data.csv' is in the same folder as 'app.py'")
        return None


@st.cache_resource
def load_model_and_index(_df):
    if _df is None:
        return None, None
        
    # Load a pre-trained model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Get descriptions
    descriptions = _df['Description'].tolist()
    
    # --- Create Embeddings ---
    with st.spinner("Creating vector index... This may take a moment on first run."):
        embeddings = model.encode(descriptions, show_progress_bar=True)
    
    # --- Create FAISS Index ---
    # Get the dimension of the embeddings (e.g., 384 for this model)
    d = embeddings.shape[1] 
    
    # Create an index for L2 (Euclidean) distance
    index = faiss.IndexFlatL2(d)
    
    # Create a map to link FAISS's internal IDs to our DataFrame 'id'
    index_map = faiss.IndexIDMap(index)
    
    # Add vectors to the index with their corresponding 'id'
    index_map.add_with_ids(embeddings.astype('float32'), _df['id'].values.astype('int64'))
    
    st.success("Vector index created successfully!")
    return model, index_map


df = load_data(DATA_FILE)
if df is not None:
    model, index = load_model_and_index(df)
else:
    model, index = None, None


st.header("1. Enter Your Startup Idea")
user_idea = st.text_area("Describe your idea in a few sentences:", height=150, 
                         placeholder="e.g., 'A mobile app that uses AI to detect plant diseases for farmers'")

search_button = st.button("Analyze My Idea")


if search_button and user_idea and df is not None and index is not None:
    with st.spinner("Analyzing your idea and finding matches..."):
        query_vector = model.encode([user_idea]).astype('float32')
        k = 5  
        D, I = index.search(query_vector, k)
        

        matched_ids = I[0] 
        results_df = df[df['id'].isin(matched_ids)]
        

        results_df = results_df.set_index('id').loc[matched_ids].reset_index()

    
    st.header("2. Analysis Results")
    st.subheader(f"Top {k} Most Similar Startups Found:")
    
    for i, row in results_df.iterrows():
        st.markdown("---")
        st.markdown(f"**{i+1}. {row['Company']}** (Industry: {row['Industries']})")
        st.markdown(f"**Description:** {row['Description']}")
        st.markdown(f"**Funding:** ${row['Funding Amount in $']:,.0f}  (Round: {row['Funding Round']})")
        st.markdown(f"**Location:** {row['City']}")

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
    st.warning("Please enter your idea above to get an analysis.")