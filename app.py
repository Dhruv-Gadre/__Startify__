import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Prevent Mac fork deadlock warnings

from flask import Flask, request, jsonify
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- CSV files ---
FUNDING_FILE = 'startup_funding.csv'
FAILURE_FILE = 'startup_failure_prediction.csv'


# --- Load and merge data ---
def load_data(funding_path, failure_path):
    try:
        df_funding = pd.read_csv(funding_path)
    except FileNotFoundError:
        print(f"Error: {funding_path} not found.")
        return None

    # Rename and create synthetic description if missing
    df_funding.rename(columns={
        'Company': 'Company',
        'Industry': 'Industries',
        'Funding_Amount_USD': 'Funding Amount in $',
        'City': 'City'
    }, inplace=True)

    if 'Description' not in df_funding.columns:
        df_funding['Description'] = (
            df_funding['Industries'].astype(str)
            + " startup based in "
            + df_funding['City'].fillna('Unknown').astype(str)
        )

    df_funding['Funding Amount in $'] = pd.to_numeric(df_funding['Funding Amount in $'], errors='coerce').fillna(0)

    try:
        df_failure = pd.read_csv(failure_path)
    except FileNotFoundError:
        print(f"Error: {failure_path} not found.")
        return None

    df_failure.rename(columns={
        'Startup_Name': 'Company',
        'Industry': 'Industries',
        'Funding_Amount': 'Funding Amount in $',
        'Startup_Status': 'Failure_Status'
    }, inplace=True)

    if 'Description' not in df_failure.columns:
        df_failure['Description'] = (
            df_failure['Industries'].astype(str)
            + " startup (Age: "
            + df_failure['Startup_Age'].fillna('N/A').astype(str)
            + ")"
        )

    # Merge both datasets
    df = pd.merge(
        df_funding,
        df_failure[['Company', 'Failure_Status', 'Description']],
        on='Company',
        how='outer',
        suffixes=('_funding', '_failure')
    )

    # Combine description columns safely
    df['Description_combined'] = (
        df.get('Description_funding', pd.Series("", index=df.index)).fillna('')
        + " "
        + df.get('Description_failure', pd.Series("", index=df.index)).fillna('')
    )

    # Assign unique IDs for FAISS
    df['id'] = df.index
    return df


# --- Initialize model and FAISS ---
df = load_data(FUNDING_FILE, FAILURE_FILE)
if df is None:
    print("❌ Error loading CSVs. Exiting.")
    exit(1)

print("🚀 Loading SentenceTransformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

descriptions = df['Description_combined'].tolist()
embeddings = model.encode(descriptions, show_progress_bar=True)

d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index_map = faiss.IndexIDMap(index)
index_map.add_with_ids(embeddings.astype('float32'), df['id'].values.astype('int64'))

print("✅ Flask model and FAISS index ready.")


# --- Flask API route ---
@app.route('/evaluate', methods=['POST', 'GET'])
def evaluate_startup():
    if request.method == 'GET':
        return jsonify({"message": "Use POST with JSON body to evaluate your startup idea."})

    try:
        data = request.get_json(force=True)
        user_idea = data.get('description', '')
    except Exception as e:
        return jsonify({'error': 'Invalid JSON body', 'details': str(e)}), 400

    if not user_idea:
        return jsonify({'error': 'Description is required'}), 400

    try:
        query_vector = model.encode([user_idea]).astype('float32')
        k = 5
        D, I = index_map.search(query_vector, k)
        matched_ids = I[0]

        results_df = df[df['id'].isin(matched_ids)].set_index('id').loc[matched_ids].reset_index()

        funded_count = (results_df['Funding Amount in $'] > 0).sum()
        avg_funding = results_df[results_df['Funding Amount in $'] > 0]['Funding Amount in $'].mean()
        avg_funding = float(avg_funding) if not np.isnan(avg_funding) else 0.0

        results = []
        for i, row in results_df.iterrows():
            results.append({
                'rank': i + 1,
                'company': row.get('Company', 'N/A'),
                'industry': row.get('Industries_funding', row.get('Industries_failure', 'N/A')),
                'description': row.get('Description_combined', ''),
                'funding': float(row.get('Funding Amount in $', 0)),
                'city': row.get('City', 'N/A'),
                'failure_status': row.get('Failure_Status', 'N/A')
            })

        return jsonify({
            'matches': results,
            'funded_count': int(funded_count),
            'average_funding': round(avg_funding, 2)
        })

    except Exception as e:
        return jsonify({'error': 'Server crashed during processing', 'details': str(e)}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False, threaded=False)
