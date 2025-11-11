# ============================================================
# SHL Assessment Recommendation System
# Streamlit App with 3 Modes:
# SentenceTransformer (Local), Gemini LLM (Cloud), Hugging Face (Offline)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer, AutoModel
import torch

# ----------------------------
# CONFIGURATION
# ----------------------------
st.set_page_config(page_title="SHL Assessment Recommendation", layout="wide")
st.title("🧠 SHL Assessment Recommendation System")

st.markdown("""
Select a **recommendation mode** below and enter a job description or query.  
This demo supports:
- 🧩 **SentenceTransformer (Local model)**
- ☁️ **Gemini LLM (Google Cloud API)**
- 💫 **Hugging Face Transformers (Offline)**
""")

# ----------------------------
# LOAD PRODUCT CATALOG
# ----------------------------
@st.cache_data
def load_catalog():
    catalog = pd.read_csv("products.csv")
    catalog = catalog.dropna(subset=["AssessmentName", "URL"], how="any").reset_index(drop=True)

    # Create combined text field for better matching
    catalog["combined_text"] = (
        catalog["AssessmentName"].fillna('') + " " +
        catalog.get("Description", pd.Series("", index=catalog.index)).fillna('') + " " +
        catalog.get("Category", pd.Series("", index=catalog.index)).fillna('')
    )
    return catalog

catalog_df = load_catalog()

# ----------------------------
# INITIALIZE MODELS
# ----------------------------
@st.cache_resource
def load_sentence_model():
    return SentenceTransformer(r"C:\Users\sdshu\models\all-MiniLM-L6-v2")

@st.cache_resource
def load_huggingface_model():
    tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\sdshu\models\all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained(r"C:\Users\sdshu\models\all-MiniLM-L6-v2")
    return tokenizer, model

st_model = load_sentence_model()
hf_tokenizer, hf_model = load_huggingface_model()

# ----------------------------
# GEMINI API CONFIGURATION
# ----------------------------
if "GEMINI_API_KEY" not in st.secrets:
    st.warning("⚠️ Gemini API key missing. Gemini mode will be disabled.")
    GEMINI_AVAILABLE = False
else:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    GEMINI_AVAILABLE = True

# ----------------------------
# EMBEDDING FUNCTIONS
# ----------------------------
def get_st_embeddings(texts):
    return st_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

def get_gemini_embeddings(texts):
    if isinstance(texts, str):
        texts = [texts]
    model = "models/embedding-001"
    embeddings = []
    for t in texts:
        result = genai.embed_content(model=model, content=t)
        if "embedding" in result:
            embeddings.append(result["embedding"])
    return np.array(embeddings)

def get_huggingface_embeddings(texts):
    inputs = hf_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = hf_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()

# ----------------------------
# RECOMMENDATION FUNCTIONS
# ----------------------------
def recommend_sentence_transformer(query, catalog_df, top_k=5):
    if not query.strip():
        st.warning("Please enter a valid query.")
        return []

    try:
        valid_texts = catalog_df["combined_text"].tolist()
        query_emb = st_model.encode([query], convert_to_tensor=True)
        catalog_emb = st_model.encode(valid_texts, convert_to_tensor=True)

        scores = util.cos_sim(query_emb, catalog_emb)[0].cpu().numpy()
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = catalog_df.iloc[top_idx][["AssessmentName", "URL"]].copy()
        results["Score"] = scores[top_idx]
        return results
    except Exception as e:
        st.error(f"Error while generating SentenceTransformer recommendations: {e}")
        return []

def recommend_gemini(query, catalog_df, top_k=5):
    try:
        query_emb = get_gemini_embeddings(query)
        catalog_embs = np.vstack([get_gemini_embeddings(name) for name in catalog_df["AssessmentName"]])
        query_emb = normalize(query_emb.reshape(1, -1))
        catalog_embs = normalize(catalog_embs)
        scores = np.dot(catalog_embs, query_emb.T).ravel()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return catalog_df.iloc[top_idx][["AssessmentName", "URL"]].assign(Score=scores[top_idx])
    except Exception as e:
        st.error(f"Gemini API Error: {e}")
        return []

def recommend_huggingface(query, catalog_df, top_k=5):
    try:
        valid_texts = catalog_df["combined_text"].tolist()
        query_emb = get_huggingface_embeddings([query])
        catalog_embs = get_huggingface_embeddings(valid_texts)
        scores = np.dot(catalog_embs, query_emb.T).ravel()
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = catalog_df.iloc[top_idx][["AssessmentName", "URL"]].copy()
        results["Score"] = scores[top_idx]
        return results
    except Exception as e:
        st.error(f"Error while generating Hugging Face recommendations: {e}")
        return []

# ----------------------------
# STREAMLIT INTERFACE
# ----------------------------
available_modes = ["SentenceTransformer", "Hugging Face Embeddings"]
if GEMINI_AVAILABLE:
    available_modes.insert(1, "Gemini LLM")

mode = st.radio("Select Recommendation Mode", available_modes)
query = st.text_area(
    "✍️ Enter Job Description or Query",
    placeholder="e.g. Looking for a Java developer who collaborates well with teams..."
)
top_k = st.slider("🔢 Number of recommendations", min_value=1, max_value=10, value=5, step=1)


if st.button("🚀 Recommend Assessments"):
    if not query.strip():
        st.warning("Please enter a valid query or JD text.")
    else:
        with st.spinner(f"Generating recommendations using {mode}..."):
            if mode == "SentenceTransformer":
                results = recommend_sentence_transformer(query, catalog_df, top_k=top_k)
            elif mode == "Gemini LLM":
                results = recommend_gemini(query, catalog_df, top_k=top_k)
            else:
                results = recommend_huggingface(query, catalog_df, top_k=top_k)

        if len(results) > 0:
            st.success(f"Top {len(results)} recommendations ({mode} mode):")
            st.dataframe(results, use_container_width=True)

            csv = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "💾 Download CSV",
                data=csv,
                file_name=f"recommendations_{mode}.csv",
                mime="text/csv"
            )

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("---")
st.caption("SHL GenAI Internship Assignment | Multi-Model Recommendation System (SentenceTransformer, Gemini, Hugging Face)")
