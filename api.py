# ============================================================
# SHL Assessment Recommendation System - API Backend
# Using FastAPI + SentenceTransformer
# ============================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# INITIALIZATION
# ------------------------------------------------------------
app = FastAPI(title="SHL Assessment Recommendation API")

# Load catalog
catalog_df = pd.read_csv("products.csv")
catalog_df = catalog_df.dropna(subset=["AssessmentName", "URL"], how="any").reset_index(drop=True)

if "combined_text" not in catalog_df.columns:
    catalog_df["combined_text"] = (
        catalog_df["AssessmentName"].fillna('') + " " +
        catalog_df.get("Description", pd.Series("", index=catalog_df.index)).fillna('') + " " +
        catalog_df.get("Category", pd.Series("", index=catalog_df.index)).fillna('')
    )

# Load model
model = SentenceTransformer(r"C:\Users\sdshu\models\all-MiniLM-L6-v2")

# ------------------------------------------------------------
# INPUT SCHEMA
# ------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

# ------------------------------------------------------------
# ENDPOINTS
# ------------------------------------------------------------
@app.get("/")
def home():
    return {"message": "SHL Recommendation API is running. Use POST /recommend"}

@app.post("/recommend")
def recommend(request: QueryRequest):
    query = request.query.strip()
    top_k = request.top_k

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        # Encode query and catalog
        query_emb = model.encode([query], convert_to_tensor=True)
        catalog_emb = model.encode(catalog_df["combined_text"].tolist(), convert_to_tensor=True)

        # Compute cosine similarity
        scores = util.cos_sim(query_emb, catalog_emb)[0].cpu().numpy()
        top_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_idx:
            results.append({
                "AssessmentName": catalog_df.iloc[idx]["AssessmentName"],
                "URL": catalog_df.iloc[idx]["URL"],
                "Score": float(scores[idx])
            })

        return {"query": query, "recommendations": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
