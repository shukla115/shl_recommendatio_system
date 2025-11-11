SHL Recommendation System

**PROJECT OVERVIEW**

This project is a Generative AI–powered recommendation system that suggests the most relevant SHL assessments based on a given job description or skill-based query.

The application is built using Streamlit and offers three powerful recommendation modes:
1. **SentenceTransformer (Offline Model)**
   - Uses locally stored embeddings from the `all-MiniLM-L6-v2` model.  
   - Generates semantic representations of both queries and SHL product descriptions.  
   - Provides quick and efficient offline recommendations.

2. **Hugging Face Transformer (Alternate Model)**
   - Employs transformer-based embeddings directly from the Hugging Face library.  
   - Ensures flexibility for experimentation with multiple pretrained models. 

3. **Google Gemini LLM** 
    - For deeper contextual understanding, allowing it to handle complex or ambiguous queries more effectively.

The system enables recruiters or hiring teams to identify the most suitable assessments by analyzing semantic similarity between job requirements and SHL product information.

**HOW IT WORKS**

**Input Processing**: The user provides a job description or skill-based query.

**Embedding Generation**: Text is converted into vector representations using a SentenceTransformer model.

**Similarity Matching**: Cosine similarity is computed between the input and SHL assessment embeddings.

**Recommendation Output**: The system returns the top-N most relevant SHL assessments.

**LLM Enhancement**: Gemini LLM refines context or interprets nuanced requirements.

**KEY FEATURES**

Semantic search using SentenceTransformer embeddings

Optional Gemini LLM integration for advanced context understanding

Efficient ranking and retrieval of SHL assessments

API-ready design for easy integration into existing Human Resource systems

Modular structure for scalability and experimentation

TECH STACK
| Layer | Technology Used |
|-------|------------------|
| **Frontend** | Streamlit |
| **Backend Logic** | Python |
| **AI/Embedding Models** | SentenceTransformer, Hugging Face Transformers |
| **Data Handling** | Pandas, NumPy |
| **Similarity Computation** | Cosine similarity via `sentence-transformers` utilities |


