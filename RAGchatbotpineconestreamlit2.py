import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import requests  # For Hugging Face API requests

# Initialize Pinecone Client
PINECONE_API_KEY = st.secrets["pinecone"]["api_key"]
INDEX_NAME = "familyhistoryrag"

pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if the index exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Connect to the index
index = pc.Index(INDEX_NAME)

# Load Sentence Transformer model for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Hugging Face API setup
HF_API_KEY = st.secrets["huggingface"]["api_key_2"]
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
HF_API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# Streamlit UI
st.title("Kwame Adjovi's Family History RAG Chatbot")

# User Query Input
user_query = st.text_input("Ask something:", placeholder="Enter your question...")

if user_query:
    # Convert Query to Embedding
    query_embedding = embedder.encode(user_query).tolist()

    # Retrieve Top Documents from Pinecone
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)

    # Extract relevant content from retrieved documents
    context_list = [match["metadata"].get("text", "") for match in results["matches"] if "text" in match["metadata"]]
    
    if context_list:
        refined_context = " ".join(context_list[:2])  # Use first 2 relevant chunks
    else:
        refined_context = "No relevant information available."

    # Format the full prompt (only for model inference, do not display context to user)
    full_prompt = f"""
    You are a helpful assistant. Based on the context below, answer the user's question.

    Context: {refined_context}

    Question: {user_query}

    
    """

    # Send request to Hugging Face API
    response = requests.post(HF_API_URL, headers=headers, json={"inputs": full_prompt})

    if response.status_code == 200:
        ai_response = response.json()[0]["generated_text"]
        
        # Display only the AI-generated response, not the context
        st.subheader("AI Response:")
        st.write(ai_response)
    else:
        st.write("Error generating response:", response.text)
