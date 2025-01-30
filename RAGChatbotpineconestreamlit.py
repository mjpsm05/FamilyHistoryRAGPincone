import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import ollama

# Initialize Pinecone Client
PINECONE_API_KEY = st.secrets["pinecone"]["api_key"] # Store securely
PINECONE_ENV = "us-east1-gcp"
INDEX_NAME = "familyhistoryrag"

pc = Pinecone(api_key=PINECONE_API_KEY)  # ✅ Corrected Pinecone initialization

# Check if the index exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # Ensure this matches your embedding model output
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",  
            region="us-east-1"
        )
    )

# Connect to the index
index = pc.Index(INDEX_NAME)  # ✅ Use pc.Index() instead of pinecone.Index()

# Load Sentence Transformer model for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Streamlit UI
st.title("Kwame Adjovi's Family History RAG Chatbot")

# User Query Input
user_query = st.text_input("Ask something:", placeholder="Enter your question...")

if user_query:
    # Convert Query to Embedding
    query_embedding = embedder.encode(user_query).tolist()

    # Retrieve Top Documents from Pinecone
    results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

    if results and results.get("matches"):
        # Extract metadata (context) from retrieved documents
        context = "\n\n".join([match["metadata"].get("text", "") for match in results["matches"]])

        # 🔹 Remove displaying the retrieved context
        # st.subheader("Retrieved Context:")
        # st.write(context)  ✅ Removed these lines to hide metadata

        # Format prompt for Ollama
        full_prompt = f"""
        You are a helpful assistant. Use the following retrieved documents to answer the user's question.

        Context:
        {context}

        Question:
        {user_query}
        """

        # Generate response using Ollama
        response = ollama.chat(model='llama3.2:latest', messages=[{"role": "user", "content": full_prompt}])

        # Display the AI Response
        st.subheader("AI Response:")
        st.write(response["message"]["content"])
    else:
        st.write("No relevant documents found in Pinecone.")
