# StudyMate: AI-Powered PDF Q&A System using Gemini 2.0 Flash and Hugging Face

# Step 1: Install required packages (do this in your terminal)
# pip install streamlit PyMuPDF faiss-cpu sentence-transformers google-generativeai

import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# UI: Authentication step
st.title("🔐 Authenticate with Gemini API Key")
api_key = st.text_input("Enter your Gemini 2.0 Flash API Key", type="password")

if api_key:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

        st.success("✅ Authentication successful! You can now upload PDFs and ask questions.")

        # Main UI starts after authentication
        st.title("📘 StudyMate: Multi-PDF Q&A Assistant")
        pdfs = st.file_uploader("Upload one or more study PDFs", type="pdf", accept_multiple_files=True)

        def extract_text_from_pdf(pdf_file):
            text = ""
            with fitz.open(stream=pdf_file, filetype="pdf") as doc:
                for page in doc:
                    text += page.get_text()
            return text

        if pdfs and "chunks" not in st.session_state:
            with st.spinner("Extracting and embedding content from PDFs..."):
                full_text = ""
                for pdf in pdfs:
                    file_bytes = pdf.read()
                    full_text += extract_text_from_pdf(file_bytes)

                chunks = [" ".join(full_text.split()[i:i+500]) for i in range(0, len(full_text.split()), 500)]
                chunk_embeddings = embed_model.encode(chunks)
                st.session_state["chunks"] = chunks
                st.session_state["chunk_embeddings"] = chunk_embeddings

        if "chunks" in st.session_state and "chunk_embeddings" in st.session_state:
            question = st.text_input("Ask a question based on the uploaded PDFs")

            if question:
                with st.spinner("Searching and generating answer..."):
                    q_embedding = embed_model.encode([question])
                    index = faiss.IndexFlatL2(len(q_embedding[0]))
                    index.add(np.array(st.session_state["chunk_embeddings"]))
                    _, I = index.search(np.array(q_embedding), 3)
                    top_chunks = [st.session_state["chunks"][i] for i in I[0]]
                    context = "\n".join(top_chunks)
                    prompt = f"""
Use the following context to answer the question:

Context:
{context}

Question:
{question}
"""
                    response = model.generate_content(prompt)
                    answer = response.text

                st.markdown("### 📖 Answer:")
                st.write(answer)
                st.markdown("---")
                with st.expander("🔍 Context Used"):
                    st.write(context)

    except Exception as e:
        st.error(f"❌ Authentication failed: {str(e)}")
else:
    st.warning("⚠️ Please enter your Gemini API key to begin.")
