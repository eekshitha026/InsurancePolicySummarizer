import streamlit as st
import requests
import json
import time
import io
import re
from pypdf import PdfReader
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
from google import genai
from google.generativeai import types
import chromadb
import uuid
import textwrap

# --- CONFIGURATION ---

# IMPORTANT: Replace this with your actual Gemini API Key
API_KEY = "AIzaSyBm_dqgi3vSBtmOYB58RUIcg3EI_pCYvMM"

try:
    # Initialize the Gemini Client
    client = genai.Client(api_key=API_KEY)
except Exception:
    client = None

# RAG Configuration
EMBEDDING_MODEL = "text-embedding-004"
# ChromaDB Client (uses an in-memory instance for simplicity)
CHROMA_CLIENT = chromadb.Client()
COLLECTION_NAME = "policy_document_collection"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100


# --- UTILITY FUNCTIONS (RAG and API Calls) ---

def generate_content_with_retry(prompt_text, system_instruction):
    """Calls the Gemini API with robust safety configuration."""

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.2,
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
            ),
        ]
    )

    if not client:
        return "Error: Gemini client not initialized due to missing API key."

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt_text,
            config=config
        )
        return response.text
    except Exception as e:
        st.error(f"Gemini API Error: {e}")
        return None


def chunk_text(text, chunk_size, chunk_overlap):
    """Splits text into chunks."""
    return textwrap.wrap(text, width=chunk_size, break_long_words=False, replace_whitespace=False)


def embed_and_store_in_chroma(policy_content):
    """Chunks content, embeds, and stores in ChromaDB."""

    st.info("🔄 Chunking document and creating vector embeddings...")
    chunks = chunk_text(policy_content, CHUNK_SIZE, CHUNK_OVERLAP)

    if not chunks:
        st.error("Document is empty or failed to chunk.")
        return None

    try:
        # --- FIX: Use the standard embed_content for a list of contents ---
        st.info(f"Generating embeddings for {len(chunks)} chunks...")
        embed_response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            content=chunks,  # Pass the list of chunks directly
            task_type="RETRIEVAL_DOCUMENT"
        )

        # The embedding list is accessed via the 'embedding' key
        embeddings = embed_response['embedding']
        # -------------------------------------------------------------------

    except Exception as e:
        st.error(f"Embedding API Error: {e}")
        return None

    ids = [str(uuid.uuid4()) for _ in chunks]

    collection = CHROMA_CLIENT.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    try:
        collection.delete(where={})
    except Exception:
        pass

    collection.add(
        embeddings=embeddings,
        documents=chunks,
        ids=ids
    )
    st.success(f"✅ Document split into {len(chunks)} chunks and stored in vector database.")
    return collection


def retrieve_context(query_text, collection, n_results=5):
    """Retrieves relevant context chunks from ChromaDB."""

    query_embedding = client.models.embed_content(
        model=EMBEDDING_MODEL,
        content=query_text
    )['embedding']

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=['documents']
    )

    context_chunks = results['documents'][0]
    context = "\n---\n".join(context_chunks)

    return context


# --- STREAMLIT APPLICATION ---

def policy_decoder_app():
    # Set page config to 'wide' to give more horizontal space, but use default UI components
    st.set_page_config(layout="wide", page_title="Insurance Policy Decoder")

    # --- Header and Description ---
    st.title("📄 Insurance Policy Decoder")
    st.markdown(
        "Use **Retrieval-Augmented Generation (RAG)** with **ChromaDB** to analyze complex policy documents. Supports **OCR** fallback for scanned PDFs.")

    st.markdown("---")

    # --- API Key Check ---
    if not API_KEY or API_KEY == "YOUR_ACTUAL_GEMINI_API_KEY_HERE":
        st.error("🚨 **API Key not set.** Please configure your `API_KEY` in the code.")
        st.stop()
    if not client:
        st.stop()

        # --- File Uploader and Instructions ---

    st.header("1. Upload Policy Document")

    uploaded_file = st.file_uploader(
        "Upload a Policy Document (PDF or TXT)",
        type=["pdf", "txt"],
        help="Supports OCR fallback for scanned PDFs."
    )

    st.markdown("---")

    # --- Analysis & Chat Flow ---

    st.header("2. Policy Inquiry Chat")

    if uploaded_file is not None:
        try:
            policy_content = ""
            file_extension = uploaded_file.name.split('.')[-1].lower()

            # --- Text Extraction Logic (with OCR Fallback) ---
            with st.spinner(f"Extracting text from {uploaded_file.name}..."):
                if file_extension == 'pdf':
                    uploaded_file.seek(0)
                    pdf_reader = PdfReader(uploaded_file)
                    for page in pdf_reader.pages:
                        policy_content += page.extract_text() or ""

                    if not policy_content.strip() or len(policy_content.strip()) < 100:
                        st.warning("⚠️ **Low Text Found:** Document may be scanned. Falling back to OCR...")
                        uploaded_file.seek(0)
                        pdf_bytes = uploaded_file.read()
                        pages = convert_from_bytes(pdf_bytes)
                        ocr_text = ""
                        for page_image in pages:
                            ocr_text += pytesseract.image_to_string(page_image) + "\n"
                        policy_content = ocr_text
                else:
                    uploaded_file.seek(0)
                    policy_content = uploaded_file.read().decode('latin-1')

                if not policy_content.strip():
                    st.error("❌ **Extraction Failure:** Could not extract text from document.")
                    return

            # --- ChromaDB RAG Integration (Embed and Store) ---
            if "policy_content_id" not in st.session_state or st.session_state.policy_content_id != uploaded_file.name:
                st.session_state.policy_content_id = uploaded_file.name
                st.session_state.messages = []
                st.session_state.chroma_collection = embed_and_store_in_chroma(policy_content)

            # --- Chat UI Flow ---
            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Initial summary on first load
            if not st.session_state.messages and st.session_state.get('chroma_collection'):
                initial_prompt = "Please provide a high-level summary of the main coverage, deductibles, and exclusions found in the uploaded policy document."

                initial_context = retrieve_context(initial_prompt, st.session_state.chroma_collection, n_results=10)
                rag_prompt = f"CONTEXT: {initial_context}\n\nQUERY: {initial_prompt}"

                with st.chat_message("assistant"):
                    with st.spinner("Analyzing policy with RAG and generating summary..."):
                        response_text = generate_content_with_retry(rag_prompt,
                                                                    "You are an expert insurance policy analyst. Answer *ONLY* based on the CONTEXT provided.")
                        if response_text:
                            st.session_state.messages.append({"role": "assistant", "content": response_text})
                            st.markdown(response_text)

            # Handle user input
            if prompt := st.chat_input("Ask a question about the policy (e.g., 'What is the liability limit?'):"):

                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking (RAG Retrieval)..."):
                        # RAG STEP 1: Retrieve context
                        retrieved_context = retrieve_context(prompt, st.session_state.chroma_collection, n_results=5)

                        # RAG STEP 2: Construct RAG prompt
                        final_rag_prompt = (
                            f"CONTEXT FROM POLICY: {retrieved_context}\n\n"
                            f"USER QUERY: {prompt}"
                        )

                        rag_system_instruction = (
                            "You are an expert insurance policy analyst. Your role is to accurately and concisely "
                            "answer the user's question. You **MUST** answer based *ONLY* on the CONTEXT FROM POLICY "
                            "provided below. Do not use outside knowledge."
                        )

                        response_text = generate_content_with_retry(final_rag_prompt, rag_system_instruction)

                        if response_text:
                            st.session_state.messages.append({"role": "assistant", "content": response_text})
                            st.markdown(response_text)

        except Exception as e:
            st.error(
                f"❌ **RAG/Processing Error:** An unexpected error occurred: {e}. Check Tesseract, Poppler, and API Key.")
            st.session_state.pop("chroma_collection", None)
            st.session_state.pop("policy_content_id", None)
    else:
        st.info("Upload a policy document above to begin the AI analysis.")


if __name__ == '__main__':
    policy_decoder_app()