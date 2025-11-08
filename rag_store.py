from google import genai
from google.genai.errors import APIError
import os
import json  # Added for robust embedding value extraction


# We will use the RAG service to find clauses relevant to key terms like "Exclusion"
def extract_single_embedding(response):
    """
    Safely extracts the single embedding vector from the response object.
    """
    try:
        # Try the most likely properties for the list of vectors
        if hasattr(response, 'embedding') and response.embedding is not None:
            return response.embedding[0]
        if hasattr(response, 'values') and response.values is not None:
            return response.values[0]

        # Try iterating over nested embedding objects
        if hasattr(response, 'embeddings') and response.embeddings is not None:
            first_embedding = response.embeddings[0]
            if hasattr(first_embedding, 'value'):
                return first_embedding.value
            if hasattr(first_embedding, 'embedding'):
                return first_embedding.embedding

        raise AttributeError("Could not find the single embedding vector in the API response object.")

    except Exception as e:
        raise AttributeError(f"Failed to extract query embedding due to response structure: {e}")


def query_rag_store(collection, query_text: str, k: int = 5) -> str:
    """
    Performs a similarity search against the indexed policy clauses.

    Returns:
        str: A concatenated string of the top-k relevant clauses, or an error message.
    """

    # Check if the collection is a DummyCollection (count == 0).
    # This also handles cases where clause extraction failed (empty clauses list).
    if collection.count() == 0:
        # Perform a dummy query to retrieve the failure reason stored in the DummyCollection
        dummy_results = collection.query(query_texts=["check_reason"], n_results=1, include=['documents'])
        failure_reason = dummy_results.get('documents', [[]])[0][0]
        return failure_reason  # This returns the explicit error message from chroma_helper.py

    try:
        client_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        # Embed the query
        query_embedding_response = client_gemini.models.embed_content(
            model='text-embedding-004',
            contents=[query_text]
        )

        # Use the robust single extraction function
        query_embedding = extract_single_embedding(query_embedding_response)

        # Query the ChromaDB collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=['documents']
        )

        # Concatenate the relevant documents (clauses)
        relevant_clauses = [doc for sublist in results.get('documents', []) for doc in sublist]

        if not relevant_clauses:
            return "No relevant clauses found in the policy."

        return "\n---\n".join(relevant_clauses)

    except APIError as e:
        print(f"Gemini API Error during RAG query: {e}")
        return "Error querying the policy index due to Gemini API failure."
    except Exception as e:
        print(f"An unexpected error occurred during RAG query: {e}")
        return "An unexpected error occurred during RAG query."