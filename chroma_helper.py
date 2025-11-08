import chromadb
from google import genai
from google.genai.errors import APIError
import os
import time

# Collection name must match the one used in rag_store.py
CHROMA_COLLECTION_NAME = "insurance_policy_clauses"


def get_gemini_client_for_rag():
    """Initializes and returns the Gemini Client for embeddings, checking for API key validity."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")
    return genai.Client(api_key=api_key)


class DummyCollection:
    """A dummy class returned when RAG setup fails, storing the specific reason for failure."""

    def __init__(self, failure_reason="RAG index is empty or failed to initialize (No clauses processed)."):
        self.failure_reason = failure_reason

    def count(self):
        # Always return 0 to trigger the check in rag_store.py
        return 0

    def query(self, *args, **kwargs):
        # Return the failure reason as the document content
        return {'documents': [[self.failure_reason]], 'metadatas': [[]]}

    def add(self, *args, **kwargs):
        pass


def extract_embeddings_from_response(response):
    """
    Safely extracts the list of embedding vectors from the response object,
    handling different SDK versions/structures by checking common attributes.
    """
    try:
        # Try the most likely properties for the list of vectors
        if hasattr(response, 'embedding') and response.embedding is not None:
            return response.embedding
        if hasattr(response, 'values') and response.values is not None:
            return response.values

        # Try iterating over nested embedding objects (like ContentEmbedding)
        if hasattr(response, 'embeddings') and response.embeddings is not None:
            return [e.value if hasattr(e, 'value') else e.embedding for e in response.embeddings]

        raise AttributeError("Could not find the embedding vector list in the API response object.")

    except Exception as e:
        raise AttributeError(f"Failed to extract embeddings due to response structure: {e}")


def get_chroma_client():
    """Initializes and returns a persistent ChromaDB client."""
    # Using a simple in-memory client for this demo.
    return chromadb.Client()


def get_rag_collection(client: chromadb.Client, clauses: list[dict]):
    """
    Creates or retrieves the ChromaDB collection and ensures the data is indexed.

    Returns:
        chromadb.api.models.Collection: The initialized collection (or a DummyCollection on failure).
    """

    try:
        # Initialize Gemini Client for embeddings lazily
        client_gemini = get_gemini_client_for_rag()

        # Get or create the collection
        collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

        if collection.count() == 0 and clauses:
            print("Generating embeddings and indexing policy clauses...")

            texts = [c['text'] for c in clauses]
            ids = [c['clause_id'] for c in clauses]
            metadatas = [{'page_num': c['page_num']} for c in clauses]

            # Generate embeddings in batches with robust error handling (backoff)
            embeddings = []
            batch_size = 50
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                # Exponential backoff retry logic for API calls
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = client_gemini.models.embed_content(
                            model='text-embedding-004',
                            contents=batch_texts
                        )

                        # Use the robust extraction function
                        extracted_vectors = extract_embeddings_from_response(response)
                        embeddings.extend(extracted_vectors)
                        break  # Success, break inner loop
                    except APIError as e:
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt  # 1, 2, 4 seconds
                            print(
                                f"API Error during embedding (Attempt {attempt + 1}): {e}. Retrying in {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            # Re-raise error if all retries fail
                            raise APIError(f"Failed to generate embeddings after {max_retries} attempts.")

            if embeddings:
                collection.add(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )

        return collection

    except ValueError as e:
        error_msg = f"RAG Setup Error (Key Missing): {e}"
        print(error_msg)
        return DummyCollection(error_msg)
    except APIError as e:
        error_msg = f"Gemini API Error during embedding generation: {e}"
        print(error_msg)
        return DummyCollection(error_msg)
    except Exception as e:
        error_msg = f"An unexpected error occurred in RAG setup: {e}"
        print(error_msg)
        return DummyCollection(error_msg)