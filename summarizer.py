from google import genai
from google.genai import types
from google.genai.errors import APIError
import json
import os

# Define the model here
MODEL = 'gemini-2.5-flash'


def get_gemini_client():
    """Initializes and returns the Gemini Client, checking for API key validity."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")

    return genai.Client(api_key=api_key)


def generate_full_summary(full_policy_text: str) -> str:
    """Generates a concise, high-level summary of the entire policy."""

    try:
        # Initialize the client lazily (on demand)
        client = get_gemini_client()
    except ValueError as e:
        return f"Client Initialization Error: {e}"

    system_instruction = (
        "You are an expert Insurance Policy Analyst. Your task is to analyze the provided policy text and generate "
        "a concise, easy-to-understand summary. The summary must be at most 50 lines long. "
        "Focus on the main coverage, key exclusions, and critical policy definitions."
    )

    user_prompt = f"Provide a complete, human-readable summary of the following insurance policy text:\n\n{full_policy_text}"

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=[user_prompt],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        )
        return response.text
    except APIError as e:
        return f"Gemini API Error during full summary generation: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"


def analyze_page_content(page_texts: list[str]) -> list[dict]:
    """
    Analyzes each page's content, classifies it, and provides a plain-language summary.
    Uses structured output (JSON schema) for reliable data extraction.
    """

    try:
        # Initialize the client lazily (on demand)
        client = get_gemini_client()
    except ValueError as e:
        print(f"Client Initialization Error during page analysis: {e}")
        return []

    system_instruction = (
        "You are a Policy Segmentation and Classification Agent. Your task is to review each provided page of the policy. "
        "For each page, you must identify the primary section it covers (Classification) and provide a simplified, "
        "plain-language summary of that page's content. Do not combine pages. Output ONLY the JSON array."
    )

    user_prompt = (
        "Analyze the following policy pages. For each page, identify its main classification (must be one of: "
        "'Coverage', 'Exclusions', 'Claims Process', 'Deductibles/Limits', 'General Terms', 'Definitions') and "
        "provide a single, simplified paragraph (the summary)."
    )

    # Define the required JSON schema for the output
    analysis_schema = types.Schema(
        type=types.Type.ARRAY,
        items=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "pageNumber": types.Schema(type=types.Type.INTEGER),
                "classification": types.Schema(
                    type=types.Type.STRING,
                    description="One of: Coverage, Exclusions, Claims Process, Deductibles/Limits, General Terms, Definitions."
                ),
                "summary": types.Schema(
                    type=types.Type.STRING,
                    description="Simplified, plain-language summary of the page's content."
                )
            },
            required=["pageNumber", "classification", "summary"]
        )
    )

    # Format the input for the model
    input_data = []
    for page_num, text in enumerate(page_texts, 1):
        input_data.append(f"--- PAGE {page_num} ---\n{text}\n")

    full_prompt = user_prompt + "\n\n" + "\n".join(input_data)

    # Generate content with structured output
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=[full_prompt],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=analysis_schema
            )
        )

        # The response text is a JSON string
        json_string = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(json_string)

    except APIError as e:
        print(f"API Error during page analysis: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}. Raw response: {response.text}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred during page analysis: {e}")
        return []