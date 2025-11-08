import re


def segment_text_into_clauses(page_texts: list[str]) -> list[dict]:
    """
    Segments raw page texts into smaller, numbered clauses (paragraphs).

    Returns:
        list[dict]: A list of dictionaries, each containing:
            - 'clause_id': Unique identifier (e.g., 'p1_c1')
            - 'page_num': The page number (1-indexed)
            - 'text': The segmented clause text
    """
    all_clauses = []
    clause_counter = 0

    for page_num, page_text in enumerate(page_texts, 1):
        # Use regex to split text by two or more newlines, indicating a paragraph break
        paragraphs = re.split(r'\n\s*\n', page_text.strip())

        for para_index, paragraph in enumerate(paragraphs):
            cleaned_paragraph = paragraph.strip()
            if cleaned_paragraph:
                clause_counter += 1
                all_clauses.append({
                    'clause_id': f'p{page_num}_c{para_index + 1}',
                    'page_num': page_num,
                    'text': cleaned_paragraph
                })

    return all_clauses


def count_clauses(clauses: list[dict]) -> int:
    """Returns the total number of identified clauses."""
    return len(clauses)