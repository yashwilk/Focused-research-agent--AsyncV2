def validate_and_clean_question(question: object) -> str:
    """
    Validate and normalize a user research question.

    This function ensures the input is a string, trims surrounding
    whitespace, rejects blank input, rejects punctuation-only input, and
    rejects ultra-short input that is unlikely to be meaningful for the
    research workflow.

    This helper raises ValueError because it is reused directly by
    Pydantic/FastAPI request validation, where ValueError is the expected
    validator failure signal.

    Args:
        question: Raw user question to validate.

    Returns:
        str: Cleaned user question.

    Raises:
        ValueError: If the question is not a string, is blank, contains no
            letters or numbers, or is too short to be meaningful.
    """
    if not isinstance(question, str):
        raise ValueError("User query must be a string")

    cleaned_question = question.strip()

    if not cleaned_question:
        raise ValueError("No user query provided")

    if not any(character.isalnum() for character in cleaned_question):
        raise ValueError("User query must contain at least one letter or number")

    if len(cleaned_question) < 2:
        raise ValueError("User query is too short to research meaningfully")

    return cleaned_question