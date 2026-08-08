import os
from dotenv import load_dotenv

load_dotenv()


def get_llm_config():
    """Load and validate LLM configuration from environment variables.

    Returns:
        dict: A dictionary containing provider, model, temperature,
        max_retries, and api_key.

    Raises:
        ValueError: If a required environment variable is missing or
        if temperature or max_retries cannot be parsed correctly.
    """
    max_tokens_raw = os.getenv("LLM_MAX_TOKENS", "4096")

    try:
        max_tokens = int(max_tokens_raw)
    except ValueError:
        raise ValueError(f"LLM_MAX_TOKENS must be an int. Got: {max_tokens_raw}")

    provider = os.getenv("LLM_PROVIDER")
    model = os.getenv("LLM_MODEL")
    temp_raw = os.getenv("LLM_TEMPERATURE")
    retries_raw = os.getenv("LLM_MAX_RETRIES")
    api_key = os.getenv("LLM_API_KEY")

    if (
        (not provider or not provider.strip())
        or (not model or not model.strip())
        or (not temp_raw or not temp_raw.strip())
        or (not retries_raw or not retries_raw.strip())
        or (not api_key or not api_key.strip())
    ):
        raise ValueError(
            "LLM provider, LLM Model, LLM temperature, number of retries and api key should be given in .env file!"
        )

    try:
        temperature = float(temp_raw)
    except ValueError:
        raise ValueError(f"LLM_TEMPERATURE must be a float. Got: {temp_raw}")

    try:
        max_retries = int(retries_raw)
    except ValueError:
        raise ValueError(f"LLM_MAX_RETRIES must be an int. Got: {retries_raw}")

    return {
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "max_retries": max_retries,
        "api_key": api_key,
        "max_tokens": max_tokens,
    }
