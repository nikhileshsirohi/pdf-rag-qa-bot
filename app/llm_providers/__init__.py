from app.llm_providers.openai_provider import OpenAIProvider
from app.llm_providers.gemini_provider import GeminiProvider
from app.llm_providers.hf_provider import HuggingFaceProvider


def get_llm_provider(
    provider: str,
    api_key: str | None = None,
    model: str | None = None
):
    provider = provider.lower()

    if provider == "openai":
        return OpenAIProvider(api_key=api_key, model=model or "gpt-4o-mini")

    if provider == "gemini":
        return GeminiProvider(api_key=api_key, model=model or "gemini-1.5-flash")

    if provider == "huggingface":
        return HuggingFaceProvider(model_name=model or "google/flan-t5-base")

    raise ValueError(f"Unsupported provider: {provider}")
