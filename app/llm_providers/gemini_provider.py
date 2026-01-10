try:
    from google import genai
except ImportError:
    genai = None

from app.llm_providers.base import BaseLLMProvider


class GeminiProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "models/gemini-flash-latest"):
        if genai is None:
            raise ImportError(
                "google-genai not installed. Run: pip install google-genai"
            )
        if not model.startswith("models/"):
            model = "models/" + model
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        return response.text.strip()
