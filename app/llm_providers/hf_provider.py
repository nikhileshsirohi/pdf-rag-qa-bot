from transformers import pipeline
from app.llm_providers.base import BaseLLMProvider


class HuggingFaceProvider(BaseLLMProvider):
    def __init__(self, model_name: str = "google/flan-t5-base"):
        self.pipe = pipeline(
            "text2text-generation",
            model=model_name,
            max_new_tokens=256
        )

    def generate(self, prompt: str) -> str:
        output = self.pipe(prompt)
        return output[0]["generated_text"].strip()
