# from google import genai

# client = genai.Client(api_key="AIzaSyAcRyDRHv3x6ahDFA_qauFddXq2bLXNfXE")

# for m in client.models.list():
#     print(m.name, m.supported_generation_methods)

from google import genai
import json

client = genai.Client(api_key="AIzaSyAcRyDRHv3x6ahDFA_qauFddXq2bLXNfXE")

for m in client.models.list():
    # Safest: dump the raw model object to dict/json
    d = m.model_dump() if hasattr(m, "model_dump") else (m.dict() if hasattr(m, "dict") else vars(m))
    print(json.dumps(d, indent=2))
    print("-" * 80)