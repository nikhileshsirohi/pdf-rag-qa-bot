import torch
from transformers import AutoTokenizer, AutoModel

class EmbeddingGenerator:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.device = self.__get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def __get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def embed_texts(self, texts: list[str]) -> torch.Tensor:
        """
        Generate embeddings for a list of texts.

        Args:
            texts (List[str]): list of text chunks

        Returns:
            torch.Tensor: tensor of embeddings
        """
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            model_output = self.model(**encoded)

        # Mean pooling with attention mask
        token_embeddings = model_output.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1)

        embeddings = (token_embeddings * attention_mask).sum(dim=1)
        embeddings = embeddings / attention_mask.sum(dim=1)

        return embeddings.cpu()
    # FAISS do not work on MPS device that's why we move embeddings to CPU before returning
    # Models can run on GPU, but vector databases work on CPU.
    # compute embeddings on GPU
    # store & search embeddings on CPU