import logging
from typing import Literal, Optional

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase

logger = logging.getLogger(__name__)


class HuggingFaceEmbedding(EmbeddingBase):
    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)

        if config.huggingface_base_url:
            # HTTP API mode (TEI) - no local dependencies needed
            from openai import OpenAI

            self.client = OpenAI(base_url=config.huggingface_base_url)
            self.config.model = self.config.model or "tei"
            self.model = None  # Not used in HTTP API mode
            logger.info(f"HuggingFaceEmbedding initialized in HTTP API mode: {config.huggingface_base_url}")
        else:
            # Local mode - requires sentence_transformers
            try:
                from sentence_transformers import SentenceTransformer

                logging.getLogger("transformers").setLevel(logging.WARNING)
                logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
                logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
            except ImportError:
                raise ImportError(
                    "sentence_transformers is required for local HuggingFace embedding mode. "
                    "Install with: pip install sentence-transformers\n"
                    "Or use HTTP API mode by setting huggingface_base_url in config."
                )

            self.config.model = self.config.model or "multi-qa-MiniLM-L6-cos-v1"
            self.model = SentenceTransformer(self.config.model, **self.config.model_kwargs)
            self.config.embedding_dims = self.config.embedding_dims or self.model.get_sentence_embedding_dimension()
            self.client = None  # Not used in local mode
            logger.info(f"HuggingFaceEmbedding initialized in local mode: {self.config.model}")

    def embed(self, text, memory_action: Optional[Literal["add", "search", "update"]] = None):
        """
        Get the embedding for the given text using Hugging Face.

        Args:
            text (str): The text to embed.
            memory_action (optional): The type of embedding to use. Must be one of "add", "search", or "update". Defaults to None.
        Returns:
            list: The embedding vector.
        """
        if self.config.huggingface_base_url:
            return self.client.embeddings.create(
                input=text, model=self.config.model, **self.config.model_kwargs
            ).data[0].embedding
        else:
            return self.model.encode(text, convert_to_numpy=True).tolist()
