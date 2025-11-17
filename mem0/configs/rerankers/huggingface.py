from typing import Optional
from pydantic import Field

from mem0.configs.rerankers.base import BaseRerankerConfig


class HuggingFaceRerankerConfig(BaseRerankerConfig):
    """
    Configuration class for HuggingFace reranker-specific parameters.
    Inherits from BaseRerankerConfig and adds HuggingFace-specific settings.

    Supports two modes:
    1. Local inference: Loads model locally using transformers (default)
    2. HTTP API inference: Calls HuggingFace Text Embeddings Inference (TEI) service
    """

    model: Optional[str] = Field(default="BAAI/bge-reranker-base", description="The HuggingFace model to use for reranking")

    # HTTP API mode parameters
    huggingface_base_url: Optional[str] = Field(
        default=None,
        description="Base URL for HuggingFace Text Embeddings Inference (TEI) service. If set, will use HTTP API instead of local model."
    )
    timeout: int = Field(default=30, description="Request timeout in seconds for HTTP API calls")

    # Local inference mode parameters
    device: Optional[str] = Field(default=None, description="Device to run the model on ('cpu', 'cuda', etc.). Only used in local mode.")
    batch_size: int = Field(default=32, description="Batch size for processing documents. Only used in local mode.")
    max_length: int = Field(default=512, description="Maximum length for tokenization. Only used in local mode.")
    normalize: bool = Field(default=True, description="Whether to normalize scores. Only used in local mode.")
