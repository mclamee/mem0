from typing import List, Dict, Any, Union
import numpy as np
import logging

from mem0.reranker.base import BaseReranker
from mem0.configs.rerankers.base import BaseRerankerConfig
from mem0.configs.rerankers.huggingface import HuggingFaceRerankerConfig

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class HuggingFaceReranker(BaseReranker):
    """
    HuggingFace reranker implementation with support for both local and HTTP API inference.

    Supports two modes:
    1. Local inference: Loads model locally using transformers (default)
    2. HTTP API inference: Calls HuggingFace Text Embeddings Inference (TEI) service
    """

    def __init__(self, config: Union[BaseRerankerConfig, HuggingFaceRerankerConfig, Dict]):
        """
        Initialize HuggingFace reranker.

        Args:
            config: Configuration object with reranker parameters

        Raises:
            ImportError: If required dependencies are not available
            ValueError: If configuration is invalid
        """
        # Convert to HuggingFaceRerankerConfig if needed
        if isinstance(config, dict):
            config = HuggingFaceRerankerConfig(**config)
        elif isinstance(config, BaseRerankerConfig) and not isinstance(config, HuggingFaceRerankerConfig):
            # Convert BaseRerankerConfig to HuggingFaceRerankerConfig with defaults
            config = HuggingFaceRerankerConfig(
                provider=getattr(config, 'provider', 'huggingface'),
                model=getattr(config, 'model', 'BAAI/bge-reranker-base'),
                api_key=getattr(config, 'api_key', None),
                top_k=getattr(config, 'top_k', None),
                huggingface_base_url=getattr(config, 'huggingface_base_url', None),
                device=None,  # Will auto-detect
                batch_size=32,  # Default
                max_length=512,  # Default
                normalize=True,  # Default
            )

        self.config = config

        # Determine mode: HTTP API or local inference
        self.use_http_api = bool(self.config.huggingface_base_url)

        if self.use_http_api:
            # HTTP API mode
            if not HTTPX_AVAILABLE:
                raise ImportError(
                    "httpx package is required for HuggingFace HTTP API mode. "
                    "Install with: pip install httpx"
                )
            self.http_client = httpx.Client(timeout=self.config.timeout)
            self.api_url = self.config.huggingface_base_url.rstrip('/') + '/rerank'
            logger.info(f"Initialized HuggingFaceReranker in HTTP API mode with URL: {self.api_url}")
        else:
            # Local inference mode
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "transformers package is required for HuggingFaceReranker local mode. "
                    "Install with: pip install transformers torch"
                )

            # Set device
            if self.config.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = self.config.device

            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.config.model)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Initialized HuggingFaceReranker in local mode with model: {self.config.model} on device: {self.device}")

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
        """
        Rerank documents using HuggingFace cross-encoder model.

        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Number of top documents to return

        Returns:
            List of reranked documents with rerank_score
        """
        if not documents:
            return documents

        if self.use_http_api:
            return self._rerank_http_api(query, documents, top_k)
        else:
            return self._rerank_local(query, documents, top_k)

    def _rerank_http_api(self, query: str, documents: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
        """
        Rerank documents using HuggingFace TEI HTTP API.

        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Number of top documents to return

        Returns:
            List of reranked documents with rerank_score
        """
        # Extract text content for reranking
        doc_texts = []
        for doc in documents:
            if 'memory' in doc:
                doc_texts.append(doc['memory'])
            elif 'text' in doc:
                doc_texts.append(doc['text'])
            elif 'content' in doc:
                doc_texts.append(doc['content'])
            else:
                doc_texts.append(str(doc))

        try:
            # Call TEI /rerank API
            response = self.http_client.post(
                self.api_url,
                json={
                    "query": query,
                    "texts": doc_texts,
                    "raw_scores": False,  # Get normalized scores
                    "return_text": False,  # We already have the texts
                }
            )
            response.raise_for_status()
            results = response.json()

            # TEI returns: [{"index": 0, "score": 0.99}, {"index": 1, "score": 0.12}, ...]
            # Results are already sorted by score in descending order
            reranked_docs = []
            for result in results:
                idx = result['index']
                score = result['score']
                reranked_doc = documents[idx].copy()
                reranked_doc['rerank_score'] = float(score)
                reranked_docs.append(reranked_doc)

            # Apply top_k limit
            final_top_k = top_k or self.config.top_k
            if final_top_k:
                reranked_docs = reranked_docs[:final_top_k]

            return reranked_docs

        except Exception as e:
            logger.warning(f"HTTP API reranking failed: {e}. Falling back to original order.")
            # Fallback to original order if reranking fails
            for doc in documents:
                doc['rerank_score'] = 0.0
            final_top_k = top_k or self.config.top_k
            return documents[:final_top_k] if final_top_k else documents

    def _rerank_local(self, query: str, documents: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
        """
        Rerank documents using local HuggingFace model.

        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Number of top documents to return

        Returns:
            List of reranked documents with rerank_score
        """
        # Extract text content for reranking
        doc_texts = []
        for doc in documents:
            if 'memory' in doc:
                doc_texts.append(doc['memory'])
            elif 'text' in doc:
                doc_texts.append(doc['text'])
            elif 'content' in doc:
                doc_texts.append(doc['content'])
            else:
                doc_texts.append(str(doc))

        try:
            scores = []

            # Process documents in batches
            for i in range(0, len(doc_texts), self.config.batch_size):
                batch_docs = doc_texts[i:i + self.config.batch_size]
                batch_pairs = [[query, doc] for doc in batch_docs]

                # Tokenize batch
                inputs = self.tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt"
                ).to(self.device)

                # Get scores
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    batch_scores = outputs.logits.squeeze(-1).cpu().numpy()

                    # Handle single item case
                    if batch_scores.ndim == 0:
                        batch_scores = [float(batch_scores)]
                    else:
                        batch_scores = batch_scores.tolist()

                    scores.extend(batch_scores)

            # Normalize scores if requested
            if self.config.normalize:
                scores = np.array(scores)
                scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
                scores = scores.tolist()

            # Combine documents with scores
            doc_score_pairs = list(zip(documents, scores))

            # Sort by score (descending)
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

            # Apply top_k limit
            final_top_k = top_k or self.config.top_k
            if final_top_k:
                doc_score_pairs = doc_score_pairs[:final_top_k]

            # Create reranked results
            reranked_docs = []
            for doc, score in doc_score_pairs:
                reranked_doc = doc.copy()
                reranked_doc['rerank_score'] = float(score)
                reranked_docs.append(reranked_doc)

            return reranked_docs

        except Exception as e:
            logger.warning(f"Local reranking failed: {e}. Falling back to original order.")
            # Fallback to original order if reranking fails
            for doc in documents:
                doc['rerank_score'] = 0.0
            final_top_k = top_k or self.config.top_k
            return documents[:final_top_k] if final_top_k else documents