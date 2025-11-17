"""
Tests for HuggingFace Reranker with HTTP API support
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from mem0.configs.rerankers.huggingface import HuggingFaceRerankerConfig
from mem0.reranker.huggingface_reranker import HuggingFaceReranker


@pytest.fixture
def sample_documents():
    """Sample documents for testing"""
    return [
        {"memory": "I love pizza with extra cheese", "id": "1"},
        {"memory": "I prefer thin crust pizza", "id": "2"},
        {"memory": "I'm allergic to mushrooms", "id": "3"},
        {"memory": "My favorite color is blue", "id": "4"},
    ]


@pytest.fixture
def tei_rerank_response():
    """Mock TEI API response"""
    return [
        {"index": 1, "score": 0.95},
        {"index": 0, "score": 0.89},
        {"index": 2, "score": 0.12},
        {"index": 3, "score": 0.05},
    ]


class TestHuggingFaceRerankerConfig:
    """Test HuggingFaceRerankerConfig"""

    def test_config_defaults(self):
        """Test default configuration values"""
        config = HuggingFaceRerankerConfig()
        assert config.model == "BAAI/bge-reranker-base"
        assert config.huggingface_base_url is None
        assert config.timeout == 30
        assert config.device is None
        assert config.batch_size == 32
        assert config.max_length == 512
        assert config.normalize is True

    def test_config_with_http_api(self):
        """Test configuration with HTTP API settings"""
        config = HuggingFaceRerankerConfig(
            model="BAAI/bge-reranker-large",
            huggingface_base_url="http://localhost:8080",
            timeout=60,
            top_k=5
        )
        assert config.model == "BAAI/bge-reranker-large"
        assert config.huggingface_base_url == "http://localhost:8080"
        assert config.timeout == 60
        assert config.top_k == 5


class TestHuggingFaceRerankerHTTPMode:
    """Test HuggingFaceReranker in HTTP API mode"""

    def test_init_http_mode(self):
        """Test initialization in HTTP mode"""
        config = HuggingFaceRerankerConfig(
            huggingface_base_url="http://localhost:8080"
        )

        with patch('mem0.reranker.huggingface_reranker.HTTPX_AVAILABLE', True):
            with patch('mem0.reranker.huggingface_reranker.httpx.Client') as mock_client:
                reranker = HuggingFaceReranker(config)

                assert reranker.use_http_api is True
                assert reranker.api_url == "http://localhost:8080/rerank"
                mock_client.assert_called_once_with(timeout=30)

    def test_init_http_mode_without_httpx(self):
        """Test initialization in HTTP mode without httpx installed"""
        config = HuggingFaceRerankerConfig(
            huggingface_base_url="http://localhost:8080"
        )

        with patch('mem0.reranker.huggingface_reranker.HTTPX_AVAILABLE', False):
            with pytest.raises(ImportError, match="httpx package is required"):
                HuggingFaceReranker(config)

    @patch('mem0.reranker.huggingface_reranker.HTTPX_AVAILABLE', True)
    @patch('mem0.reranker.huggingface_reranker.httpx.Client')
    def test_rerank_http_api_success(self, mock_client_class, sample_documents, tei_rerank_response):
        """Test successful reranking via HTTP API"""
        # Setup mock client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = tei_rerank_response
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        config = HuggingFaceRerankerConfig(
            huggingface_base_url="http://localhost:8080",
            top_k=3
        )
        reranker = HuggingFaceReranker(config)

        # Perform reranking
        results = reranker.rerank("pizza preferences", sample_documents)

        # Verify API call
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "http://localhost:8080/rerank"
        assert "query" in call_args[1]["json"]
        assert "texts" in call_args[1]["json"]
        assert call_args[1]["json"]["query"] == "pizza preferences"

        # Verify results
        assert len(results) == 3  # top_k=3
        assert results[0]["id"] == "2"  # Index 1 with score 0.95
        assert results[0]["rerank_score"] == 0.95
        assert results[1]["id"] == "1"  # Index 0 with score 0.89
        assert results[1]["rerank_score"] == 0.89

    @patch('mem0.reranker.huggingface_reranker.HTTPX_AVAILABLE', True)
    @patch('mem0.reranker.huggingface_reranker.httpx.Client')
    def test_rerank_http_api_failure_fallback(self, mock_client_class, sample_documents):
        """Test fallback behavior when HTTP API fails"""
        # Setup mock client to raise exception
        mock_client = Mock()
        mock_client.post.side_effect = Exception("Connection failed")
        mock_client_class.return_value = mock_client

        config = HuggingFaceRerankerConfig(
            huggingface_base_url="http://localhost:8080",
            top_k=2
        )
        reranker = HuggingFaceReranker(config)

        # Perform reranking
        results = reranker.rerank("pizza preferences", sample_documents)

        # Should fallback to original order with zero scores
        assert len(results) == 2  # top_k=2
        assert all(doc.get("rerank_score") == 0.0 for doc in results)

    @patch('mem0.reranker.huggingface_reranker.HTTPX_AVAILABLE', True)
    @patch('mem0.reranker.huggingface_reranker.httpx.Client')
    def test_rerank_empty_documents(self, mock_client_class):
        """Test reranking with empty document list"""
        config = HuggingFaceRerankerConfig(
            huggingface_base_url="http://localhost:8080"
        )
        reranker = HuggingFaceReranker(config)

        results = reranker.rerank("test query", [])
        assert results == []


class TestHuggingFaceRerankerLocalMode:
    """Test HuggingFaceReranker in local inference mode"""

    def test_init_local_mode_without_transformers(self):
        """Test initialization in local mode without transformers installed"""
        config = HuggingFaceRerankerConfig()

        with patch('mem0.reranker.huggingface_reranker.TRANSFORMERS_AVAILABLE', False):
            with pytest.raises(ImportError, match="transformers package is required"):
                HuggingFaceReranker(config)

    @patch('mem0.reranker.huggingface_reranker.TRANSFORMERS_AVAILABLE', True)
    @patch('mem0.reranker.huggingface_reranker.torch')
    @patch('mem0.reranker.huggingface_reranker.AutoTokenizer')
    @patch('mem0.reranker.huggingface_reranker.AutoModelForSequenceClassification')
    def test_init_local_mode(self, mock_model_class, mock_tokenizer_class, mock_torch):
        """Test initialization in local mode"""
        mock_torch.cuda.is_available.return_value = False
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        config = HuggingFaceRerankerConfig()
        reranker = HuggingFaceReranker(config)

        assert reranker.use_http_api is False
        assert reranker.device == "cpu"
        mock_tokenizer_class.from_pretrained.assert_called_once_with("BAAI/bge-reranker-base")
        mock_model_class.from_pretrained.assert_called_once_with("BAAI/bge-reranker-base")


class TestHuggingFaceRerankerTextExtraction:
    """Test text extraction from different document formats"""

    @patch('mem0.reranker.huggingface_reranker.HTTPX_AVAILABLE', True)
    @patch('mem0.reranker.huggingface_reranker.httpx.Client')
    def test_text_extraction_memory_field(self, mock_client_class):
        """Test extraction from 'memory' field"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = [{"index": 0, "score": 0.9}]
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        config = HuggingFaceRerankerConfig(huggingface_base_url="http://localhost:8080")
        reranker = HuggingFaceReranker(config)

        docs = [{"memory": "test memory"}]
        reranker.rerank("query", docs)

        call_args = mock_client.post.call_args
        assert call_args[1]["json"]["texts"] == ["test memory"]

    @patch('mem0.reranker.huggingface_reranker.HTTPX_AVAILABLE', True)
    @patch('mem0.reranker.huggingface_reranker.httpx.Client')
    def test_text_extraction_text_field(self, mock_client_class):
        """Test extraction from 'text' field"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = [{"index": 0, "score": 0.9}]
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        config = HuggingFaceRerankerConfig(huggingface_base_url="http://localhost:8080")
        reranker = HuggingFaceReranker(config)

        docs = [{"text": "test text"}]
        reranker.rerank("query", docs)

        call_args = mock_client.post.call_args
        assert call_args[1]["json"]["texts"] == ["test text"]

    @patch('mem0.reranker.huggingface_reranker.HTTPX_AVAILABLE', True)
    @patch('mem0.reranker.huggingface_reranker.httpx.Client')
    def test_text_extraction_content_field(self, mock_client_class):
        """Test extraction from 'content' field"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = [{"index": 0, "score": 0.9}]
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        config = HuggingFaceRerankerConfig(huggingface_base_url="http://localhost:8080")
        reranker = HuggingFaceReranker(config)

        docs = [{"content": "test content"}]
        reranker.rerank("query", docs)

        call_args = mock_client.post.call_args
        assert call_args[1]["json"]["texts"] == ["test content"]
