"""
Example: Using HuggingFace Text Embeddings Inference (TEI) for Reranking

This example demonstrates how to use the HuggingFaceReranker with TEI HTTP API
for high-performance reranking in production environments.

Prerequisites:
1. Install httpx: pip install httpx
2. Deploy TEI service:
   docker run --gpus all -p 8080:80 \
     ghcr.io/huggingface/text-embeddings-inference:1.6 \
     --model-id BAAI/bge-reranker-large
"""

from mem0 import Memory

# Configuration for mem0 with TEI reranker
config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4.1-nano-2025-04-14",
            "temperature": 0.0
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small"
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "mem0",
            "embedding_model_dims": 1536
        }
    },
    # HuggingFace Reranker with TEI HTTP API
    "reranker": {
        "provider": "huggingface",
        "config": {
            "model": "BAAI/bge-reranker-large",  # Model name (for reference)
            "huggingface_base_url": "http://localhost:8080",  # TEI service URL
            "timeout": 30,  # Request timeout in seconds
            "top_k": 5  # Return top 5 reranked results
        }
    }
}

# Alternative: Kubernetes service URL
# "huggingface_base_url": "http://reranker-service.namespace.svc.cluster.local:8080"

# Initialize memory with TEI reranker
memory = Memory.from_config(config)

# Add some memories
messages = [
    {"role": "user", "content": "I love pizza with extra cheese"},
    {"role": "assistant", "content": "Great! I'll remember you like extra cheese on pizza."},
    {"role": "user", "content": "I prefer thin crust over thick crust"},
    {"role": "assistant", "content": "Noted! Thin crust is your preference."},
    {"role": "user", "content": "I'm allergic to mushrooms"},
    {"role": "assistant", "content": "Important! I'll remember your mushroom allergy."},
]

memory.add(messages, user_id="user_123")

# Search with reranking
query = "What are my pizza preferences?"
results = memory.search(query, user_id="user_123")

print("Search Results with TEI Reranking:")
print("=" * 60)
for i, result in enumerate(results, 1):
    print(f"\n{i}. Memory: {result.get('memory', 'N/A')}")
    print(f"   Rerank Score: {result.get('rerank_score', 'N/A'):.4f}")
    print(f"   Relevance Score: {result.get('score', 'N/A'):.4f}")


# Example for local inference (without TEI)
print("\n\n" + "=" * 60)
print("Alternative: Local HuggingFace Reranker")
print("=" * 60)

config_local = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4.1-nano-2025-04-14"
        }
    },
    "reranker": {
        "provider": "huggingface",
        "config": {
            "model": "BAAI/bge-reranker-base",  # Smaller model for local use
            "device": "cpu",  # or "cuda" if GPU available
            "batch_size": 32,
            "max_length": 512,
            "normalize": True,
            "top_k": 5
        }
    }
}

print("""
To use local inference mode:
1. Install transformers and torch: pip install transformers torch
2. Remove or omit 'huggingface_base_url' from config
3. The model will be downloaded and run locally
""")
