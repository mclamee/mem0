# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **mem0** (mem-zero) - an intelligent memory layer for personalized AI assistants and agents. The project provides persistent, personalized memory capabilities that enable AI systems to remember user preferences and adapt over time.

**Key Performance:**
- +26% Accuracy over OpenAI Memory on LOCOMO benchmark
- 91% Faster responses than full-context approaches
- 90% Lower token usage

## Project Structure

```
mem0/                    # Core library package
├── memory/             # Memory management (main.py, storage.py, telemetry.py)
├── client/             # API clients (main.py for hosted, project.py for projects)
├── llms/               # LLM integrations (OpenAI, Groq, Together, Ollama, etc.)
├── embeddings/         # Embedding providers
├── vector_stores/      # Vector DB integrations (Qdrant, Chroma, Weaviate, etc.)
├── graphs/             # Graph database integrations (Neo4j, Memgraph, Kuzu)
├── reranker/           # Reranking implementations
├── configs/            # Configuration schemas and prompts
└── utils/              # Shared utilities

server/                 # FastAPI REST API server
tests/                  # Test suite
examples/               # Usage examples
docs/                   # Documentation (mintlify)
embedchain/             # Legacy embedchain implementation
openmemory/             # OpenMemory implementation
mem0-ts/                # TypeScript SDK
```

## Common Commands

### Setup & Installation

```bash
# Install hatch package manager (if not installed)
pip install hatch

# Create development environment
hatch env create

# Install all optional dependencies (for full feature development)
make install_all

# Install pre-commit hooks (required before first commit)
pre-commit install
```

### Development

```bash
# Activate specific Python version environment
hatch shell dev_py_3_9   # Python 3.9
hatch shell dev_py_3_10  # Python 3.10
hatch shell dev_py_3_11  # Python 3.11
hatch shell dev_py_3_12  # Python 3.12

# Format code (uses ruff)
make format

# Sort imports (uses isort with black profile)
make sort

# Lint code (uses ruff)
make lint

# All quality checks at once
make all
```

### Testing

```bash
# Run all tests with default Python version
make test

# Run tests for specific Python versions
make test-py-3.9
make test-py-3.10
make test-py-3.11
make test-py-3.12

# Run specific test file
hatch run pytest tests/test_memory.py

# Run with coverage
hatch run pytest tests/ --cov=mem0 --cov-report=html

# Run specific test
hatch run pytest tests/test_memory.py::test_function_name
```

### Building & Publishing

```bash
# Build package
make build

# Publish to PyPI
make publish

# Clean build artifacts
make clean
```

### Documentation

```bash
# Run local documentation server
make docs
# Opens mintlify dev server at http://localhost:3000
```

### Running REST API Server

```bash
cd server/

# Start development server
uvicorn main:app --reload --port 8000

# Access API docs at http://localhost:8000/docs

# With docker-compose
docker-compose up -d
```

## Architecture Patterns

### Memory System Architecture

The core `Memory` class in `mem0/memory/main.py` follows a modular factory pattern:

1. **Configuration**: Uses Pydantic-based `MemoryConfig` from `mem0/configs/base.py`
2. **Components**: Instantiated via factory classes in `mem0/utils/factory.py`:
   - `LlmFactory` - LLM provider (OpenAI, Groq, Together, Ollama, etc.)
   - `EmbedderFactory` - Embedding provider
   - `VectorStoreFactory` - Vector database (Qdrant default)
   - `GraphStoreFactory` - Optional graph database (Neo4j, Memgraph, Kuzu)
   - `RerankerFactory` - Optional reranker

3. **Storage**: SQLite-based local storage in `~/.mem0/` via `mem0/memory/storage.py`

### Memory Operations

Key methods in the `Memory` class:

```python
# Add memories from conversation messages
memory.add(messages, user_id=None, agent_id=None, run_id=None, metadata={})

# Search memories by query
memory.search(query, user_id=None, agent_id=None, run_id=None, limit=100)

# Get all memories for an entity
memory.get_all(user_id=None, agent_id=None, run_id=None)

# Update specific memory
memory.update(memory_id, data)

# Delete specific memory
memory.delete(memory_id)

# Delete all memories for an entity
memory.delete_all(user_id=None, agent_id=None, run_id=None)

# Reset all memories
memory.reset()
```

### Memory Types

Defined in `mem0/configs/enums.py`:
- **User memories**: Associated with `user_id`
- **Agent memories**: Associated with `agent_id`
- **Session memories**: Associated with `run_id`

### Client API Architecture

Two client types in `mem0/client/`:

1. **MemoryClient** (`main.py`): For hosted platform API
2. **ProjectMemory** (`project.py`): For project-based memory management

Both use async/sync patterns with session management.

## Configuration

### Memory Configuration

Default config structure (can be customized):

```python
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
    }
}

memory = Memory.from_config(config)
```

### Environment Variables

Common environment variables:
- `OPENAI_API_KEY` - OpenAI API key
- `QDRANT_URL` / `QDRANT_API_KEY` - Qdrant connection
- `NEO4J_URL` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` - Neo4j connection

## Development Best Practices

### Code Quality

- **Linting**: Uses `ruff` for fast Python linting (line length: 120)
- **Formatting**: Uses `ruff format` for code formatting
- **Import Sorting**: Uses `isort` with black profile
- **Pre-commit**: Runs ruff and isort automatically on commits
- **Type Hints**: Use type hints throughout new code
- **Excluded Paths**: `embedchain/` and `openmemory/` are excluded from linting

### Testing Requirements

- Write tests for all new features and bug fixes
- Maintain compatibility with Python 3.9-3.12
- Tests must pass across all supported Python versions
- Use pytest fixtures for common setup
- Mock external services (LLMs, vector stores) in unit tests

### Package Management

- **Tool**: `hatch` for environment and package management
- **Dependencies**: Defined in `pyproject.toml` with optional dependency groups:
  - `graph` - Graph database support
  - `vector_stores` - Additional vector store providers
  - `llms` - Additional LLM providers
  - `extras` - Additional features (sentence-transformers, etc.)
  - `test` - Testing dependencies
  - `dev` - Development tools

### Adding New Integrations

When adding new LLM/embedding/vector store providers:

1. Create provider class in appropriate directory (`llms/`, `embeddings/`, `vector_stores/`)
2. Inherit from base class (e.g., `LLMBase`, `EmbedderBase`, `VectorStoreBase`)
3. Register in corresponding factory (`mem0/utils/factory.py`)
4. Add required dependencies to appropriate optional dependency group in `pyproject.toml`
5. Add tests in `tests/` directory
6. Update documentation

## Important Notes

1. **Default LLM**: `gpt-4.1-nano-2025-04-14` from OpenAI (requires `OPENAI_API_KEY`)
2. **Default Vector Store**: Qdrant (runs in-memory mode by default)
3. **Memory Storage**: SQLite database in `~/.mem0/` directory
4. **Telemetry**: Built-in telemetry via PostHog (can be disabled)
5. **Async Support**: Most operations support both sync and async patterns
6. **Message Format**: Uses OpenAI-compatible message format with `role` and `content`
7. **Migration**: Version 1.0.0 includes breaking changes - see `MIGRATION_GUIDE_v1.0.md`

## Related Subprojects

- **server/**: FastAPI REST API server for mem0
- **mem0-ts/**: TypeScript/JavaScript SDK
- **openmemory/**: OpenMemory implementation
- **embedchain/**: Legacy embedchain implementation (deprecated)
- **vercel-ai-sdk/**: Vercel AI SDK integration examples
