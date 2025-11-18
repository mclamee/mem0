import os
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field

from mem0.embeddings.configs import EmbedderConfig
from mem0.graphs.configs import GraphStoreConfig
from mem0.llms.configs import LlmConfig
from mem0.vector_stores.configs import VectorStoreConfig
from mem0.configs.rerankers.config import RerankerConfig

# Set up the directory path
home_dir = os.path.expanduser("~")
mem0_dir = os.environ.get("MEM0_DIR") or os.path.join(home_dir, ".mem0")


class MemoryItem(BaseModel):
    id: str = Field(..., description="The unique identifier for the text data")
    memory: str = Field(
        ..., description="The memory deduced from the text data"
    )  # TODO After prompt changes from platform, update this
    hash: Optional[str] = Field(None, description="The hash of the memory")
    # The metadata value can be anything and not just string. Fix it
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the text data")
    score: Optional[float] = Field(None, description="The score associated with the text data")
    created_at: Optional[str] = Field(None, description="The timestamp when the memory was created")
    updated_at: Optional[str] = Field(None, description="The timestamp when the memory was updated")


class MemoryConfig(BaseModel):
    vector_store: VectorStoreConfig = Field(
        description="Configuration for the vector store",
        default_factory=VectorStoreConfig,
    )
    llm: LlmConfig = Field(
        description="Configuration for the language model",
        default_factory=LlmConfig,
    )
    embedder: EmbedderConfig = Field(
        description="Configuration for the embedding model",
        default_factory=EmbedderConfig,
    )
    history_db_path: str = Field(
        description="Path to the history database",
        default=os.path.join(mem0_dir, "history.db"),
    )
    graph_store: GraphStoreConfig = Field(
        description="Configuration for the graph",
        default_factory=GraphStoreConfig,
    )
    reranker: Optional[RerankerConfig] = Field(
        description="Configuration for the reranker",
        default=None,
    )
    version: str = Field(
        description="The version of the API",
        default="v1.1",
    )
    custom_fact_extraction_prompt: Optional[str] = Field(
        description="Custom prompt for the fact extraction (legacy, use custom_user_memory_prompt instead)",
        default=None,
    )
    custom_user_memory_prompt: Optional[str] = Field(
        description="Custom prompt for user memory extraction (from user messages)",
        default=None,
    )
    custom_agent_memory_prompt: Optional[str] = Field(
        description="Custom prompt for agent memory extraction (from assistant messages)",
        default=None,
    )
    custom_update_memory_prompt: Optional[str] = Field(
        description="Custom prompt for the update memory",
        default=None,
    )
    fact_categories: Optional[Union[list[str], dict[str, dict]]] = Field(
        description="Fact categories for memory extraction. Supports:\n"
                    "1. Simple list: ['preference', 'plan', 'conversation']\n"
                    "2. Detailed dict: {\n"
                    "     'preference': {'description': '个人偏好', 'examples': ['喜欢披萨'], 'temporal': False},\n"
                    "     'plan': {'description': '计划意图', 'examples': ['计划去日本'], 'temporal': True}\n"
                    "   }\n"
                    "3. Mixed mode (use defaults + custom): {\n"
                    "     'preference': 'use_default',  # Use built-in default config\n"
                    "     'plan': 'use_default',\n"
                    "     'astro_question': {'description': '...', 'examples': [...], 'temporal': True}\n"
                    "   }",
        default=None,
    )
    enable_structured_facts: bool = Field(
        description="Enable structured fact extraction with category metadata. "
                    "When True, facts are extracted as {'text': '...', 'category': '...', 'date': '...'} "
                    "instead of plain strings. Requires fact_categories to be set.",
        default=False,
    )


class AzureConfig(BaseModel):
    """
    Configuration settings for Azure.

    Args:
        api_key (str): The API key used for authenticating with the Azure service.
        azure_deployment (str): The name of the Azure deployment.
        azure_endpoint (str): The endpoint URL for the Azure service.
        api_version (str): The version of the Azure API being used.
        default_headers (Dict[str, str]): Headers to include in requests to the Azure API.
    """

    api_key: str = Field(
        description="The API key used for authenticating with the Azure service.",
        default=None,
    )
    azure_deployment: str = Field(description="The name of the Azure deployment.", default=None)
    azure_endpoint: str = Field(description="The endpoint URL for the Azure service.", default=None)
    api_version: str = Field(description="The version of the Azure API being used.", default=None)
    default_headers: Optional[Dict[str, str]] = Field(
        description="Headers to include in requests to the Azure API.", default=None
    )
