import logging
import os
import shutil

from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    Range,
    VectorParams,
)

from mem0.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


class Qdrant(VectorStoreBase):
    def __init__(
        self,
        collection_name: str,
        embedding_model_dims: int,
        client: QdrantClient = None,
        host: str = None,
        port: int = None,
        path: str = None,
        url: str = None,
        api_key: str = None,
        on_disk: bool = False,
        hybrid_search: bool = False,
    ):
        """
        Initialize the Qdrant vector store.

        Args:
            collection_name (str): Name of the collection.
            embedding_model_dims (int): Dimensions of the embedding model.
            client (QdrantClient, optional): Existing Qdrant client instance.
            host (str, optional): Host address for Qdrant server.
            port (int, optional): Port for Qdrant server.
            path (str, optional): Path for local Qdrant database.
            url (str, optional): Full URL for Qdrant server.
            api_key (str, optional): API key for Qdrant server.
            on_disk (bool, optional): Enables persistent storage. Defaults to False.
            hybrid_search (bool, optional): Enable BM25 hybrid search with RRF fusion. Defaults to False.
        """
        if client:
            self.client = client
            self.is_local = False
        else:
            params = {}
            if api_key:
                params["api_key"] = api_key
            if url:
                params["url"] = url
            if host and port:
                params["host"] = host
                params["port"] = port

            if not params:
                params["path"] = path
                self.is_local = True
                if not on_disk:
                    if os.path.exists(path) and os.path.isdir(path):
                        shutil.rmtree(path)
            else:
                self.is_local = False

            self.client = QdrantClient(**params)

        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims
        self.on_disk = on_disk
        self.hybrid_search = hybrid_search
        self.create_col(embedding_model_dims, on_disk)

    def create_col(self, vector_size: int, on_disk: bool, distance: Distance = Distance.COSINE):
        """
        Create a new collection.

        Args:
            vector_size (int): Size of the vectors to be stored.
            on_disk (bool): Enables persistent storage.
            distance (Distance, optional): Distance metric. Defaults to Distance.COSINE.
        """
        # Skip creating collection if already exists
        response = self.list_cols()
        for collection in response.collections:
            if collection.name == self.collection_name:
                logger.debug(f"Collection {self.collection_name} already exists. Skipping creation.")
                self._create_filter_indexes()
                return

        if self.hybrid_search:
            # Named vectors: dense + BM25 sparse
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(size=vector_size, distance=distance, on_disk=on_disk),
                },
                sparse_vectors_config={
                    "bm25": models.SparseVectorParams(
                        modifier=models.Modifier.IDF,
                    ),
                },
            )
        else:
            # Anonymous vectors (legacy mode)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance, on_disk=on_disk),
            )
        self._create_filter_indexes()

    def _create_filter_indexes(self):
        """Create indexes for commonly used filter fields to enable filtering."""
        # Only create payload indexes for remote Qdrant servers
        if self.is_local:
            logger.debug("Skipping payload index creation for local Qdrant (not supported)")
            return

        common_fields = ["user_id", "agent_id", "run_id", "actor_id"]

        for field in common_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema="keyword"
                )
                logger.info(f"Created index for {field} in collection {self.collection_name}")
            except Exception as e:
                logger.debug(f"Index for {field} might already exist: {e}")

    def _build_bm25_document(self, text: str) -> models.Document:
        """Build a BM25 Document for sparse vector indexing."""
        return models.Document(
            text=text,
            model="Qdrant/bm25",
            options=models.Bm25Config(
                tokenizer=models.TokenizerType.MULTILINGUAL,
                language="none",
            ),
        )

    def insert(self, vectors: list, payloads: list = None, ids: list = None):
        """
        Insert vectors into a collection.

        Args:
            vectors (list): List of vectors to insert.
            payloads (list, optional): List of payloads corresponding to vectors.
            ids (list, optional): List of IDs corresponding to vectors.
        """
        logger.info(f"Inserting {len(vectors)} vectors into collection {self.collection_name}")

        if self.hybrid_search:
            points = []
            for idx, vector in enumerate(vectors):
                point_id = idx if ids is None else ids[idx]
                payload = payloads[idx] if payloads else {}
                text = payload.get("data", "") if payload else ""

                point = PointStruct(
                    id=point_id,
                    vector={
                        "dense": vector,
                        "bm25": self._build_bm25_document(text),
                    },
                    payload=payload,
                )
                points.append(point)
        else:
            points = [
                PointStruct(
                    id=idx if ids is None else ids[idx],
                    vector=vector,
                    payload=payloads[idx] if payloads else {},
                )
                for idx, vector in enumerate(vectors)
            ]

        self.client.upsert(collection_name=self.collection_name, points=points)

    def _create_filter(self, filters: dict) -> Filter:
        """
        Create a Filter object from the provided filters.

        Args:
            filters (dict): Filters to apply.

        Returns:
            Filter: The created Filter object.
        """
        if not filters:
            return None

        conditions = []
        for key, value in filters.items():
            if isinstance(value, dict) and "gte" in value and "lte" in value:
                conditions.append(FieldCondition(key=key, range=Range(gte=value["gte"], lte=value["lte"])))
            else:
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
        return Filter(must=conditions) if conditions else None

    def search(self, query: str, vectors: list, limit: int = 5, filters: dict = None) -> list:
        """
        Search for similar vectors.

        Args:
            query (str): Query text (used for BM25 in hybrid mode).
            vectors (list): Query vector.
            limit (int, optional): Number of results to return. Defaults to 5.
            filters (dict, optional): Filters to apply to the search.

        Returns:
            list: Search results.
        """
        query_filter = self._create_filter(filters) if filters else None

        if self.hybrid_search:
            # Hybrid search: dense + BM25 prefetch, RRF fusion
            prefetch_limit = 20
            hits = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(
                        query=vectors,
                        using="dense",
                        limit=prefetch_limit,
                        filter=query_filter,
                    ),
                    models.Prefetch(
                        query=self._build_bm25_document(query),
                        using="bm25",
                        limit=prefetch_limit,
                        filter=query_filter,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
            )
        else:
            # Pure dense search (legacy mode)
            hits = self.client.query_points(
                collection_name=self.collection_name,
                query=vectors,
                query_filter=query_filter,
                limit=limit,
            )

        return hits.points

    def delete(self, vector_id: int):
        """
        Delete a vector by ID.

        Args:
            vector_id (int): ID of the vector to delete.
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(
                points=[vector_id],
            ),
        )

    def update(self, vector_id: int, vector: list = None, payload: dict = None):
        """
        Update a vector and its payload.

        Args:
            vector_id (int): ID of the vector to update.
            vector (list, optional): Updated vector.
            payload (dict, optional): Updated payload.
        """
        if vector is None:
            # Only update payload, keep existing vector
            if payload:
                self.client.set_payload(
                    collection_name=self.collection_name,
                    payload=payload,
                    points=[vector_id],
                )
        else:
            if self.hybrid_search:
                text = payload.get("data", "") if payload else ""
                point = PointStruct(
                    id=vector_id,
                    vector={
                        "dense": vector,
                        "bm25": self._build_bm25_document(text),
                    },
                    payload=payload,
                )
            else:
                point = PointStruct(id=vector_id, vector=vector, payload=payload)
            self.client.upsert(collection_name=self.collection_name, points=[point])

    def get(self, vector_id: int) -> dict:
        """
        Retrieve a vector by ID.

        Args:
            vector_id (int): ID of the vector to retrieve.

        Returns:
            dict: Retrieved vector.
        """
        result = self.client.retrieve(collection_name=self.collection_name, ids=[vector_id], with_payload=True)
        return result[0] if result else None

    def list_cols(self) -> list:
        """
        List all collections.

        Returns:
            list: List of collection names.
        """
        return self.client.get_collections()

    def delete_col(self):
        """Delete a collection."""
        self.client.delete_collection(collection_name=self.collection_name)

    def col_info(self) -> dict:
        """
        Get information about a collection.

        Returns:
            dict: Collection information.
        """
        return self.client.get_collection(collection_name=self.collection_name)

    def list(self, filters: dict = None, limit: int = 100) -> list:
        """
        List all vectors in a collection.

        Args:
            filters (dict, optional): Filters to apply to the list.
            limit (int, optional): Number of vectors to return. Defaults to 100.

        Returns:
            list: List of vectors.
        """
        query_filter = self._create_filter(filters) if filters else None
        result = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return result

    def reset(self):
        """Reset the index by deleting and recreating it."""
        logger.warning(f"Resetting index {self.collection_name}...")
        self.delete_col()
        self.create_col(self.embedding_model_dims, self.on_disk)
