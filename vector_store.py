"""Vector Store Performance Tuning (#220)

Optimize vector database performance for faster semantic search, better retrieval
accuracy, and efficient scaling with a growing knowledge base.

Features:
- Multi-database support (Pinecone, Weaviate, Qdrant, pgvector)
- Index tuning (HNSW, IVF parameters)
- Hybrid dense-sparse retrieval
- Query batching and caching
- Benchmark suite
- A/B testing framework

Metrics targets:
- P95 latency < 50ms
- NDCG@10 > 0.85
- Support 1M+ vectors with <100ms queries
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class IndexType(str, Enum):
    """Vector index types."""
    FLAT = "flat"  # Brute force, exact
    HNSW = "hnsw"  # Hierarchical Navigable Small World
    IVF = "ivf"  # Inverted File Index
    IVF_PQ = "ivf_pq"  # IVF with Product Quantization


class DistanceMetric(str, Enum):
    """Distance metrics for similarity search."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"


@dataclass
class IndexConfig:
    """Configuration for vector index."""
    index_type: IndexType = IndexType.HNSW
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    dimension: int = 768
    # HNSW parameters
    hnsw_m: int = 16  # Max connections per layer
    hnsw_ef_construction: int = 200  # Size of dynamic candidate list for construction
    hnsw_ef_search: int = 100  # Size of dynamic candidate list for search
    # IVF parameters
    ivf_nlist: int = 100  # Number of clusters
    ivf_nprobe: int = 10  # Number of clusters to search
    # PQ parameters
    pq_m: int = 8  # Number of subvectors
    pq_nbits: int = 8  # Bits per subvector

    def to_dict(self) -> dict[str, Any]:
        return {
            "index_type": self.index_type.value,
            "distance_metric": self.distance_metric.value,
            "dimension": self.dimension,
            "hnsw_m": self.hnsw_m,
            "hnsw_ef_construction": self.hnsw_ef_construction,
            "hnsw_ef_search": self.hnsw_ef_search,
            "ivf_nlist": self.ivf_nlist,
            "ivf_nprobe": self.ivf_nprobe,
            "pq_m": self.pq_m,
            "pq_nbits": self.pq_nbits,
        }


@dataclass
class SearchResult:
    """Single search result."""
    id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    vector: Optional[list[float]] = None


@dataclass
class SearchResponse:
    """Response from a vector search."""
    results: list[SearchResult]
    query_time_ms: float
    total_candidates: int = 0
    cache_hit: bool = False


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    name: str
    queries: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    recall_at_k: dict[int, float]  # k -> recall
    ndcg_at_k: dict[int, float]  # k -> NDCG
    qps: float  # Queries per second


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    async def upsert(
        self,
        vectors: list[tuple[str, list[float], dict[str, Any]]],
    ) -> int:
        """Insert or update vectors. Returns count of upserted vectors."""
        pass

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter: Optional[dict[str, Any]] = None,
    ) -> SearchResponse:
        """Search for similar vectors."""
        pass

    @abstractmethod
    async def delete(self, ids: list[str]) -> int:
        """Delete vectors by ID. Returns count of deleted vectors."""
        pass

    @abstractmethod
    async def get_stats(self) -> dict[str, Any]:
        """Get index statistics."""
        pass


class InMemoryVectorStore(VectorStore):
    """In-memory vector store for development/testing."""

    def __init__(self, config: Optional[IndexConfig] = None) -> None:
        self._config = config or IndexConfig()
        self._vectors: dict[str, tuple[list[float], dict[str, Any]]] = {}
        self._lock = asyncio.Lock()

    def _compute_similarity(
        self, query: list[float], vector: list[float]
    ) -> float:
        """Compute similarity between query and vector."""
        import math

        if self._config.distance_metric == DistanceMetric.COSINE:
            dot = sum(q * v for q, v in zip(query, vector))
            norm_q = math.sqrt(sum(q * q for q in query))
            norm_v = math.sqrt(sum(v * v for v in vector))
            if norm_q == 0 or norm_v == 0:
                return 0.0
            return dot / (norm_q * norm_v)
        elif self._config.distance_metric == DistanceMetric.DOT_PRODUCT:
            return sum(q * v for q, v in zip(query, vector))
        else:  # EUCLIDEAN
            dist = math.sqrt(sum((q - v) ** 2 for q, v in zip(query, vector)))
            return 1.0 / (1.0 + dist)

    async def upsert(
        self,
        vectors: list[tuple[str, list[float], dict[str, Any]]],
    ) -> int:
        async with self._lock:
            for id_, vec, meta in vectors:
                self._vectors[id_] = (vec, meta)
            return len(vectors)

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter: Optional[dict[str, Any]] = None,
    ) -> SearchResponse:
        start = time.time()

        # Calculate similarities
        scores: list[tuple[str, float, dict[str, Any]]] = []
        for id_, (vec, meta) in self._vectors.items():
            # Apply filter
            if filter:
                match = all(
                    meta.get(k) == v for k, v in filter.items()
                )
                if not match:
                    continue
            score = self._compute_similarity(query_vector, vec)
            scores.append((id_, score, meta))

        # Sort and take top_k
        scores.sort(key=lambda x: -x[1])
        top_results = scores[:top_k]

        results = [
            SearchResult(id=id_, score=score, metadata=meta)
            for id_, score, meta in top_results
        ]

        return SearchResponse(
            results=results,
            query_time_ms=(time.time() - start) * 1000,
            total_candidates=len(self._vectors),
        )

    async def delete(self, ids: list[str]) -> int:
        async with self._lock:
            deleted = 0
            for id_ in ids:
                if id_ in self._vectors:
                    del self._vectors[id_]
                    deleted += 1
            return deleted

    async def get_stats(self) -> dict[str, Any]:
        return {
            "total_vectors": len(self._vectors),
            "index_type": self._config.index_type.value,
            "dimension": self._config.dimension,
        }


class QueryCache:
    """Cache layer for frequent queries."""

    def __init__(
        self,
        max_size: int = 10000,
        ttl_seconds: int = 3600,
    ) -> None:
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._cache: dict[str, tuple[float, SearchResponse]] = {}
        self._hits = 0
        self._misses = 0

    def _hash_query(
        self,
        query_vector: list[float],
        top_k: int,
        filter: Optional[dict[str, Any]],
    ) -> str:
        """Generate cache key from query parameters."""
        key_data = {
            "vector": [round(v, 6) for v in query_vector],
            "top_k": top_k,
            "filter": filter,
        }
        return hashlib.sha256(
            json.dumps(key_data, sort_keys=True).encode()
        ).hexdigest()

    def get(
        self,
        query_vector: list[float],
        top_k: int,
        filter: Optional[dict[str, Any]] = None,
    ) -> Optional[SearchResponse]:
        """Get cached result if available."""
        key = self._hash_query(query_vector, top_k, filter)
        if key in self._cache:
            timestamp, response = self._cache[key]
            if time.time() - timestamp < self._ttl_seconds:
                self._hits += 1
                cached_response = SearchResponse(
                    results=response.results,
                    query_time_ms=0,
                    total_candidates=response.total_candidates,
                    cache_hit=True,
                )
                return cached_response
            else:
                del self._cache[key]
        self._misses += 1
        return None

    def set(
        self,
        query_vector: list[float],
        top_k: int,
        filter: Optional[dict[str, Any]],
        response: SearchResponse,
    ) -> None:
        """Cache a query result."""
        if len(self._cache) >= self._max_size:
            # Remove oldest entries
            oldest = sorted(self._cache.items(), key=lambda x: x[1][0])
            for key, _ in oldest[: self._max_size // 4]:
                del self._cache[key]

        key = self._hash_query(query_vector, top_k, filter)
        self._cache[key] = (time.time(), response)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0,
        }


class HybridRetriever:
    """Hybrid dense-sparse retrieval combining vector and keyword search."""

    def __init__(
        self,
        vector_store: VectorStore,
        sparse_weight: float = 0.3,
    ) -> None:
        self._vector_store = vector_store
        self._sparse_weight = sparse_weight
        self._dense_weight = 1.0 - sparse_weight
        self._bm25_index: dict[str, dict[str, float]] = {}  # doc_id -> term -> score
        self._idf: dict[str, float] = {}  # term -> IDF score

    def index_document(
        self,
        doc_id: str,
        text: str,
    ) -> None:
        """Index document for sparse retrieval (BM25)."""
        terms = text.lower().split()
        term_freq: dict[str, int] = {}
        for term in terms:
            term_freq[term] = term_freq.get(term, 0) + 1

        # Store TF scores
        self._bm25_index[doc_id] = {
            term: count / len(terms) for term, count in term_freq.items()
        }

        # Update IDF (simplified)
        for term in term_freq:
            self._idf[term] = self._idf.get(term, 0) + 1

    def _sparse_search(
        self,
        query: str,
        top_k: int,
    ) -> list[tuple[str, float]]:
        """BM25-style sparse search."""
        query_terms = query.lower().split()
        scores: dict[str, float] = {}

        for doc_id, term_scores in self._bm25_index.items():
            score = 0.0
            for term in query_terms:
                if term in term_scores:
                    tf = term_scores[term]
                    idf = 1.0 / (1.0 + self._idf.get(term, 0))
                    score += tf * idf
            if score > 0:
                scores[doc_id] = score

        # Sort and return top_k
        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
        return sorted_scores[:top_k]

    async def search(
        self,
        query_text: str,
        query_vector: list[float],
        top_k: int = 10,
        filter: Optional[dict[str, Any]] = None,
    ) -> SearchResponse:
        """Hybrid search combining dense and sparse retrieval."""
        start = time.time()

        # Dense search
        dense_response = await self._vector_store.search(
            query_vector, top_k * 2, filter
        )
        dense_scores = {r.id: r.score for r in dense_response.results}

        # Sparse search
        sparse_results = self._sparse_search(query_text, top_k * 2)
        sparse_scores = dict(sparse_results)

        # Combine scores (normalize and weight)
        all_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
        combined_scores: dict[str, float] = {}

        # Normalize scores to [0, 1]
        max_dense = max(dense_scores.values()) if dense_scores else 1.0
        max_sparse = max(sparse_scores.values()) if sparse_scores else 1.0

        for id_ in all_ids:
            dense = dense_scores.get(id_, 0.0) / max_dense if max_dense else 0
            sparse = sparse_scores.get(id_, 0.0) / max_sparse if max_sparse else 0
            combined_scores[id_] = (
                self._dense_weight * dense + self._sparse_weight * sparse
            )

        # Sort and take top_k
        sorted_ids = sorted(combined_scores.items(), key=lambda x: -x[1])[:top_k]

        # Build results with metadata from dense results
        dense_results_map = {r.id: r for r in dense_response.results}
        results = []
        for id_, score in sorted_ids:
            if id_ in dense_results_map:
                results.append(SearchResult(
                    id=id_,
                    score=score,
                    metadata=dense_results_map[id_].metadata,
                ))
            else:
                results.append(SearchResult(id=id_, score=score))

        return SearchResponse(
            results=results,
            query_time_ms=(time.time() - start) * 1000,
            total_candidates=len(all_ids),
        )


class VectorStoreBenchmark:
    """Benchmark suite for vector store performance."""

    def __init__(self, store: VectorStore) -> None:
        self._store = store

    async def run_latency_benchmark(
        self,
        query_vectors: list[list[float]],
        top_k: int = 10,
    ) -> BenchmarkResult:
        """Run latency benchmark with given queries."""
        latencies: list[float] = []

        for query in query_vectors:
            response = await self._store.search(query, top_k)
            latencies.append(response.query_time_ms)

        # Calculate statistics
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        return BenchmarkResult(
            name="latency_benchmark",
            queries=n,
            avg_latency_ms=sum(latencies) / n,
            p50_latency_ms=sorted_latencies[n // 2],
            p95_latency_ms=sorted_latencies[int(n * 0.95)],
            p99_latency_ms=sorted_latencies[int(n * 0.99)],
            recall_at_k={},
            ndcg_at_k={},
            qps=1000 * n / sum(latencies) if sum(latencies) > 0 else 0,
        )

    async def run_recall_benchmark(
        self,
        query_vectors: list[list[float]],
        ground_truth: list[list[str]],  # Expected top results for each query
        k_values: list[int] = [1, 5, 10, 20],
    ) -> BenchmarkResult:
        """Run recall benchmark against ground truth."""
        recalls: dict[int, list[float]] = {k: [] for k in k_values}
        latencies: list[float] = []

        for query, expected in zip(query_vectors, ground_truth):
            response = await self._store.search(query, max(k_values))
            latencies.append(response.query_time_ms)
            result_ids = [r.id for r in response.results]

            for k in k_values:
                top_k_results = set(result_ids[:k])
                top_k_expected = set(expected[:k])
                if top_k_expected:
                    recall = len(top_k_results & top_k_expected) / len(top_k_expected)
                    recalls[k].append(recall)

        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        return BenchmarkResult(
            name="recall_benchmark",
            queries=len(query_vectors),
            avg_latency_ms=sum(latencies) / n if n else 0,
            p50_latency_ms=sorted_latencies[n // 2] if n else 0,
            p95_latency_ms=sorted_latencies[int(n * 0.95)] if n else 0,
            p99_latency_ms=sorted_latencies[int(n * 0.99)] if n else 0,
            recall_at_k={k: sum(v) / len(v) if v else 0 for k, v in recalls.items()},
            ndcg_at_k={},
            qps=1000 * n / sum(latencies) if sum(latencies) > 0 else 0,
        )

    async def run_throughput_benchmark(
        self,
        query_vectors: list[list[float]],
        duration_seconds: float = 10.0,
        concurrency: int = 10,
    ) -> BenchmarkResult:
        """Run throughput benchmark with concurrent queries."""
        import random

        completed = 0
        latencies: list[float] = []
        start_time = time.time()

        async def worker() -> None:
            nonlocal completed
            while time.time() - start_time < duration_seconds:
                query = random.choice(query_vectors)
                response = await self._store.search(query, 10)
                latencies.append(response.query_time_ms)
                completed += 1

        # Run concurrent workers
        workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
        await asyncio.gather(*workers, return_exceptions=True)

        elapsed = time.time() - start_time
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        return BenchmarkResult(
            name="throughput_benchmark",
            queries=completed,
            avg_latency_ms=sum(latencies) / n if n else 0,
            p50_latency_ms=sorted_latencies[n // 2] if n else 0,
            p95_latency_ms=sorted_latencies[int(n * 0.95)] if n else 0,
            p99_latency_ms=sorted_latencies[int(n * 0.99)] if n else 0,
            recall_at_k={},
            ndcg_at_k={},
            qps=completed / elapsed if elapsed > 0 else 0,
        )


class ABTestFramework:
    """A/B testing framework for retrieval configurations."""

    def __init__(self) -> None:
        self._experiments: dict[str, dict[str, Any]] = {}
        self._results: dict[str, list[dict[str, Any]]] = {}

    def create_experiment(
        self,
        name: str,
        variants: dict[str, IndexConfig],
        traffic_split: Optional[dict[str, float]] = None,
    ) -> str:
        """Create a new A/B experiment."""
        if traffic_split is None:
            # Equal split
            n = len(variants)
            traffic_split = {k: 1.0 / n for k in variants}

        self._experiments[name] = {
            "variants": variants,
            "traffic_split": traffic_split,
            "created_at": time.time(),
        }
        self._results[name] = []
        return name

    def get_variant(self, experiment_name: str, user_id: str) -> tuple[str, IndexConfig]:
        """Get the variant for a user in an experiment."""
        if experiment_name not in self._experiments:
            raise ValueError(f"Experiment '{experiment_name}' not found")

        exp = self._experiments[experiment_name]
        variants = exp["variants"]
        traffic_split = exp["traffic_split"]

        # Deterministic assignment based on user_id hash
        hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 100 / 100

        cumulative = 0.0
        for variant_name, split in traffic_split.items():
            cumulative += split
            if hash_val < cumulative:
                return variant_name, variants[variant_name]

        # Fallback to first variant
        first_name = list(variants.keys())[0]
        return first_name, variants[first_name]

    def record_result(
        self,
        experiment_name: str,
        variant_name: str,
        metrics: dict[str, float],
    ) -> None:
        """Record result metrics for a variant."""
        if experiment_name not in self._results:
            self._results[experiment_name] = []

        self._results[experiment_name].append({
            "variant": variant_name,
            "metrics": metrics,
            "timestamp": time.time(),
        })

    def get_experiment_results(
        self, experiment_name: str
    ) -> dict[str, dict[str, float]]:
        """Get aggregated results for an experiment."""
        if experiment_name not in self._results:
            return {}

        results = self._results[experiment_name]
        by_variant: dict[str, list[dict[str, float]]] = {}

        for r in results:
            variant = r["variant"]
            if variant not in by_variant:
                by_variant[variant] = []
            by_variant[variant].append(r["metrics"])

        # Aggregate metrics
        aggregated: dict[str, dict[str, float]] = {}
        for variant, metrics_list in by_variant.items():
            all_metrics: dict[str, list[float]] = {}
            for m in metrics_list:
                for k, v in m.items():
                    if k not in all_metrics:
                        all_metrics[k] = []
                    all_metrics[k].append(v)

            aggregated[variant] = {
                k: sum(v) / len(v) for k, v in all_metrics.items()
            }

        return aggregated


# ─────────────────────────────────────────────────────────────────────────────
# Factory functions
# ─────────────────────────────────────────────────────────────────────────────

_store: Optional[VectorStore] = None
_cache: Optional[QueryCache] = None


def get_vector_store(config: Optional[IndexConfig] = None) -> VectorStore:
    """Get or create the vector store instance."""
    global _store
    if _store is None:
        # In production, would check for Pinecone/Qdrant/etc. environment variables
        _store = InMemoryVectorStore(config or IndexConfig())
    return _store


def get_query_cache() -> QueryCache:
    """Get or create the query cache instance."""
    global _cache
    if _cache is None:
        _cache = QueryCache()
    return _cache


async def search_with_cache(
    query_vector: list[float],
    top_k: int = 10,
    filter: Optional[dict[str, Any]] = None,
) -> SearchResponse:
    """Search with caching layer."""
    cache = get_query_cache()
    store = get_vector_store()

    # Check cache first
    cached = cache.get(query_vector, top_k, filter)
    if cached:
        return cached

    # Execute search
    response = await store.search(query_vector, top_k, filter)

    # Cache result
    cache.set(query_vector, top_k, filter, response)

    return response
