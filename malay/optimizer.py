from __future__ import annotations

class QueryOptimizer:
    @staticmethod
    def optimize_query(query: str) -> str:
        # Clean up and optimize query tokens for search and RAG retrieval
        return query.strip().lower()
