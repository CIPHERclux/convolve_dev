"""
Memory Controller — Qdrant interaction layer.

KEY ARCHITECTURAL CHANGES (Phase 1 + 2):
  • SINGLE vector search (384-dim semantic embedding only)
  • Biomarkers stored as payload metadata (NOT searched on)
  • Graph-RAG killed — replaced with episodic summarization
  • ColBERT/multi-vector/trajectory vectors removed
  • New Qdrant schema: embedding[384] + rich payload
"""

import uuid
from datetime import datetime
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.config import settings
from app.models.schemas import BiomarkerPayload
from app.utils.logger import get_logger

log = get_logger("memory.controller")


class MemoryController:
    """
    Lean memory layer backed by Qdrant.

    Schema:
        vector: 384-dim semantic embedding (ONLY vector used for search)
        payload:
            text:         str   — original user text
            type:         str   — "turn" | "episodic_summary"
            biomarkers:   dict  — full biomarker telemetry (metadata)
            emotion_tag:  str   — LLM-extracted emotion label
            timestamp:    str   — ISO timestamp
            turn_number:  int
            session_id:   str
            user_id:      str
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.collection = settings.QDRANT_COLLECTION
        self.client: Optional[QdrantClient] = None
        self._connect()

    def _connect(self):
        """Connect to Qdrant (Docker container)."""
        try:
            self.client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                timeout=10,
            )
            self._ensure_collection()
            log.info(f"Connected to Qdrant at {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
        except Exception as e:
            log.warning(f"Qdrant connection failed: {e} — running without persistent memory")
            self.client = None

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        if self.client is None:
            return
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection not in collections:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=settings.SEMANTIC_DIM,
                    distance=Distance.COSINE,
                ),
            )
            log.info(f"Created collection: {self.collection}")

    def is_connected(self) -> bool:
        return self.client is not None

    # ── Store ────────────────────────────────────────────────────────────

    def store_turn(
        self,
        text: str,
        embedding: list[float],
        biomarkers: BiomarkerPayload,
        turn_number: int,
        session_id: str,
        emotion_tag: str = "neutral",
        system_response: str = "",
    ):
        """Store a conversation turn in Qdrant."""
        if self.client is None or not embedding:
            return

        point_id = str(uuid.uuid4())
        payload = {
            "text": text,
            "system_response": system_response,
            "type": "turn",
            "biomarkers": biomarkers.model_dump(),
            "emotion_tag": emotion_tag,
            "timestamp": datetime.now().isoformat(),
            "turn_number": turn_number,
            "session_id": session_id,
            "user_id": self.user_id,
        }

        try:
            self.client.upsert(
                collection_name=self.collection,
                points=[PointStruct(id=point_id, vector=embedding, payload=payload)],
            )
            log.debug(f"Stored turn {turn_number} (emotion={emotion_tag})")
        except Exception as e:
            log.error(f"Failed to store turn: {e}")

    def store_episodic_summary(
        self,
        summary_text: str,
        embedding: list[float],
        session_id: str,
        turn_range: str = "",
    ):
        """Store an episodic summary (Phase 2 — replaces Graph-RAG)."""
        if self.client is None or not embedding:
            return

        point_id = str(uuid.uuid4())
        payload = {
            "text": summary_text,
            "type": "episodic_summary",
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "user_id": self.user_id,
            "turn_range": turn_range,
            "biomarkers": {},
            "emotion_tag": "summary",
        }

        try:
            self.client.upsert(
                collection_name=self.collection,
                points=[PointStruct(id=point_id, vector=embedding, payload=payload)],
            )
            log.info(f"Stored episodic summary ({turn_range})")
        except Exception as e:
            log.error(f"Failed to store episodic summary: {e}")

    # ── Retrieve ─────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: list[float],
        limit: int = 5,
        score_threshold: float = 0.3,
        type_filter: Optional[str] = None,
        emotion_filter: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Semantic search — ONLY uses embedding distance.
        Filters (type, emotion) are applied as metadata filters.
        """
        if self.client is None or not query_embedding:
            return []

        # Build filters
        must_conditions = [FieldCondition(key="user_id", match=MatchValue(value=self.user_id))]
        if type_filter:
            must_conditions.append(FieldCondition(key="type", match=MatchValue(value=type_filter)))
        if emotion_filter:
            must_conditions.append(
                FieldCondition(key="emotion_tag", match=MatchValue(value=emotion_filter))
            )

        query_filter = Filter(must=must_conditions)

        try:
            results = self.client.search(
                collection_name=self.collection,
                query_vector=query_embedding,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
            )

            memories = []
            for hit in results:
                memories.append(
                    {
                        "text": hit.payload.get("text", ""),
                        "system_response": hit.payload.get("system_response", ""),
                        "type": hit.payload.get("type", "turn"),
                        "score": hit.score,
                        "emotion_tag": hit.payload.get("emotion_tag", ""),
                        "timestamp": hit.payload.get("timestamp", ""),
                        "biomarkers": hit.payload.get("biomarkers", {}),
                        "turn_number": hit.payload.get("turn_number", 0),
                    }
                )

            log.debug(f"Search returned {len(memories)} results")
            return memories

        except Exception as e:
            log.error(f"Search failed: {e}")
            return []

    def build_context(
        self,
        query_embedding: list[float],
        limit: int = 5,
    ) -> dict[str, Any]:
        """
        Build retrieval context for the LLM.
        Searches both turns and episodic summaries.
        """
        # Get recent turns
        turns = self.search(query_embedding, limit=limit, type_filter="turn")

        # Get relevant episodic summaries
        episodes = self.search(query_embedding, limit=3, type_filter="episodic_summary")

        return {
            "relevant_turns": turns,
            "episodic_summaries": episodes,
        }

    def get_stats(self) -> dict[str, Any]:
        if self.client is None:
            return {"connected": False}
        try:
            info = self.client.get_collection(self.collection)
            return {
                "connected": True,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
            }
        except Exception:
            return {"connected": False}
