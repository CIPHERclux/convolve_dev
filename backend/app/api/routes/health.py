"""Health check endpoint."""

from fastapi import APIRouter, Depends

from app.models.schemas import HealthResponse
from app.models.model_registry import ModelRegistry
from app.config import settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check — verifies core services."""
    qdrant_ok = False
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT, timeout=3)
        client.get_collections()
        qdrant_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        qdrant_connected=qdrant_ok,
        models_loaded=ModelRegistry.loaded_models(),
    )
