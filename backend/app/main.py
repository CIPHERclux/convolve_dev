"""
Convolve MAS — FastAPI Backend Entry Point

Run with: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat, health, profile, session
from app.config import settings
from app.models.model_registry import ModelRegistry
from app.services.orchestrator import OrchestrationService
from app.utils.logger import get_logger

log = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    log.info("=" * 60)
    log.info(f"  {settings.APP_NAME} v{settings.APP_VERSION}")
    log.info(f"  Model: {settings.LLM_MODEL}")
    if settings.LLM_BASE_URL:
        log.info(f"  Provider: {settings.LLM_BASE_URL}")
    log.info("=" * 60)

    if not settings.LLM_API_KEY:
        log.warning("⚠️  NO LLM API KEY FOUND!")
        log.warning("   Set one of: LLM_API_KEY, OPENAI_API_KEY, or GROQ_API_KEY")
        log.warning("   Create backend/.env with: LLM_API_KEY=sk-your-key-here")
        log.warning("   LLM responses will fall back to a default message.")

    # Pre-load models
    log.info("Loading ML models…")
    ModelRegistry.get_semantic_encoder()
    ModelRegistry.get_vader()

    # Initialize orchestrator and store on app state (Dependency Injection)
    app.state.orchestrator = OrchestrationService()

    log.info("🟢 Server ready")
    yield

    # Shutdown
    log.info("Shutting down…")
    orchestrator: OrchestrationService = app.state.orchestrator
    for _uid, state in orchestrator._user_state.items():
        state["baseline"].force_save()
    log.info("🔴 Server stopped")


# ── App ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# CORS — restricted to configured origins (no wildcard in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(health.router)
app.include_router(session.router)
app.include_router(chat.router)
app.include_router(profile.router)
