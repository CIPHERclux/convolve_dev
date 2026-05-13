"""
Dependency Injection — FastAPI Depends() providers.

Replaces the global `_orchestrator = None` + `set_orchestrator()` pattern
with request-scoped dependency injection. This makes route handlers:
  1. Testable (swap the orchestrator in tests)
  2. Thread-safe (no mutable module-level state)
  3. Self-documenting (dependencies are explicit in function signatures)
"""

from fastapi import HTTPException, Request

from app.services.orchestrator import OrchestrationService


def get_orchestrator(request: Request) -> OrchestrationService:
    """
    Retrieve the OrchestrationService from application state.

    Injected via FastAPI's `Depends()` mechanism. The orchestrator
    is initialized once during the app lifespan and stored on
    `app.state.orchestrator`.

    Raises:
        HTTPException(503): If the orchestrator hasn't been initialized yet
                            (e.g., startup still in progress).
    """
    orchestrator: OrchestrationService | None = getattr(
        request.app.state, "orchestrator", None
    )
    if orchestrator is None:
        raise HTTPException(
            status_code=503,
            detail="Service is starting up. Please retry in a moment.",
        )
    return orchestrator
