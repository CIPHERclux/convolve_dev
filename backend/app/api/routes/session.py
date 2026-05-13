"""Session management endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_orchestrator
from app.models.schemas import SessionStartRequest, SessionStartResponse
from app.services.orchestrator import OrchestrationService

router = APIRouter(prefix="/api/v1/session", tags=["session"])


@router.post("/start", response_model=SessionStartResponse)
async def start_session(
    request: SessionStartRequest,
    orchestrator: OrchestrationService = Depends(get_orchestrator),
):
    """Start a new chat session."""
    session = orchestrator.session_mgr.create_session(request.user_id)
    return SessionStartResponse(
        session_id=session.session_id,
        user_id=session.user_id,
        message="Session started. How are you feeling today?",
    )


@router.delete("/{session_id}")
async def end_session(
    session_id: str,
    orchestrator: OrchestrationService = Depends(get_orchestrator),
):
    """End a chat session."""
    # Save baselines before ending
    for _uid, s in orchestrator._user_state.items():
        s["baseline"].force_save()

    success = orchestrator.session_mgr.end_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session ended", "session_id": session_id}
