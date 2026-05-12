"""Session management endpoints."""

from fastapi import APIRouter, HTTPException

from app.models.schemas import SessionStartRequest, SessionStartResponse

router = APIRouter(prefix="/api/v1/session", tags=["session"])

# Will be injected by dependencies
_orchestrator = None

def set_orchestrator(orch):
    global _orchestrator
    _orchestrator = orch


@router.post("/start", response_model=SessionStartResponse)
async def start_session(request: SessionStartRequest):
    """Start a new chat session."""
    session = _orchestrator.session_mgr.create_session(request.user_id)
    return SessionStartResponse(
        session_id=session.session_id,
        user_id=session.user_id,
        message="Session started. How are you feeling today?",
    )


@router.delete("/{session_id}")
async def end_session(session_id: str):
    """End a chat session."""
    # Save baselines before ending
    state = _orchestrator._user_state
    for uid, s in state.items():
        s["baseline"].force_save()

    success = _orchestrator.session_mgr.end_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session ended", "session_id": session_id}
