"""Profile endpoints."""

from fastapi import APIRouter, HTTPException

from app.models.schemas import ProfileResponse

router = APIRouter(prefix="/api/v1/profile", tags=["profile"])

_orchestrator = None

def set_orchestrator(orch):
    global _orchestrator
    _orchestrator = orch


@router.get("/{user_id}", response_model=ProfileResponse)
async def get_profile(user_id: str):
    """Get user profile."""
    state = _orchestrator._get_user_state(user_id)
    profile = state["profile"]
    return ProfileResponse(
        user_id=user_id,
        facts=profile.get_all_facts(),
        stats={"total_interactions": profile.facts.get("total_interactions", 0)},
    )
