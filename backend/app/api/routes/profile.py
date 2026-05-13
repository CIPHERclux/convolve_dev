"""Profile endpoints."""

from fastapi import APIRouter, Depends

from app.api.dependencies import get_orchestrator
from app.models.schemas import ProfileResponse
from app.services.orchestrator import OrchestrationService

router = APIRouter(prefix="/api/v1/profile", tags=["profile"])


@router.get("/{user_id}", response_model=ProfileResponse)
async def get_profile(
    user_id: str,
    orchestrator: OrchestrationService = Depends(get_orchestrator),
):
    """Get user profile."""
    state = orchestrator._get_user_state(user_id)
    profile = state["profile"]
    return ProfileResponse(
        user_id=user_id,
        facts=profile.get_all_facts(),
        stats={"total_interactions": profile.facts.get("total_interactions", 0)},
    )
