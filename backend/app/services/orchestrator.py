"""
Orchestration Service — thin coordinator (~200 lines, down from 2054).

Routes requests through services. No business logic lives here.
"""

import asyncio

import numpy as np

from app.config import settings
from app.extraction.feature_engine import FeatureEngine
from app.memory.baseline_manager import BaselineManager
from app.memory.biomarker_tracker import BiomarkerTracker
from app.memory.memory_controller import MemoryController
from app.memory.user_profile import UserProfile
from app.models.model_registry import ModelRegistry
from app.models.schemas import (
    ChatResponse,
    ToolCall,
)
from app.services.llm_service import LLMService
from app.services.safety_service import SafetyService
from app.services.session_manager import Session, SessionManager
from app.utils.logger import get_logger

log = get_logger("services.orchestrator")


class OrchestrationService:
    """
    Thin coordinator — ~200 lines instead of 2054.

    Flow:
        1. Extract features (semantic + biomarkers)
        2. Safety check (deterministic, fast)
        3. Self-query → retrieve memory context
        4. Single LLM call (response + tool calls)
        5. Execute tool calls
        6. Store turn (fire-and-forget)
        7. Episodic rollup check
    """

    def __init__(self):
        self.feature_engine = FeatureEngine()
        self.llm = LLMService()
        self.safety = SafetyService()
        self.session_mgr = SessionManager()

        # Per-user state (single-user MVP)
        self._user_state = {}

        # Store strong references to background tasks (RUF006)
        self._background_tasks = set()

        log.info("OrchestrationService initialized")

    def _get_user_state(self, user_id: str) -> dict:
        """Get or create per-user state objects."""
        if user_id not in self._user_state:
            self._user_state[user_id] = {
                "memory": MemoryController(user_id),
                "tracker": BiomarkerTracker(),
                "baseline": BaselineManager(user_id),
                "profile": UserProfile(user_id),
            }
        return self._user_state[user_id]

    async def process_text_turn(
        self, session_id: str, text: str
    ) -> ChatResponse:
        """Process a text-only turn."""
        return await self._process_turn(
            session_id=session_id,
            text=text,
            audio_array=None,
            modality="text",
        )

    async def process_audio_turn(
        self, session_id: str, text: str, audio_array: np.ndarray
    ) -> ChatResponse:
        """Process an audio turn (text = transcription or user-provided)."""
        return await self._process_turn(
            session_id=session_id,
            text=text,
            audio_array=audio_array,
            modality="audio",
        )

    async def _process_turn(
        self,
        session_id: str,
        text: str,
        audio_array: np.ndarray | None,
        modality: str,
    ) -> ChatResponse:
        """Core processing pipeline."""
        session = self.session_mgr.get_session(session_id)
        if not session:
            return ChatResponse(
                session_id=session_id, response_text="Session not found.", turn_number=0
            )

        state = self._get_user_state(session.user_id)
        memory: MemoryController = state["memory"]
        tracker: BiomarkerTracker = state["tracker"]
        baseline: BaselineManager = state["baseline"]
        profile: UserProfile = state["profile"]

        session.add_user_message(text)

        # ── 1. Feature extraction (CPU-bound — offloaded to thread pool) ───
        extraction = await asyncio.to_thread(
            self.feature_engine.extract,
            text=text,
            audio_array=audio_array,
            modality=modality,
            last_system_end_time=session.last_system_end_time,
        )

        # Update baseline + tracker
        raw_vec = np.array(extraction.raw_vector, dtype=np.float32)
        baseline.update(raw_vec)
        tracker.add_turn(extraction.raw_vector, modality, session.turn_number)

        # ── 2. Safety check (fast, deterministic) ────────────────────────
        safety_result = self.safety.check(text, extraction.biomarkers)

        # ── 3. Profile extraction ────────────────────────────────────────
        profile.extract_facts(text)

        # ── 4. Self-querying → memory retrieval ──────────────────────────
        memory_text = ""
        if memory.is_connected():
            search_query, filters = await self.llm.generate_search_query(text)

            # Encode the optimized query
            encoder = ModelRegistry.get_semantic_encoder()
            query_embedding = encoder.encode(search_query, convert_to_numpy=True).tolist()

            # Search with optional emotion filter
            filters.get("emotion", None)
            context = memory.build_context(query_embedding, limit=5)

            # Format memory context for LLM
            parts = []
            for mem in context.get("relevant_turns", []):
                parts.append(
                    f"[Turn {mem.get('turn_number', '?')}, {mem.get('emotion_tag', '')}] "
                    f"User: {mem['text']}"
                )
                if mem.get("system_response"):
                    parts.append(f"  Assistant: {mem['system_response'][:200]}")
            for ep in context.get("episodic_summaries", []):
                parts.append(f"[Episode] {ep['text']}")
            memory_text = "\n".join(parts)

        # ── 5. Single LLM call ──────────────────────────────────────────
        response_text, tool_calls, emotion_tag = await self.llm.generate_response(
            user_text=text,
            conversation_history=session.get_recent_history(),
            biomarker_summary=tracker.get_summary_for_llm(),
            memory_context=memory_text,
            profile_summary=profile.get_summary_for_llm(),
            safety=safety_result,
        )

        session.add_assistant_message(response_text)

        # ── 6. Execute tool calls ────────────────────────────────────────
        for tc in tool_calls:
            self._execute_tool(tc, profile)

        # ── 7. Store in memory (fire-and-forget) ─────────────────────────
        if memory.is_connected():
            memory.store_turn(
                text=text,
                embedding=extraction.semantic_embedding,
                biomarkers=extraction.biomarkers,
                turn_number=session.turn_number,
                session_id=session_id,
                emotion_tag=emotion_tag,
                system_response=response_text,
            )

        # ── 8. Episodic rollup check ────────────────────────────────────
        if (
            session.turn_number > 0
            and session.turn_number % settings.EPISODIC_ROLLUP_INTERVAL == 0
            and memory.is_connected()
        ):
            task = asyncio.create_task(
                self._create_episodic_summary(session, memory)
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        # ── 9. Masking detection ─────────────────────────────────────────
        masking = tracker.detect_masking()

        return ChatResponse(
            session_id=session_id,
            response_text=response_text,
            turn_number=session.turn_number,
            user_transcription=text if modality == "audio" else None,
            biomarkers=extraction.biomarkers if modality != "text" else None,
            masking_detected=masking["detected"],
            masking_details=masking if masking["detected"] else None,
            safety=safety_result if safety_result.is_crisis else None,
            tool_calls=tool_calls,
            emotion_tag=emotion_tag,
        )

    def _execute_tool(self, tc: ToolCall, profile: UserProfile):
        """Execute a tool call from the LLM."""
        try:
            if tc.name == "update_user_profile":
                fact_type = tc.arguments.get("fact_type", "")
                value = tc.arguments.get("value", "")
                if fact_type == "relationship":
                    rel_type = tc.arguments.get("relationship_type", "other")
                    profile.set_fact("relationships", {rel_type: value})
                elif fact_type in ("name", "age", "location", "occupation", "school"):
                    profile.set_fact(fact_type, value)
                elif fact_type == "diagnosis":
                    diags = profile.facts.get("diagnoses", [])
                    if value.lower() not in [d.lower() for d in diags]:
                        diags.append(value)
                        profile.set_fact("diagnoses", diags)
                else:
                    profile.set_fact(fact_type, value)
                log.info(f"Profile updated: {fact_type}={value}")

            elif tc.name == "log_trigger":
                log.info(f"Trigger logged: {tc.arguments}")

            elif tc.name == "flag_crisis":
                log.warning(f"LLM flagged crisis: {tc.arguments}")

        except Exception as e:
            log.warning(f"Tool execution error ({tc.name}): {e}")

    async def _create_episodic_summary(self, session: Session, memory: MemoryController):
        """Create an episodic summary from recent conversation block."""
        try:
            block = session.get_conversation_block(settings.EPISODIC_ROLLUP_INTERVAL * 2)
            if not block:
                return

            summary = await self.llm.generate_episodic_summary(block)
            if not summary:
                return

            # Encode the summary
            encoder = ModelRegistry.get_semantic_encoder()
            embedding = encoder.encode(summary, convert_to_numpy=True).tolist()

            # Store as episodic summary
            turn_range = f"turns {max(1, session.turn_number - settings.EPISODIC_ROLLUP_INTERVAL)}-{session.turn_number}"
            memory.store_episodic_summary(
                summary_text=summary,
                embedding=embedding,
                session_id=session.session_id,
                turn_range=turn_range,
            )
        except Exception as e:
            log.error(f"Episodic summarization failed: {e}")
