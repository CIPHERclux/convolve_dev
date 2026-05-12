"""Chat endpoints — text and audio."""

import os
import uuid
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from app.models.schemas import TextChatRequest, ChatResponse
from app.config import settings
from app.utils.media_processor import MediaProcessor
from app.utils.logger import get_logger
from openai import OpenAI

log = get_logger("api.chat")

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

# Will be injected by dependencies
_orchestrator = None
_media = MediaProcessor()

def set_orchestrator(orch):
    global _orchestrator
    _orchestrator = orch


@router.post("/text", response_model=ChatResponse)
async def chat_text(request: TextChatRequest):
    """Process a text-only message."""
    if not _orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")

    session = _orchestrator.session_mgr.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return await _orchestrator.process_text_turn(
        session_id=request.session_id,
        text=request.text,
    )


@router.post("/audio", response_model=ChatResponse)
async def chat_audio(
    session_id: str = Form(...),
    text: str = Form(default=""),
    audio: UploadFile = File(...),
):
    """Process an audio file upload with optional text."""
    if not _orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")

    session = _orchestrator.session_mgr.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Save uploaded file
    ext = os.path.splitext(audio.filename or "upload.wav")[1]
    temp_path = os.path.join(settings.UPLOAD_DIR, f"{uuid.uuid4()}{ext}")

    try:
        content = await audio.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        # Load audio
        audio_array, sr = _media.load_audio_file(temp_path)

        if audio_array is None:
            raise HTTPException(status_code=400, detail="Could not process audio file")

        # Transcribe audio using Groq Whisper if text is empty
        user_text = text.strip()
        if not user_text:
            try:
                # Use Groq for free fast transcription
                client = OpenAI(
                    api_key=settings.GROQ_API_KEY or settings.LLM_API_KEY,
                    base_url="https://api.groq.com/openai/v1" if settings.GROQ_API_KEY else None
                )
                with open(temp_path, "rb") as af:
                    transcription = client.audio.transcriptions.create(
                        model="whisper-large-v3" if settings.GROQ_API_KEY else "whisper-1",
                        file=af,
                        response_format="text"
                    )
                user_text = transcription.strip()
                log.info(f"Transcribed audio: '{user_text}'")
            except Exception as e:
                log.error(f"Transcription failed: {e}")
                user_text = "[Audio message could not be transcribed]"

        return await _orchestrator.process_audio_turn(
            session_id=session_id,
            text=user_text,
            audio_array=audio_array,
        )

    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
