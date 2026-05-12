"""
Session Manager — manages session state across API calls.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

from app.utils.logger import get_logger

log = get_logger("services.session")


class Session:
    """A single chat session."""

    def __init__(self, session_id: str, user_id: str):
        self.session_id = session_id
        self.user_id = user_id
        self.turn_number = 0
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.history: List[Dict[str, str]] = []  # {role, content}
        self.last_system_end_time: Optional[str] = None

    def add_user_message(self, text: str):
        self.history.append({"role": "user", "content": text})
        self.turn_number += 1
        self.last_activity = datetime.now()

    def add_assistant_message(self, text: str):
        self.history.append({"role": "assistant", "content": text})
        self.last_system_end_time = datetime.now().isoformat()
        self.last_activity = datetime.now()

    def get_recent_history(self, n: int = 20) -> List[Dict[str, str]]:
        return self.history[-n:]

    def get_conversation_block(self, n: int = 20) -> List[Dict[str, str]]:
        """Get last N messages for episodic summarization."""
        return self.history[-n:]


class SessionManager:
    """Manages active chat sessions."""

    def __init__(self):
        self.sessions: Dict[str, Session] = {}

    def create_session(self, user_id: str) -> Session:
        session_id = str(uuid.uuid4())[:8]
        session = Session(session_id=session_id, user_id=user_id)
        self.sessions[session_id] = session
        log.info(f"Session created: {session_id} for user {user_id}")
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        return self.sessions.get(session_id)

    def end_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            del self.sessions[session_id]
            log.info(f"Session ended: {session_id}")
            return True
        return False

    def get_active_sessions(self) -> List[str]:
        return list(self.sessions.keys())
