"""Session persistence for saving and loading agent state."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from agentic_learn.core.types import AgentState, Message, MessageRole, ToolCall


@dataclass
class SessionMetadata:
    """Metadata about a saved session."""

    id: str
    name: str
    created_at: str
    updated_at: str
    message_count: int
    token_usage: dict[str, int]
    summary: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionMetadata:
        return cls(**data)


@dataclass
class SessionManager:
    """Manages saving and loading agent sessions.

    Sessions are stored as JSON files in the sessions directory.
    Each session contains:
    - Metadata (id, name, timestamps, summary)
    - Messages (full conversation history)
    - Token usage statistics
    """

    sessions_dir: Path = field(default_factory=lambda: Path.home() / ".agentic-learn" / "sessions")

    def __post_init__(self) -> None:
        """Ensure sessions directory exists."""
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _get_session_path(self, session_id: str) -> Path:
        """Get path for a session file."""
        return self.sessions_dir / f"{session_id}.json"

    def _serialize_message(self, msg: Message) -> dict[str, Any]:
        """Serialize a Message to dict."""
        data: dict[str, Any] = {
            "role": msg.role.value,
            "content": msg.content,
        }
        if msg.tool_calls:
            data["tool_calls"] = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in msg.tool_calls
            ]
        if msg.tool_call_id:
            data["tool_call_id"] = msg.tool_call_id
        if msg.name:
            data["name"] = msg.name
        return data

    def _deserialize_message(self, data: dict[str, Any]) -> Message:
        """Deserialize a dict to Message."""
        tool_calls = None
        if "tool_calls" in data:
            tool_calls = [
                ToolCall(id=tc["id"], name=tc["name"], arguments=tc["arguments"])
                for tc in data["tool_calls"]
            ]

        return Message(
            role=MessageRole(data["role"]),
            content=data["content"],
            tool_calls=tool_calls,
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
        )

    def save(
        self,
        state: AgentState,
        name: str | None = None,
        session_id: str | None = None,
    ) -> SessionMetadata:
        """Save agent state to a session file.

        Args:
            state: The agent state to save
            name: Optional name for the session
            session_id: Optional ID to update existing session

        Returns:
            SessionMetadata for the saved session
        """
        now = datetime.now().isoformat()

        # Generate or use existing ID
        if session_id is None:
            session_id = str(uuid.uuid4())[:8]
            created_at = now
        else:
            # Load existing metadata for created_at
            existing = self.get_metadata(session_id)
            created_at = existing.created_at if existing else now

        # Generate name if not provided
        if name is None:
            # Use first user message as name, or timestamp
            for msg in state.messages:
                if msg.role == MessageRole.USER:
                    name = msg.content[:50] + ("..." if len(msg.content) > 50 else "")
                    break
            if name is None:
                name = f"Session {now[:10]}"

        # Generate summary from last few messages
        summary = self._generate_summary(state.messages)

        metadata = SessionMetadata(
            id=session_id,
            name=name,
            created_at=created_at,
            updated_at=now,
            message_count=len(state.messages),
            token_usage=state.token_usage.copy(),
            summary=summary,
        )

        # Build session data
        session_data = {
            "metadata": metadata.to_dict(),
            "messages": [self._serialize_message(msg) for msg in state.messages],
        }

        # Write to file
        path = self._get_session_path(session_id)
        with open(path, "w") as f:
            json.dump(session_data, f, indent=2)

        return metadata

    def load(self, session_id: str) -> tuple[AgentState, SessionMetadata] | None:
        """Load a session by ID.

        Args:
            session_id: The session ID to load

        Returns:
            Tuple of (AgentState, SessionMetadata) or None if not found
        """
        path = self._get_session_path(session_id)
        if not path.exists():
            return None

        with open(path) as f:
            data = json.load(f)

        metadata = SessionMetadata.from_dict(data["metadata"])
        messages = [self._deserialize_message(msg) for msg in data["messages"]]

        state = AgentState(
            messages=messages,
            token_usage=metadata.token_usage.copy(),
        )

        return state, metadata

    def delete(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: The session ID to delete

        Returns:
            True if deleted, False if not found
        """
        path = self._get_session_path(session_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_sessions(self, limit: int = 20) -> list[SessionMetadata]:
        """List all saved sessions, sorted by most recent.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of SessionMetadata, sorted by updated_at descending
        """
        sessions = []

        for path in self.sessions_dir.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                metadata = SessionMetadata.from_dict(data["metadata"])
                sessions.append(metadata)
            except (json.JSONDecodeError, KeyError):
                continue

        # Sort by updated_at descending
        sessions.sort(key=lambda s: s.updated_at, reverse=True)

        return sessions[:limit]

    def get_metadata(self, session_id: str) -> SessionMetadata | None:
        """Get metadata for a session without loading full state.

        Args:
            session_id: The session ID

        Returns:
            SessionMetadata or None if not found
        """
        path = self._get_session_path(session_id)
        if not path.exists():
            return None

        with open(path) as f:
            data = json.load(f)

        return SessionMetadata.from_dict(data["metadata"])

    def search(self, query: str, limit: int = 10) -> list[SessionMetadata]:
        """Search sessions by name or summary.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            Matching sessions
        """
        query_lower = query.lower()
        results = []

        for session in self.list_sessions(limit=100):
            if (
                query_lower in session.name.lower()
                or (session.summary and query_lower in session.summary.lower())
            ):
                results.append(session)
                if len(results) >= limit:
                    break

        return results

    def _generate_summary(self, messages: list[Message], max_length: int = 100) -> str | None:
        """Generate a brief summary from recent messages."""
        if not messages:
            return None

        # Get last assistant message
        for msg in reversed(messages):
            if msg.role == MessageRole.ASSISTANT and msg.content:
                summary = msg.content[:max_length]
                if len(msg.content) > max_length:
                    summary += "..."
                return summary

        return None
