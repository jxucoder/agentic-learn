"""Tests for session persistence."""

import tempfile
from pathlib import Path

import pytest

from agentic_learn.core.session import SessionManager, SessionMetadata
from agentic_learn.core.types import AgentState, Message, MessageRole, ToolCall


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_sessions_dir():
    """Create a temporary directory for sessions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def session_manager(temp_sessions_dir):
    """Create a session manager with temp directory."""
    return SessionManager(sessions_dir=temp_sessions_dir)


@pytest.fixture
def sample_state():
    """Create a sample agent state."""
    state = AgentState()
    state.messages = [
        Message.user("Hello, can you help me?"),
        Message.assistant("Of course! What do you need help with?"),
        Message.user("I need to train a model."),
        Message.assistant(
            "I can help with that. Let me check your data.",
            tool_calls=[ToolCall(id="tc1", name="read", arguments={"path": "data.csv"})],
        ),
        Message.tool_response("tc1", "read", "CSV file with 1000 rows"),
        Message.assistant("I see you have a CSV file with 1000 rows. What model would you like to train?"),
    ]
    state.token_usage = {"input": 500, "output": 200}
    return state


# =============================================================================
# SessionManager Tests
# =============================================================================


class TestSessionManager:
    """Tests for SessionManager."""

    def test_init_creates_directory(self, temp_sessions_dir):
        """Test that init creates sessions directory."""
        sessions_dir = temp_sessions_dir / "new_sessions"
        sm = SessionManager(sessions_dir=sessions_dir)
        assert sessions_dir.exists()

    def test_save_creates_file(self, session_manager, sample_state):
        """Test that save creates a session file."""
        meta = session_manager.save(sample_state, name="Test Session")

        path = session_manager._get_session_path(meta.id)
        assert path.exists()

    def test_save_returns_metadata(self, session_manager, sample_state):
        """Test that save returns correct metadata."""
        meta = session_manager.save(sample_state, name="Test Session")

        assert meta.name == "Test Session"
        assert meta.message_count == 6
        assert meta.token_usage == {"input": 500, "output": 200}
        assert meta.id is not None
        assert meta.created_at is not None
        assert meta.updated_at is not None

    def test_save_auto_generates_name(self, session_manager, sample_state):
        """Test that save auto-generates name from first user message."""
        meta = session_manager.save(sample_state)

        assert "Hello, can you help me?" in meta.name

    def test_save_updates_existing(self, session_manager, sample_state):
        """Test updating an existing session."""
        meta1 = session_manager.save(sample_state, name="First Save")

        # Add more messages
        sample_state.messages.append(Message.user("Thanks!"))

        meta2 = session_manager.save(sample_state, session_id=meta1.id)

        assert meta2.id == meta1.id
        assert meta2.created_at == meta1.created_at
        assert meta2.updated_at >= meta1.updated_at
        assert meta2.message_count == 7

    def test_load_returns_state(self, session_manager, sample_state):
        """Test loading a saved session."""
        meta = session_manager.save(sample_state, name="Test Session")

        result = session_manager.load(meta.id)
        assert result is not None

        loaded_state, loaded_meta = result
        assert len(loaded_state.messages) == 6
        assert loaded_state.token_usage == {"input": 500, "output": 200}
        assert loaded_meta.id == meta.id

    def test_load_preserves_messages(self, session_manager, sample_state):
        """Test that loaded messages match original."""
        meta = session_manager.save(sample_state)
        loaded_state, _ = session_manager.load(meta.id)

        # Check first message
        assert loaded_state.messages[0].role == MessageRole.USER
        assert loaded_state.messages[0].content == "Hello, can you help me?"

        # Check message with tool calls
        assert loaded_state.messages[3].tool_calls is not None
        assert loaded_state.messages[3].tool_calls[0].name == "read"

        # Check tool response
        assert loaded_state.messages[4].role == MessageRole.TOOL
        assert loaded_state.messages[4].tool_call_id == "tc1"

    def test_load_nonexistent_returns_none(self, session_manager):
        """Test loading nonexistent session returns None."""
        result = session_manager.load("nonexistent")
        assert result is None

    def test_delete_removes_file(self, session_manager, sample_state):
        """Test deleting a session."""
        meta = session_manager.save(sample_state)
        path = session_manager._get_session_path(meta.id)

        assert path.exists()
        assert session_manager.delete(meta.id) is True
        assert not path.exists()

    def test_delete_nonexistent_returns_false(self, session_manager):
        """Test deleting nonexistent session returns False."""
        assert session_manager.delete("nonexistent") is False

    def test_list_sessions_empty(self, session_manager):
        """Test listing sessions when none exist."""
        sessions = session_manager.list_sessions()
        assert sessions == []

    def test_list_sessions_returns_all(self, session_manager, sample_state):
        """Test listing all sessions."""
        session_manager.save(sample_state, name="Session 1")
        session_manager.save(sample_state, name="Session 2")
        session_manager.save(sample_state, name="Session 3")

        sessions = session_manager.list_sessions()
        assert len(sessions) == 3

    def test_list_sessions_sorted_by_updated(self, session_manager, sample_state):
        """Test that sessions are sorted by updated_at descending."""
        meta1 = session_manager.save(sample_state, name="First")
        meta2 = session_manager.save(sample_state, name="Second")
        meta3 = session_manager.save(sample_state, name="Third")

        sessions = session_manager.list_sessions()

        # Most recent should be first
        assert sessions[0].name == "Third"
        assert sessions[2].name == "First"

    def test_list_sessions_respects_limit(self, session_manager, sample_state):
        """Test that list_sessions respects limit."""
        for i in range(10):
            session_manager.save(sample_state, name=f"Session {i}")

        sessions = session_manager.list_sessions(limit=5)
        assert len(sessions) == 5

    def test_get_metadata_returns_metadata(self, session_manager, sample_state):
        """Test getting metadata without full state."""
        meta = session_manager.save(sample_state, name="Test")

        loaded_meta = session_manager.get_metadata(meta.id)
        assert loaded_meta is not None
        assert loaded_meta.id == meta.id
        assert loaded_meta.name == "Test"

    def test_get_metadata_nonexistent_returns_none(self, session_manager):
        """Test getting metadata for nonexistent session."""
        meta = session_manager.get_metadata("nonexistent")
        assert meta is None

    def test_search_by_name(self, session_manager, sample_state):
        """Test searching sessions by name."""
        session_manager.save(sample_state, name="Training Pipeline")
        session_manager.save(sample_state, name="Data Analysis")
        session_manager.save(sample_state, name="Model Training")

        results = session_manager.search("training")
        assert len(results) == 2

    def test_search_case_insensitive(self, session_manager, sample_state):
        """Test that search is case insensitive."""
        session_manager.save(sample_state, name="UPPERCASE Session")

        results = session_manager.search("uppercase")
        assert len(results) == 1

    def test_search_respects_limit(self, session_manager, sample_state):
        """Test that search respects limit."""
        for i in range(10):
            session_manager.save(sample_state, name=f"Test {i}")

        results = session_manager.search("Test", limit=3)
        assert len(results) == 3


# =============================================================================
# SessionMetadata Tests
# =============================================================================


class TestSessionMetadata:
    """Tests for SessionMetadata."""

    def test_to_dict(self):
        """Test serializing metadata to dict."""
        meta = SessionMetadata(
            id="abc123",
            name="Test Session",
            created_at="2024-01-01T12:00:00",
            updated_at="2024-01-01T13:00:00",
            message_count=10,
            token_usage={"input": 100, "output": 50},
            summary="Test summary",
        )

        data = meta.to_dict()
        assert data["id"] == "abc123"
        assert data["name"] == "Test Session"
        assert data["message_count"] == 10

    def test_from_dict(self):
        """Test deserializing metadata from dict."""
        data = {
            "id": "abc123",
            "name": "Test Session",
            "created_at": "2024-01-01T12:00:00",
            "updated_at": "2024-01-01T13:00:00",
            "message_count": 10,
            "token_usage": {"input": 100, "output": 50},
            "summary": None,
        }

        meta = SessionMetadata.from_dict(data)
        assert meta.id == "abc123"
        assert meta.name == "Test Session"
        assert meta.token_usage == {"input": 100, "output": 50}


# =============================================================================
# Message Serialization Tests
# =============================================================================


class TestMessageSerialization:
    """Tests for message serialization."""

    def test_user_message_roundtrip(self, session_manager):
        """Test user message serialization roundtrip."""
        msg = Message.user("Hello!")
        serialized = session_manager._serialize_message(msg)
        deserialized = session_manager._deserialize_message(serialized)

        assert deserialized.role == MessageRole.USER
        assert deserialized.content == "Hello!"

    def test_assistant_message_with_tool_calls(self, session_manager):
        """Test assistant message with tool calls roundtrip."""
        msg = Message.assistant(
            "Let me check.",
            tool_calls=[
                ToolCall(id="tc1", name="read", arguments={"path": "file.py"}),
                ToolCall(id="tc2", name="bash", arguments={"command": "ls"}),
            ],
        )
        serialized = session_manager._serialize_message(msg)
        deserialized = session_manager._deserialize_message(serialized)

        assert deserialized.role == MessageRole.ASSISTANT
        assert deserialized.content == "Let me check."
        assert len(deserialized.tool_calls) == 2
        assert deserialized.tool_calls[0].name == "read"
        assert deserialized.tool_calls[1].arguments == {"command": "ls"}

    def test_tool_response_message(self, session_manager):
        """Test tool response message roundtrip."""
        msg = Message.tool_response("tc1", "read", "file contents")
        serialized = session_manager._serialize_message(msg)
        deserialized = session_manager._deserialize_message(serialized)

        assert deserialized.role == MessageRole.TOOL
        assert deserialized.tool_call_id == "tc1"
        assert deserialized.name == "read"
        assert deserialized.content == "file contents"
