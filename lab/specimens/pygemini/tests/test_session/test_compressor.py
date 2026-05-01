"""Tests for pygemini.session.compressor (ConversationCompressor)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from google.genai import types

from pygemini.core.config import Config
from pygemini.core.history import ConversationHistory
from pygemini.session.compressor import CompressionResult, ConversationCompressor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> Config:
    return Config(api_key="test-key", model="gemini-2.5-flash", fallback_model="gemini-2.5-flash")


@pytest.fixture
def compressor(config: Config) -> ConversationCompressor:
    return ConversationCompressor(config)


def _make_history(n_messages: int) -> ConversationHistory:
    """Build a ConversationHistory with n_messages alternating user/model turns."""
    history = ConversationHistory()
    for i in range(n_messages):
        if i % 2 == 0:
            history.add_user_message(f"User message {i}")
        else:
            history.add_model_response([types.Part.from_text(text=f"Model response {i}")])
    return history


# ---------------------------------------------------------------------------
# TestShouldCompress
# ---------------------------------------------------------------------------


class TestShouldCompress:
    """should_compress() should return False below the threshold."""

    def test_empty_history_returns_false(self, compressor: ConversationCompressor) -> None:
        history = ConversationHistory()
        assert compressor.should_compress(history) is False

    def test_below_threshold_returns_false(self, compressor: ConversationCompressor) -> None:
        history = _make_history(10)
        assert compressor.should_compress(history) is False

    def test_at_threshold_returns_false(self, compressor: ConversationCompressor) -> None:
        # Default threshold is 50 — exactly at 50 should not trigger
        history = _make_history(50)
        assert compressor.should_compress(history) is False

    def test_above_threshold_returns_true(self, compressor: ConversationCompressor) -> None:
        history = _make_history(51)
        assert compressor.should_compress(history) is True

    def test_far_above_threshold_returns_true(self, compressor: ConversationCompressor) -> None:
        history = _make_history(100)
        assert compressor.should_compress(history) is True


# ---------------------------------------------------------------------------
# TestFormatForSummarization
# ---------------------------------------------------------------------------


class TestFormatForSummarization:
    """_format_for_summarization() should produce readable transcripts."""

    def test_user_messages_labeled(self, compressor: ConversationCompressor) -> None:
        history = ConversationHistory()
        history.add_user_message("Hello there")
        msgs = history.get_messages()
        result = compressor._format_for_summarization(msgs)
        assert "User:" in result
        assert "Hello there" in result

    def test_model_messages_labeled(self, compressor: ConversationCompressor) -> None:
        history = ConversationHistory()
        history.add_model_response([types.Part.from_text(text="Hi, how can I help?")])
        msgs = history.get_messages()
        result = compressor._format_for_summarization(msgs)
        assert "Assistant:" in result
        assert "Hi, how can I help?" in result

    def test_function_call_included(self, compressor: ConversationCompressor) -> None:
        msg = types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="read_file", args={"path": "/foo.py"}
                    )
                )
            ],
        )
        result = compressor._format_for_summarization([msg])
        assert "read_file" in result

    def test_function_response_included(self, compressor: ConversationCompressor) -> None:
        msg = types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        name="read_file",
                        response={"output": "file contents"},
                    )
                )
            ],
        )
        result = compressor._format_for_summarization([msg])
        assert "read_file" in result

    def test_empty_parts_skipped(self, compressor: ConversationCompressor) -> None:
        msg = types.Content(role="user", parts=[])
        # Should not raise
        result = compressor._format_for_summarization([msg])
        assert isinstance(result, str)

    def test_tool_output_truncated(self, compressor: ConversationCompressor) -> None:
        long_output = "x" * 500
        msg = types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        name="read_file",
                        response={"output": long_output},
                    )
                )
            ],
        )
        result = compressor._format_for_summarization([msg])
        # The output should be truncated (200 chars + "...")
        assert "..." in result


# ---------------------------------------------------------------------------
# TestBuildSummaryPrompt
# ---------------------------------------------------------------------------


class TestBuildSummaryPrompt:
    """_build_summary_prompt() should embed the transcript."""

    def test_includes_transcript(self, compressor: ConversationCompressor) -> None:
        transcript = "User: hello\nAssistant: hi"
        prompt = compressor._build_summary_prompt(transcript)
        assert transcript in prompt

    def test_returns_string(self, compressor: ConversationCompressor) -> None:
        prompt = compressor._build_summary_prompt("some transcript")
        assert isinstance(prompt, str)

    def test_prompt_not_empty(self, compressor: ConversationCompressor) -> None:
        prompt = compressor._build_summary_prompt("transcript text")
        assert len(prompt) > 0


# ---------------------------------------------------------------------------
# TestCompress
# ---------------------------------------------------------------------------


class TestCompress:
    """compress() should return a CompressionResult with mocked LLM."""

    async def test_compress_returns_result(self, compressor: ConversationCompressor) -> None:
        history = _make_history(55)
        with patch.object(
            compressor, "_call_llm_for_summary", new=AsyncMock(return_value="Summary text here")
        ):
            result = await compressor.compress(history)

        assert isinstance(result, CompressionResult)

    async def test_compress_summary_set(self, compressor: ConversationCompressor) -> None:
        history = _make_history(55)
        with patch.object(
            compressor, "_call_llm_for_summary", new=AsyncMock(return_value="This is a summary")
        ):
            result = await compressor.compress(history)

        assert result.summary == "This is a summary"

    async def test_compress_original_message_count(self, compressor: ConversationCompressor) -> None:
        history = _make_history(55)
        with patch.object(
            compressor, "_call_llm_for_summary", new=AsyncMock(return_value="Summary")
        ):
            result = await compressor.compress(history)

        # 55 messages - 10 recent = 45 to compress
        assert result.original_message_count == 45

    async def test_compress_token_estimate_positive(
        self, compressor: ConversationCompressor
    ) -> None:
        history = _make_history(55)
        with patch.object(
            compressor,
            "_call_llm_for_summary",
            new=AsyncMock(return_value="A summary with several words in it"),
        ):
            result = await compressor.compress(history)

        assert result.compressed_token_estimate > 0

    async def test_compress_few_messages_returns_empty(
        self, compressor: ConversationCompressor
    ) -> None:
        """When history has <= keep_recent messages, return empty result."""
        history = _make_history(5)
        with patch.object(
            compressor, "_call_llm_for_summary", new=AsyncMock(return_value="Summary")
        ):
            result = await compressor.compress(history)

        assert result.summary == ""
        assert result.original_message_count == 0

    async def test_no_api_key_returns_fallback(self, tmp_path: Path) -> None:
        """Without an API key, compress falls back gracefully."""
        no_key_config = Config(api_key=None)
        comp = ConversationCompressor(no_key_config)
        history = _make_history(55)

        result = await comp.compress(history)

        assert "unavailable" in result.summary.lower() or result.summary != ""
