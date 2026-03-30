"""Extended tests for pygemini.core.history (token/slice/mutation methods)."""

from __future__ import annotations

import pytest
from google.genai import types

from pygemini.core.history import ConversationHistory, estimate_tokens


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_history(*texts: str) -> ConversationHistory:
    """Build a history from alternating user/model text messages."""
    h = ConversationHistory()
    for i, text in enumerate(texts):
        if i % 2 == 0:
            h.add_user_message(text)
        else:
            h.add_model_response([types.Part.from_text(text=text)])
    return h


# ---------------------------------------------------------------------------
# TestTokenCount
# ---------------------------------------------------------------------------


class TestTokenCount:
    """token_count property should estimate total tokens across all messages."""

    def test_empty_history_has_zero_tokens(self) -> None:
        h = ConversationHistory()
        assert h.token_count == 0

    def test_single_user_message_has_tokens(self) -> None:
        h = ConversationHistory()
        h.add_user_message("hello world")
        assert h.token_count > 0

    def test_token_count_increases_with_messages(self) -> None:
        h = ConversationHistory()
        h.add_user_message("short")
        count_1 = h.token_count
        h.add_model_response([types.Part.from_text(text="a much longer response")])
        count_2 = h.token_count
        assert count_2 > count_1

    def test_token_count_reflects_text_length(self) -> None:
        h = ConversationHistory()
        # 40-char string → estimate_tokens("x" * 40) = 10
        h.add_user_message("a" * 40)
        assert h.token_count == 10

    def test_function_call_contributes_tokens(self) -> None:
        h = ConversationHistory()
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
        h._messages.append(msg)
        assert h.token_count > 0

    def test_function_response_contributes_tokens(self) -> None:
        h = ConversationHistory()
        msg = types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        name="read_file",
                        response={"output": "file content here"},
                    )
                )
            ],
        )
        h._messages.append(msg)
        assert h.token_count > 0


# ---------------------------------------------------------------------------
# TestMessageCount
# ---------------------------------------------------------------------------


class TestMessageCount:
    """message_count property should track total messages accurately."""

    def test_empty_history_zero(self) -> None:
        h = ConversationHistory()
        assert h.message_count == 0

    def test_one_user_message(self) -> None:
        h = ConversationHistory()
        h.add_user_message("hi")
        assert h.message_count == 1

    def test_user_and_model_messages(self) -> None:
        h = _make_history("user msg", "model msg")
        assert h.message_count == 2

    def test_increments_on_add(self) -> None:
        h = ConversationHistory()
        for i in range(5):
            h.add_user_message(f"msg {i}")
        assert h.message_count == 5

    def test_matches_len(self) -> None:
        h = _make_history("a", "b", "c")
        assert h.message_count == len(h)

    def test_resets_after_clear(self) -> None:
        h = _make_history("a", "b", "c")
        h.clear()
        assert h.message_count == 0


# ---------------------------------------------------------------------------
# TestGetMessagesRange
# ---------------------------------------------------------------------------


class TestGetMessagesRange:
    """get_messages_range() should return correct message slices."""

    def test_full_range(self) -> None:
        h = _make_history("a", "b", "c")
        result = h.get_messages_range(0, 3)
        assert len(result) == 3

    def test_partial_range(self) -> None:
        h = _make_history("a", "b", "c", "d")
        result = h.get_messages_range(1, 3)
        assert len(result) == 2

    def test_range_to_end(self) -> None:
        h = _make_history("a", "b", "c")
        result = h.get_messages_range(1)
        assert len(result) == 2

    def test_range_from_start(self) -> None:
        h = _make_history("a", "b", "c")
        result = h.get_messages_range(0, 2)
        assert len(result) == 2

    def test_returns_new_list(self) -> None:
        h = _make_history("a", "b")
        result = h.get_messages_range(0, 2)
        result.clear()
        # Original history should be unaffected
        assert len(h) == 2

    def test_negative_start(self) -> None:
        h = _make_history("a", "b", "c")
        result = h.get_messages_range(-1)
        assert len(result) == 1

    def test_empty_range(self) -> None:
        h = _make_history("a", "b")
        result = h.get_messages_range(1, 1)
        assert result == []

    def test_content_preserved(self) -> None:
        h = ConversationHistory()
        h.add_user_message("specific text")
        result = h.get_messages_range(0, 1)
        assert result[0].parts[0].text == "specific text"


# ---------------------------------------------------------------------------
# TestReplaceMessages
# ---------------------------------------------------------------------------


class TestReplaceMessages:
    """replace_messages() should splice messages in and out correctly."""

    def test_replace_all_with_single(self) -> None:
        h = _make_history("a", "b", "c")
        replacement = [types.Content(role="user", parts=[types.Part.from_text(text="summary")])]
        h.replace_messages(0, 3, replacement)
        assert len(h) == 1
        assert h.get_messages()[0].parts[0].text == "summary"

    def test_replace_partial_range(self) -> None:
        h = _make_history("a", "b", "c", "d")
        replacement = [types.Content(role="user", parts=[types.Part.from_text(text="x")])]
        h.replace_messages(1, 3, replacement)
        # 4 - 2 replaced + 1 inserted = 3
        assert len(h) == 3

    def test_replace_with_empty_list_deletes(self) -> None:
        h = _make_history("a", "b", "c")
        h.replace_messages(1, 2, [])
        assert len(h) == 2

    def test_replace_preserves_surrounding_messages(self) -> None:
        h = _make_history("first", "middle", "last")
        replacement = [types.Content(role="user", parts=[types.Part.from_text(text="new middle")])]
        h.replace_messages(1, 2, replacement)
        msgs = h.get_messages()
        assert msgs[0].parts[0].text == "first"
        assert msgs[1].parts[0].text == "new middle"
        assert msgs[2].parts[0].text == "last"

    def test_invalid_start_raises_index_error(self) -> None:
        h = _make_history("a", "b")
        with pytest.raises(IndexError):
            h.replace_messages(-1, 1, [])

    def test_start_beyond_length_raises_index_error(self) -> None:
        h = _make_history("a", "b")
        with pytest.raises(IndexError):
            h.replace_messages(10, 11, [])

    def test_end_beyond_length_raises_index_error(self) -> None:
        h = _make_history("a", "b")
        with pytest.raises(IndexError):
            h.replace_messages(0, 10, [])

    def test_end_less_than_start_raises_index_error(self) -> None:
        h = _make_history("a", "b", "c")
        with pytest.raises(IndexError):
            h.replace_messages(2, 1, [])


# ---------------------------------------------------------------------------
# TestGetRecentMessages
# ---------------------------------------------------------------------------


class TestGetRecentMessages:
    """get_recent_messages() should return the last n messages."""

    def test_returns_last_n(self) -> None:
        h = _make_history("a", "b", "c", "d")
        result = h.get_recent_messages(2)
        assert len(result) == 2

    def test_returns_correct_messages(self) -> None:
        h = ConversationHistory()
        h.add_user_message("first")
        h.add_user_message("second")
        h.add_user_message("third")
        result = h.get_recent_messages(2)
        texts = [m.parts[0].text for m in result]
        assert "second" in texts
        assert "third" in texts
        assert "first" not in texts

    def test_n_greater_than_length_returns_all(self) -> None:
        h = _make_history("a", "b")
        result = h.get_recent_messages(100)
        assert len(result) == 2

    def test_n_zero_returns_empty(self) -> None:
        h = _make_history("a", "b", "c")
        result = h.get_recent_messages(0)
        assert result == []

    def test_negative_n_returns_empty(self) -> None:
        h = _make_history("a", "b")
        result = h.get_recent_messages(-1)
        assert result == []

    def test_returns_new_list(self) -> None:
        h = _make_history("a", "b")
        result = h.get_recent_messages(2)
        result.clear()
        assert len(h) == 2

    def test_empty_history_returns_empty(self) -> None:
        h = ConversationHistory()
        assert h.get_recent_messages(5) == []


# ---------------------------------------------------------------------------
# TestEstimateTokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    """Module-level estimate_tokens() function."""

    def test_empty_string_returns_zero(self) -> None:
        assert estimate_tokens("") == 0

    def test_four_chars_is_one_token(self) -> None:
        assert estimate_tokens("abcd") == 1

    def test_eight_chars_is_two_tokens(self) -> None:
        assert estimate_tokens("abcdefgh") == 2

    def test_returns_integer(self) -> None:
        result = estimate_tokens("some text")
        assert isinstance(result, int)

    def test_longer_text_more_tokens(self) -> None:
        short = estimate_tokens("hi")
        long = estimate_tokens("This is a much longer piece of text with many characters")
        assert long > short

    def test_single_char_returns_zero(self) -> None:
        # 1 // 4 == 0
        assert estimate_tokens("x") == 0

    def test_three_chars_returns_zero(self) -> None:
        assert estimate_tokens("abc") == 0

    def test_whitespace_counted(self) -> None:
        # 4 spaces = 1 token
        assert estimate_tokens("    ") == 1
