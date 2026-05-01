"""Tests for pygemini.safety.policy (PolicyEngine, PolicyRule, PolicyDecision)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock


from pygemini.safety.policy import PolicyEngine, PolicyRule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine(rules: list[PolicyRule] | None = None) -> PolicyEngine:
    """Build a PolicyEngine with a mocked config (no files to load)."""
    config = MagicMock()
    # config_dir must return a non-existent path so no TOML files are loaded.
    config.config_dir = Path("/tmp/__nonexistent_pygemini_test__")
    engine = PolicyEngine.__new__(PolicyEngine)
    engine._rules = rules or []
    return engine


# ---------------------------------------------------------------------------
# TestPolicyRule
# ---------------------------------------------------------------------------


class TestPolicyRule:
    """PolicyRule dataclass construction and attribute access."""

    def test_basic_construction(self) -> None:
        rule = PolicyRule(tool_pattern="run_shell_*", action="deny")
        assert rule.tool_pattern == "run_shell_*"
        assert rule.action == "deny"
        assert rule.path_pattern is None
        assert rule.command_pattern is None

    def test_full_construction(self) -> None:
        rule = PolicyRule(
            tool_pattern="write_file",
            action="allow",
            path_pattern="*.test.py",
            command_pattern=None,
        )
        assert rule.path_pattern == "*.test.py"

    def test_with_command_pattern(self) -> None:
        rule = PolicyRule(
            tool_pattern="run_shell_command",
            action="deny",
            command_pattern=r"rm -rf.*",
        )
        assert rule.command_pattern == r"rm -rf.*"

    def test_valid_actions(self) -> None:
        for action in ("allow", "deny", "confirm"):
            rule = PolicyRule(tool_pattern="*", action=action)
            assert rule.action == action


# ---------------------------------------------------------------------------
# TestEvaluate
# ---------------------------------------------------------------------------


class TestEvaluate:
    """evaluate() should return the first matching rule or default to confirm."""

    def test_no_rules_defaults_to_confirm(self) -> None:
        engine = _make_engine([])
        decision = engine.evaluate("some_tool", {})
        assert decision.action == "confirm"
        assert decision.matched_rule is None

    def test_matching_rule_is_returned(self) -> None:
        rule = PolicyRule(tool_pattern="write_file", action="allow")
        engine = _make_engine([rule])
        decision = engine.evaluate("write_file", {})
        assert decision.action == "allow"
        assert decision.matched_rule is rule

    def test_no_match_returns_confirm(self) -> None:
        rule = PolicyRule(tool_pattern="write_file", action="deny")
        engine = _make_engine([rule])
        decision = engine.evaluate("read_file", {})
        assert decision.action == "confirm"
        assert decision.matched_rule is None

    def test_first_match_wins(self) -> None:
        rule1 = PolicyRule(tool_pattern="write_file", action="allow")
        rule2 = PolicyRule(tool_pattern="write_file", action="deny")
        engine = _make_engine([rule1, rule2])
        decision = engine.evaluate("write_file", {})
        assert decision.action == "allow"
        assert decision.matched_rule is rule1

    def test_decision_has_reason(self) -> None:
        rule = PolicyRule(tool_pattern="web_*", action="deny")
        engine = _make_engine([rule])
        decision = engine.evaluate("web_fetch", {})
        assert decision.reason != ""

    def test_deny_action_returned(self) -> None:
        rule = PolicyRule(tool_pattern="run_shell_command", action="deny")
        engine = _make_engine([rule])
        decision = engine.evaluate("run_shell_command", {})
        assert decision.action == "deny"


# ---------------------------------------------------------------------------
# TestToolPatternMatching
# ---------------------------------------------------------------------------


class TestToolPatternMatching:
    """Tool name patterns use fnmatch (supports * and ?)."""

    def test_star_matches_suffix(self) -> None:
        rule = PolicyRule(tool_pattern="run_*", action="deny")
        engine = _make_engine([rule])
        assert engine.evaluate("run_shell_command", {}).action == "deny"
        assert engine.evaluate("run_anything", {}).action == "deny"

    def test_star_does_not_match_prefix(self) -> None:
        rule = PolicyRule(tool_pattern="run_*", action="deny")
        engine = _make_engine([rule])
        decision = engine.evaluate("my_run_tool", {})
        assert decision.action == "confirm"

    def test_question_mark_matches_single_char(self) -> None:
        rule = PolicyRule(tool_pattern="tool_?", action="allow")
        engine = _make_engine([rule])
        assert engine.evaluate("tool_a", {}).action == "allow"
        assert engine.evaluate("tool_z", {}).action == "allow"

    def test_question_mark_does_not_match_multiple_chars(self) -> None:
        rule = PolicyRule(tool_pattern="tool_?", action="allow")
        engine = _make_engine([rule])
        decision = engine.evaluate("tool_ab", {})
        assert decision.action == "confirm"

    def test_exact_match(self) -> None:
        rule = PolicyRule(tool_pattern="write_file", action="deny")
        engine = _make_engine([rule])
        assert engine.evaluate("write_file", {}).action == "deny"
        assert engine.evaluate("write_file_extra", {}).action == "confirm"

    def test_wildcard_star_matches_all(self) -> None:
        rule = PolicyRule(tool_pattern="*", action="allow")
        engine = _make_engine([rule])
        assert engine.evaluate("anything", {}).action == "allow"
        assert engine.evaluate("whatever_tool", {}).action == "allow"


# ---------------------------------------------------------------------------
# TestPathPatternMatching
# ---------------------------------------------------------------------------


class TestPathPatternMatching:
    """Path pattern uses fnmatch against params["path"]."""

    def test_path_pattern_matches_param(self) -> None:
        rule = PolicyRule(tool_pattern="write_file", action="deny", path_pattern="*.test.py")
        engine = _make_engine([rule])
        decision = engine.evaluate("write_file", {"path": "test_foo.test.py"})
        assert decision.action == "deny"

    def test_path_pattern_no_match(self) -> None:
        rule = PolicyRule(tool_pattern="write_file", action="deny", path_pattern="*.test.py")
        engine = _make_engine([rule])
        decision = engine.evaluate("write_file", {"path": "src/main.py"})
        assert decision.action == "confirm"

    def test_path_pattern_no_path_param_skips_rule(self) -> None:
        """If rule has path_pattern but call has no path param, rule does not match."""
        rule = PolicyRule(tool_pattern="write_file", action="deny", path_pattern="*.py")
        engine = _make_engine([rule])
        decision = engine.evaluate("write_file", {})
        assert decision.action == "confirm"

    def test_path_pattern_with_slash_wildcards(self) -> None:
        rule = PolicyRule(tool_pattern="*", action="allow", path_pattern="/tmp/*")
        engine = _make_engine([rule])
        assert engine.evaluate("read_file", {"path": "/tmp/foo.txt"}).action == "allow"
        assert engine.evaluate("read_file", {"path": "/home/user/foo.txt"}).action == "confirm"


# ---------------------------------------------------------------------------
# TestCommandPatternMatching
# ---------------------------------------------------------------------------


class TestCommandPatternMatching:
    """Command pattern uses re.search against params["command"]."""

    def test_command_pattern_matches(self) -> None:
        rule = PolicyRule(
            tool_pattern="run_shell_command",
            action="deny",
            command_pattern=r"rm -rf",
        )
        engine = _make_engine([rule])
        decision = engine.evaluate(
            "run_shell_command", {"command": "rm -rf /tmp/junk"}
        )
        assert decision.action == "deny"

    def test_command_pattern_no_match(self) -> None:
        rule = PolicyRule(
            tool_pattern="run_shell_command",
            action="deny",
            command_pattern=r"rm -rf",
        )
        engine = _make_engine([rule])
        decision = engine.evaluate("run_shell_command", {"command": "ls -la"})
        assert decision.action == "confirm"

    def test_command_pattern_regex_anchors(self) -> None:
        rule = PolicyRule(
            tool_pattern="*",
            action="deny",
            command_pattern=r"^sudo ",
        )
        engine = _make_engine([rule])
        assert engine.evaluate("*", {"command": "sudo apt-get install"}).action == "deny"
        assert engine.evaluate("*", {"command": "echo sudo"}).action == "confirm"

    def test_command_pattern_no_command_param_skips_rule(self) -> None:
        """If rule has command_pattern but call has no command param, rule skips."""
        rule = PolicyRule(
            tool_pattern="*",
            action="deny",
            command_pattern=r"rm -rf",
        )
        engine = _make_engine([rule])
        decision = engine.evaluate("run_shell_command", {})
        assert decision.action == "confirm"


# ---------------------------------------------------------------------------
# TestLoadPolicies
# ---------------------------------------------------------------------------


class TestLoadPolicies:
    """_parse_toml loads rules from a TOML file correctly."""

    def test_load_valid_toml(self, tmp_path: Path) -> None:
        policies_file = tmp_path / "policies.toml"
        policies_file.write_text(
            '[rules]\n'  # will be overridden below
        )
        # Write proper array-of-tables format
        policies_file.write_bytes(
            b'[[rules]]\ntool = "write_file"\naction = "deny"\n'
            b'\n[[rules]]\ntool = "run_shell_*"\naction = "confirm"\n'
        )
        engine = PolicyEngine.__new__(PolicyEngine)
        engine._rules = []
        rules = engine._parse_toml(policies_file)
        assert len(rules) == 2
        assert rules[0].tool_pattern == "write_file"
        assert rules[0].action == "deny"
        assert rules[1].tool_pattern == "run_shell_*"
        assert rules[1].action == "confirm"

    def test_load_with_path_pattern(self, tmp_path: Path) -> None:
        policies_file = tmp_path / "policies.toml"
        policies_file.write_bytes(
            b'[[rules]]\ntool = "write_file"\naction = "allow"\npath_pattern = "*.test.py"\n'
        )
        engine = PolicyEngine.__new__(PolicyEngine)
        engine._rules = []
        rules = engine._parse_toml(policies_file)
        assert rules[0].path_pattern == "*.test.py"

    def test_load_with_command_pattern(self, tmp_path: Path) -> None:
        policies_file = tmp_path / "policies.toml"
        policies_file.write_bytes(
            b'[[rules]]\ntool = "run_shell_command"\naction = "deny"\ncommand_pattern = "rm -rf.*"\n'
        )
        engine = PolicyEngine.__new__(PolicyEngine)
        engine._rules = []
        rules = engine._parse_toml(policies_file)
        assert rules[0].command_pattern == "rm -rf.*"

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        engine = PolicyEngine.__new__(PolicyEngine)
        engine._rules = []
        rules = engine._parse_toml(tmp_path / "nonexistent.toml")
        assert rules == []

    def test_engine_rules_populated_via_add_rule(self) -> None:
        """add_rule() appends to the internal rule list."""
        engine = _make_engine()
        rule = PolicyRule(tool_pattern="*", action="allow")
        engine.add_rule(rule)
        assert rule in engine.rules

    def test_rules_property_returns_copy(self) -> None:
        """Modifying the returned list should not affect internal state."""
        rule = PolicyRule(tool_pattern="*", action="allow")
        engine = _make_engine([rule])
        returned = engine.rules
        returned.clear()
        assert len(engine.rules) == 1


# ---------------------------------------------------------------------------
# TestParseTOML
# ---------------------------------------------------------------------------


class TestParseTOML:
    """_parse_toml handles valid, invalid, and malformed inputs."""

    def _engine(self) -> PolicyEngine:
        engine = PolicyEngine.__new__(PolicyEngine)
        engine._rules = []
        return engine

    def test_valid_toml_with_allow_deny_confirm(self, tmp_path: Path) -> None:
        f = tmp_path / "p.toml"
        f.write_bytes(
            b'[[rules]]\ntool = "a"\naction = "allow"\n'
            b'[[rules]]\ntool = "b"\naction = "deny"\n'
            b'[[rules]]\ntool = "c"\naction = "confirm"\n'
        )
        rules = self._engine()._parse_toml(f)
        assert len(rules) == 3
        assert {r.action for r in rules} == {"allow", "deny", "confirm"}

    def test_invalid_action_is_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "p.toml"
        f.write_bytes(
            b'[[rules]]\ntool = "some_tool"\naction = "forbidden_action"\n'
        )
        rules = self._engine()._parse_toml(f)
        assert rules == []

    def test_missing_tool_field_is_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "p.toml"
        f.write_bytes(b'[[rules]]\naction = "allow"\n')
        rules = self._engine()._parse_toml(f)
        assert rules == []

    def test_malformed_toml_returns_empty(self, tmp_path: Path) -> None:
        f = tmp_path / "p.toml"
        f.write_bytes(b"this is not valid toml [[[")
        rules = self._engine()._parse_toml(f)
        assert rules == []

    def test_bad_regex_command_pattern_is_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "p.toml"
        f.write_bytes(
            b'[[rules]]\ntool = "run_shell_command"\naction = "deny"\ncommand_pattern = "[invalid"\n'
        )
        rules = self._engine()._parse_toml(f)
        assert rules == []

    def test_non_string_path_pattern_is_ignored(self, tmp_path: Path) -> None:
        """A non-string path_pattern is ignored (rule still parsed, just no path_pattern)."""
        f = tmp_path / "p.toml"
        f.write_bytes(
            b'[[rules]]\ntool = "write_file"\naction = "allow"\npath_pattern = 123\n'
        )
        rules = self._engine()._parse_toml(f)
        assert len(rules) == 1
        assert rules[0].path_pattern is None

    def test_empty_rules_array_returns_empty(self, tmp_path: Path) -> None:
        f = tmp_path / "p.toml"
        f.write_bytes(b"rules = []\n")
        rules = self._engine()._parse_toml(f)
        assert rules == []
