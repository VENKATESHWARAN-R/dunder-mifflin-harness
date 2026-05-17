from pathlib import Path

from jac.cli.parser import (
    ParsedInputKind,
    extract_file_references,
    parse_input,
)


def test_parse_slash_command(tmp_path: Path) -> None:
    parsed = parse_input("/model test-model", cwd=tmp_path, max_attachment_bytes=1000)

    assert parsed.kind == ParsedInputKind.SLASH
    assert parsed.slash is not None
    assert parsed.slash.command == "model"
    assert parsed.slash.args == "test-model"


def test_parse_shell_only_when_input_starts_with_bang(tmp_path: Path) -> None:
    shell = parse_input("!echo hi", cwd=tmp_path, max_attachment_bytes=1000)
    plain = parse_input(
        "please explain !echo hi", cwd=tmp_path, max_attachment_bytes=1000
    )

    assert shell.kind == ParsedInputKind.SHELL
    assert shell.shell_command == "echo hi"
    assert plain.kind == ParsedInputKind.PLAIN


def test_parse_file_attachments_and_warnings(tmp_path: Path) -> None:
    note = tmp_path / "note.txt"
    note.write_text("hello", encoding="utf-8")

    parsed = parse_input(
        "read @note.txt and @missing.txt",
        cwd=tmp_path,
        max_attachment_bytes=1000,
    )

    assert parsed.kind == ParsedInputKind.PLAIN
    assert [item.display_path for item in parsed.attachments] == ["note.txt"]
    assert parsed.attachments[0].content == "hello"
    assert len(parsed.warnings) == 1
    assert "file not found" in parsed.warnings[0].message


def test_extract_file_references_ignores_email_and_supports_quotes() -> None:
    refs = extract_file_references(
        'email me@example.com then read @"two words.md" @one.md'
    )

    assert refs == ["two words.md", "one.md"]
