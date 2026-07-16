"""Privacy mode: redaction of sensitive data before persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from errlore import AgentMemory
from errlore.redaction import Redactor


class TestRedactorPatterns:
    def setup_method(self) -> None:
        self.r = Redactor()

    def test_email(self) -> None:
        out = self.r.redact("mail sent to admin@example.com failed")
        assert "admin@example.com" not in out
        assert "[REDACTED_EMAIL]" in out

    def test_ipv4(self) -> None:
        out = self.r.redact("connection to 192.168.1.42 refused")
        assert "192.168.1.42" not in out
        assert "[REDACTED_IP]" in out

    def test_bearer_token(self) -> None:
        out = self.r.redact("header was Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.abc")
        assert "eyJhbGciOiJIUzI1NiJ9" not in out
        assert "[REDACTED_TOKEN]" in out

    def test_key_value_credentials(self) -> None:
        out = self.r.redact("failed with password=hunter2 and api_key: abc123def")
        assert "hunter2" not in out
        assert "abc123def" not in out
        assert "password=[REDACTED]" in out

    def test_known_key_prefixes(self) -> None:
        for secret in (
            "sk-proj-abcdefghijklmnop1234",
            "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456",
            "github_pat_11ABCDEFGHIJKLMNOPQRST",
            "AKIAIOSFODNN7EXAMPLE",
            "xoxb-123456789012-abcdefghij",
        ):
            out = self.r.redact(f"leaked {secret} in log")
            assert secret not in out, secret
            assert "[REDACTED_KEY]" in out, secret

    def test_plain_error_text_untouched(self) -> None:
        text = "TimeoutError: request to the load balancer took 30s (retry 3/3)"
        assert self.r.redact(text) == text

    def test_custom_patterns(self) -> None:
        r = Redactor(extra_patterns=[r"CUST-\d{6}"])
        out = r.redact("order failed for CUST-483920")
        assert "CUST-483920" not in out
        assert "[REDACTED]" in out

    def test_invalid_custom_pattern_fails_at_construction(self) -> None:
        import re

        with pytest.raises(re.error):
            Redactor(extra_patterns=["("])


class TestPrivacyModeFacade:
    def test_log_error_scrubs_before_disk(self, data_dir: Path) -> None:
        mem = AgentMemory(data_dir, trust=False, privacy_mode=True)
        mem.log_error(
            "m", "bash",
            "CommandFailed: curl -H 'Authorization: Bearer secr3ttok3nvalue' "
            "http://10.0.0.5/admin :: denied for ops@corp.io",
        )
        raw = (data_dir / "errors.jsonl").read_text()
        assert "secr3ttok3nvalue" not in raw
        assert "10.0.0.5" not in raw
        assert "ops@corp.io" not in raw
        # The weakness profile file must be scrubbed too.
        acc = (data_dir / "model_accuracy.jsonl").read_text()
        assert "secr3ttok3nvalue" not in acc

    def test_lesson_scrubbed_via_resolve_and_add(self, data_dir: Path) -> None:
        mem = AgentMemory(data_dir, trust=False, privacy_mode=True)
        eid = mem.log_error("m", "t", "AuthError: boom")
        mem.resolve(eid, "rotated key sk-proj-abcdefghijklmnop1234",
                    lesson="use api_key=abc123def only over TLS")
        mem.add_lesson("db timeouts", "connect as dba@corp.io with password=pw1",
                       task_type="t")
        raw = (data_dir / "lessons.jsonl").read_text() + (
            data_dir / "errors.jsonl"
        ).read_text()
        assert "abc123def" not in raw
        assert "sk-proj-abcdefghijklmnop1234" not in raw
        assert "dba@corp.io" not in raw
        assert "pw1" not in raw

    def test_off_by_default(self, data_dir: Path) -> None:
        mem = AgentMemory(data_dir, trust=False)
        mem.log_error("m", "t", "Error: mail admin@example.com about this")
        assert "admin@example.com" in (data_dir / "errors.jsonl").read_text()

    def test_custom_patterns_reach_disk_path(self, data_dir: Path) -> None:
        mem = AgentMemory(
            data_dir, trust=False, privacy_mode=True,
            redact_patterns=[r"ORD-\d+"],
        )
        mem.log_error("m", "t", "Error: order ORD-9911 stuck")
        raw = (data_dir / "errors.jsonl").read_text()
        assert "ORD-9911" not in raw
        assert "[REDACTED]" in raw


class TestClaudeCodeHookEnv:
    def test_env_flag_parsing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from errlore.integrations import claude_code as cc

        for val, expected in (
            ("1", True), ("true", True), ("YES", True), ("on", True),
            ("", False), ("0", False), ("off", False),
        ):
            monkeypatch.setenv("ERRLORE_PRIVACY_MODE", val)
            assert cc.privacy_mode() is expected, val

    def test_hook_scrubs_failed_command(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from errlore.integrations import claude_code as cc

        monkeypatch.setenv("ERRLORE_DATA", str(data_dir))
        monkeypatch.setenv("ERRLORE_PRIVACY_MODE", "1")
        event = json.dumps({
            "tool_name": "Bash",
            "tool_input": {"command": "curl -u admin:hunter2 http://10.1.2.3/"},
            "error": "401 from 10.1.2.3 for admin@corp.io",
        })
        assert cc.post_tool_use_failure(event) == 0
        raw = (data_dir / "errors.jsonl").read_text()
        assert "10.1.2.3" not in raw
        assert "admin@corp.io" not in raw
