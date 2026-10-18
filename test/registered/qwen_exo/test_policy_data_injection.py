"""The persona (PolicyData) document: injected in full, hot-edited, hot-toggled."""

from types import SimpleNamespace

from pydantic import BaseModel

from qwen_exo_booster.policy_data import PolicyDataRepository
from qwen_exo_booster.runtime import QwenExoRuntime


class _Request(BaseModel):
    instructions: str | None = None
    extra_key: str | None = None


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text) // 4))


def _runtime(tmp_path, enabled=True):
    runtime = QwenExoRuntime.__new__(QwenExoRuntime)
    runtime.policy_data = PolicyDataRepository(tmp_path / "policydata")
    runtime.policy_data.refresh()
    runtime.policy_data_enabled = enabled
    runtime.tokenizer_manager = SimpleNamespace(tokenizer=_Tokenizer())
    runtime.telemetry = SimpleNamespace(emit=lambda *args, **kwargs: None)
    return runtime


def test_persona_is_injected_in_full_and_follows_disk_edits_and_the_switch(tmp_path):
    """A 4096-token cap once cut a 13k-token persona to its first third; disk
    edits needed a manual reindex; the switch needed a restart."""
    runtime = _runtime(tmp_path)
    document = tmp_path / "policydata" / "persona.md"
    document.write_text("# Persona\n\n" + "rule. " * 20_000 + "LAST_RULE\n", encoding="utf-8")

    attached = runtime._attach_policy_data(_Request(instructions="client"))
    assert attached.instructions.startswith("client\n\n")
    assert "LAST_RULE" in attached.instructions
    assert runtime.policy_data_injection()["attached_tokens"] > 20_000

    document.write_text("# Persona\n\nEDITED_ON_DISK\n", encoding="utf-8")
    edited = runtime._attach_policy_data(_Request())
    assert "EDITED_ON_DISK" in edited.instructions and "LAST_RULE" not in edited.instructions
    assert edited.extra_key != attached.extra_key  # new cache namespace for new text

    assert runtime.set_policy_data_enabled(False)["enabled"] is False
    assert runtime._attach_policy_data(_Request(instructions="client")).instructions == "client"
