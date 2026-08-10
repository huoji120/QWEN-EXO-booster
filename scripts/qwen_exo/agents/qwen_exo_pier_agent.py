from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Any

from pier.agents.installed.mini_swe_agent import MiniSweAgent
from pier.environments.base import BaseEnvironment
from pier.models.agent.context import AgentContext

_PRESERVE_WORKTREE_COMMAND = """root="$(git -C /app rev-parse --show-toplevel 2>/dev/null || pwd)"
git -C "$root" add -A
if ! git -C "$root" diff --cached --quiet; then
  git -C "$root" -c user.name="QWEN-EXO Recovery" \
    -c user.email="qwen-exo@local" commit --no-verify \
    -m "Preserve QWEN-EXO agent work"
fi
"""


class QwenExoMiniSweAgent(MiniSweAgent):
    """Pier adapter that injects the QWEN-EXO mini-swe agent without mounts."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        source = Path(__file__).with_name("qwen_exo_compacting_agent.py").read_bytes()
        self._encoded_agent_source = base64.b64encode(source).decode("ascii")
        super().__init__(*args, **kwargs)

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        target_directory = "/tmp/qwen-exo-agent"
        target_file = f"{target_directory}/qwen_exo_compacting_agent.py"
        await self.exec_as_agent(
            environment,
            command=(
                f"mkdir -p {target_directory}\n"
                f"printf '%s' '{self._encoded_agent_source}' "
                f"| base64 --decode > {target_file}"
            ),
            env=self.build_process_env(),
        )
        try:
            await super().run(instruction, environment, context)
        finally:
            try:
                await self.exec_as_agent(
                    environment,
                    command=_PRESERVE_WORKTREE_COMMAND,
                    env=self.build_process_env(),
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[qwen-exo] emergency patch snapshot failed: {exc}")
