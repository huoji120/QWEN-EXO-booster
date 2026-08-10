from __future__ import annotations

import os
import sys

from qwen_exo_booster.fingerprint import (
    _COMPATIBILITY_GUIDANCE,
    validate_qwen_exo_model_path,
)


def _argument_value(arguments: list[str], option: str) -> str | None:
    value = None
    prefix = f"{option}="
    for index, argument in enumerate(arguments):
        if argument.startswith(prefix):
            value = argument[len(prefix) :]
        elif argument == option:
            value = arguments[index + 1] if index + 1 < len(arguments) else None
    return value


def _validate_qwen_exo_model_arguments(arguments: list[str]) -> str | None:
    if "--enable-qwen-exo" not in arguments:
        return None
    model_path = _argument_value(arguments, "--model-path")
    if not model_path:
        raise SystemExit(
            "QWEN-EXO startup blocked: --model-path is required when "
            f"--enable-qwen-exo is set. {_COMPATIBILITY_GUIDANCE}."
        )
    try:
        return validate_qwen_exo_model_path(model_path)
    except ValueError as exc:
        raise SystemExit(f"QWEN-EXO startup blocked: {exc}.") from exc


def main() -> None:
    base_args = sys.argv[1:]
    if base_args[:1] == ["--"]:
        base_args = base_args[1:]
    if not base_args:
        raise SystemExit(
            "usage: python -m qwen_exo_booster.service_launcher -- <sglang args>"
        )
    _validate_qwen_exo_model_arguments(base_args)

    from qwen_exo_booster.activation_training import run_pending_activation_training
    from qwen_exo_booster.service_config import ServiceConfigError, ServiceConfigStore

    try:
        training = run_pending_activation_training()
        if training is not None:
            print(
                "QWEN-EXO activation training "
                f"{training.get('status')}: {training.get('editor')}",
                flush=True,
            )
    except Exception as exc:
        print(
            f"QWEN-EXO activation training coordinator failed closed: {exc}",
            file=sys.stderr,
            flush=True,
        )

    store = ServiceConfigStore.from_environment()
    try:
        _, effective_args = store.mark_applied(base_args)
    except ServiceConfigError as exc:
        raise SystemExit(f"QWEN-EXO service config error [{exc.code}]: {exc}") from exc

    os.execvp(
        sys.executable,
        [sys.executable, "-m", "sglang.launch_server", *effective_args],
    )


if __name__ == "__main__":
    main()
