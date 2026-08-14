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


def _runtime_state_dtype(selected_model: dict[str, object]) -> str:
    return (
        "float16"
        if selected_model.get("runtime_quantization") in {"gptq", "gptq_marlin"}
        else "bfloat16"
    )


def main() -> None:
    base_args = sys.argv[1:]
    if base_args[:1] == ["--"]:
        base_args = base_args[1:]
    if not base_args:
        raise SystemExit(
            "usage: python -m qwen_exo_booster.service_launcher -- <sglang args>"
        )

    if (
        "--enable-qwen-exo" not in base_args
        or not os.getenv("QWEN_EXO_MODEL_CATALOG_ROOTS", "").strip()
    ):
        _validate_qwen_exo_model_arguments(base_args)
    if (
        "--enable-qwen-exo" in base_args
        and os.getenv("QWEN_EXO_MODEL_CATALOG_ROOTS", "").strip()
    ):
        from qwen_exo_booster.model_catalog import ModelCatalogError, ModelCatalogStore

        try:
            _, base_args, selected_model = (
                ModelCatalogStore.from_environment().mark_applied(base_args)
            )
        except ModelCatalogError as exc:
            raise SystemExit(
                f"QWEN-EXO model catalog error [{exc.code}]: {exc}"
            ) from exc
        _validate_qwen_exo_model_arguments(base_args)
        os.environ["QWEN_EXO_ACTIVE_MODEL_PROFILE"] = str(
            ModelCatalogStore.from_environment().profiles_root
            / selected_model["model_fingerprint"]
        )
        os.environ["QWEN_EXO_ACTIVE_MODEL_PATH"] = str(selected_model["model_path"])
        os.environ["SGLANG_MAMBA_SSM_DTYPE"] = _runtime_state_dtype(selected_model)

        print(
            "QWEN-EXO selected model profile: "
            f"{selected_model['name']} ({selected_model['model_fingerprint'][:16]})",
            flush=True,
        )

    from qwen_exo_booster.activation_training import run_pending_activation_training
    from qwen_exo_booster.service_config import ServiceConfigError, ServiceConfigStore

    try:
        training = run_pending_activation_training(
            model_path=os.getenv("QWEN_EXO_ACTIVE_MODEL_PATH") or None
        )
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
