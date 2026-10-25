"""Attach the runtime-wide initial GDN selection to user generate requests.

Every user-facing OpenAI entrypoint (Chat Completions, legacy Completions,
Responses) funnels through :func:`bind_initial_gdn_request` so each new
sequence starts from the same memory-derived recurrent state and shares one
radix cache namespace per state identity.
"""

from __future__ import annotations

from typing import Any

from qwen_exo_booster.api_keys import is_valid_api_key_id, validate_memory_scope

INITIAL_GDN_PARAM = "qwen_exo_session_initial_gdn"
DISABLE_INITIAL_GDN_PARAM = "qwen_exo_disable_session_initial_gdn"
INITIAL_GDN_DISABLED_CACHE_NAMESPACE = "qwen-exo:v1:global-initial-gdn:disabled"
INITIAL_GDN_UNAVAILABLE_CACHE_NAMESPACE = (
    "qwen-exo:v1:global-initial-gdn:unavailable"
)


def bind_initial_gdn_request(
    runtime: Any,
    *,
    custom_params: dict[str, Any],
    extra_key: str | None,
    previous_response_id: str | None = None,
    response_id: str | None = None,
    memory_key_id: str | None = None,
    memory_scope: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Return request custom params and radix key bound to the initial GDN.

    Client-supplied artifact identities are never trusted; the runtime chooses
    the snapshot. A conversation continued via ``previous_response_id`` keeps
    the snapshot it started with (pinned under ``response_id`` for the next
    turn); new conversations get the current global one.
    ``qwen_exo_disable_session_initial_gdn`` opts one request out (for A/B
    comparison). Requests without a bound state move to their own namespace so
    they never reuse cached recurrent state that descends from an initial GDN,
    and vice versa.
    """

    scope = validate_memory_scope(
        memory_scope if memory_scope is not None else ("private" if memory_key_id else "shared")
    )
    if (scope == "shared") != (memory_key_id is None) or (
        memory_key_id is not None and not is_valid_api_key_id(memory_key_id)
    ):
        raise ValueError("Memory scope does not match authenticated owner")
    params = dict(custom_params)
    disabled = params.pop(DISABLE_INITIAL_GDN_PARAM, None) is True
    disabled = disabled or scope == "private_only"
    params.pop(INITIAL_GDN_PARAM, None)
    if scope == "private_only":
        params.pop("qwen_exo_latent_transplant", None)
        params.pop("qwen_exo_activation_editor", None)
        params["qwen_exo_disable_activation_editor"] = True
    selection = runtime.initial_gdn_selection(
        previous_response_id=previous_response_id, response_id=response_id,
        memory_key_id=memory_key_id, memory_scope=scope, disabled=disabled,
    )
    if isinstance(selection, dict) and selection.get("cache_namespace"):
        params[INITIAL_GDN_PARAM] = dict(selection)
        namespace = str(selection["cache_namespace"])
    elif disabled:
        namespace = INITIAL_GDN_DISABLED_CACHE_NAMESPACE
    else:
        namespace = INITIAL_GDN_UNAVAILABLE_CACHE_NAMESPACE
    if memory_key_id is not None:
        namespace = f"memory-{scope}:{memory_key_id}|{namespace}"
    return params, f"{namespace}|{extra_key}" if extra_key else namespace


__all__ = [
    "DISABLE_INITIAL_GDN_PARAM",
    "INITIAL_GDN_DISABLED_CACHE_NAMESPACE",
    "INITIAL_GDN_PARAM",
    "INITIAL_GDN_UNAVAILABLE_CACHE_NAMESPACE",
    "bind_initial_gdn_request",
]
