"""Private two-case knowledge import; stdout/errors contain metadata only.

--normalize --rl FILE --events FILE --html FILE --output PRIVATE_DIRECTORY
--check --output PRIVATE_DIRECTORY replays the source-to-canonical fidelity check.
No captured content is executed. HTML is an explicitly user-associated additional
assistant target in the event case, not an original trajectory message.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]


class NormalizationError(ValueError):
    """Only stable structural codes may escape through this exception."""


def require(condition, code):
    if not condition:
        raise NormalizationError(code)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def wrap(tag, value):
    return "<" + tag + ">\n" + canonical(value).decode("utf-8") + "\n</" + tag + ">"


def append_message(messages, role, text, counts):
    require(role in ("system", "user", "assistant") and isinstance(text, str), "message_shape_invalid")
    if messages and messages[-1]["role"] == role:
        messages[-1]["content"] += "\n" + text
        counts["same_role_merges"] += 1
    else:
        require(role != "system" or not messages, "mid_conversation_system_unsupported")
        messages.append({"role": role, "content": text})


def role_report(messages):
    roles = [m["role"] for m in messages]
    turns = roles[1:] if roles and roles[0] == "system" else roles
    strict = bool(turns) and len(turns) % 2 == 0 and all(
        role == ("user" if n % 2 == 0 else "assistant") for n, role in enumerate(turns))
    return {"message_roles": dict(collections.Counter(roles)), "strict_user_assistant_pairs": strict,
            "terminal_role": roles[-1], "messages": len(messages)}


def convert(rl_path, events_path, html_path):
    sources = {"rl": Path(rl_path), "events": Path(events_path), "html": Path(html_path)}
    raw = {key: path.read_bytes() for key, path in sources.items()}
    rl = json.loads(raw["rl"])
    events = json.loads(raw["events"])
    html = raw["html"].decode("utf-8")
    require(html.encode("utf-8") == raw["html"], "html_utf8_roundtrip_failed")
    original = rl["session"]["messages"]
    require(isinstance(original, list) and original, "rl_messages_missing")
    rl_counts = collections.Counter()
    messages = []
    components = []
    for index, message in enumerate(original):
        require(isinstance(message, dict), "rl_message_invalid")
        role, text = message.get("role"), message.get("content")
        require(isinstance(text, str), "rl_content_not_string")
        components.append({"index": index, "sha256": sha(canonical(message)), "utf8_bytes": len(text.encode())})
        try:
            encoded = json.loads(text)
        except ValueError:
            encoded = None
        if isinstance(encoded, dict) and encoded.get("type") == "function_call_output":
            require(set(encoded) == {"type", "call_id", "output"}
                    and isinstance(encoded["call_id"], str) and isinstance(encoded["output"], str),
                    "native_function_output_shape_unsupported")
            require(role == "user", "native_function_output_role_invalid")
            text = "<tool_response>\n" + text + "\n</tool_response>"
            rl_counts["native_function_outputs"] += 1
        append_message(messages, role, text, rl_counts)
    result = [{"group": "case_1", "kind": "original_rl", "knowledge_case": True,
               "messages": messages, "provenance": {"source": "rl", "original_message_count": len(original)}}]
    rl_report = {**role_report(messages), **dict(rl_counts), "source_messages": len(original),
                 "source_messages_sha256": sha(canonical(original)), "components": components,
                 "coverage": "all_provided_messages", "dropped_messages": 0}
    steps = events.get("steps")
    require(isinstance(steps, list) and steps, "event_steps_missing")
    sequences = [step.get("seq") for step in steps]
    require(all(isinstance(seq, int) for seq in sequences) and len(set(sequences)) == len(sequences),
            "event_sequence_invalid")
    require(sequences == sorted(sequences), "event_order_invalid")
    kinds = collections.Counter(item.get("kind") for step in steps for item in step["items"])
    allowed = {"system_note": "system", "user_text": "user", "assistant_text": "assistant",
               "thinking": "assistant", "tool_call": "assistant", "tool_result": "tool"}
    unknown = sum(count for kind, count in kinds.items() if kind not in allowed)
    require(unknown == 0, f"unhandled_event_kind_count_{unknown}")
    counts = collections.Counter()
    event_messages = []
    components = []
    pending = collections.Counter()
    unmatched_results = 0
    first_user = None
    for step in steps:
        for index, item in enumerate(step["items"]):
            kind = item["kind"]
            require(item.get("role") == allowed[kind] and isinstance(item.get("text"), str),
                    "event_item_role_or_text_invalid")
            text = item["text"]
            components.append({"seq": step["seq"], "item": index, "kind": kind,
                               "sha256": sha(canonical(item)), "text_utf8_bytes": len(text.encode())})
            role = allowed[kind]
            if kind == "thinking":
                text = "<think>" + text + "</think>"
            elif kind == "tool_call":
                require(isinstance(item.get("args"), dict) and isinstance(item.get("call_id"), str)
                        and isinstance(item.get("name"), str), "event_tool_call_shape_invalid")
                text = wrap("tool_call", {"call_id": item["call_id"], "name": item["name"], "arguments": item["args"]})
                pending[item["call_id"]] += 1
            elif kind == "tool_result":
                require(isinstance(item.get("call_id"), str), "event_tool_result_shape_invalid")
                text = wrap("tool_response", {"call_id": item["call_id"], "name": item.get("name"),
                                             "output": text, "is_error": item.get("is_error")})
                role = "user"
                if pending[item["call_id"]]:
                    pending[item["call_id"]] -= 1
                else:
                    unmatched_results += 1
            if kind == "user_text" and first_user is None:
                first_user = item["text"]
            append_message(event_messages, role, text, counts)
    require(first_user is not None, "event_first_original_user_missing")
    pagination = events.get("pagination")
    require(isinstance(pagination, dict) and isinstance(pagination.get("total"), int), "event_pagination_missing")
    total = pagination["total"]
    require(total >= len(steps), "event_total_less_than_provided")
    result.append({"group": "case_2", "kind": "original_events", "knowledge_case": True,
                   "messages": event_messages, "provenance": {"source": "events", "provided_steps": len(steps),
                   "declared_total_steps": total, "complete_session": False}})
    system = [dict(m) for m in event_messages[:1] if m["role"] == "system"]
    result.append({"group": "case_2", "kind": "source_artifact", "knowledge_case": True,
                   "messages": system + [{"role": "user", "content": first_user}, {"role": "assistant", "content": html}],
                   "provenance": {"source": "html", "association": "synthetic_pair_user_authorized",
                   "original_trajectory_message": False, "prompt_sha256": sha(first_user.encode()),
                   "artifact_utf8_sha256": sha(raw["html"])}})
    event_report = {**role_report(event_messages), **dict(counts), "kind_counts": dict(kinds),
                    "provided_steps": len(steps), "declared_total_steps": total,
                    "unprovided_steps": total - len(steps), "pagination": pagination,
                    "sequence_min": min(sequences), "sequence_max": max(sequences),
                    "sequence_gaps": sum(b - a - 1 for a, b in zip(sequences, sequences[1:])),
                    "unresolved_tool_calls": sum(pending.values()), "unmatched_tool_results": unmatched_results,
                    "unsettled_tail_preserved": bool(sum(pending.values()) or event_messages[-1]["role"] != "assistant"),
                    "coverage": "partial_provided_export_only", "complete_session": False,
                    "dropped_items": 0, "components": components}
    document = b"\n".join(canonical(record) for record in result) + b"\n"
    manifest = {"schema_version": 1, "case_groups": 2, "records": len(result),
                "sources": {key: {"path": str(path.resolve()), "bytes": len(raw[key]), "sha256": sha(raw[key])}
                            for key, path in sources.items()},
                "canonical_sha256": sha(document), "canonical_bytes": len(document),
                "rl": rl_report, "events": event_report, "artifact": {"additional_case_2_target": True,
                "association": "synthetic_pair_user_authorized", "bytes": len(raw["html"]), "sha256": sha(raw["html"])},
                "content_fidelity": "exact_original_text_and_json_values_preserved_with_explicit_wrappers",
                "redaction_performed": False, "training_started": False}
    return document, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--normalize", action="store_true")
    mode.add_argument("--check", action="store_true")
    for name in ("rl", "events", "html"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = args.output.resolve()
        require(not output.is_relative_to(REPO_ROOT), "output_must_be_outside_public_repository")
        if args.check:
            manifest = json.loads((output / "provenance.json").read_bytes())
            paths = {key: entry["path"] for key, entry in manifest["sources"].items()}
            data, rebuilt = convert(paths["rl"], paths["events"], paths["html"])
            require(rebuilt == manifest, "provenance_replay_mismatch")
            require((output / "cases.jsonl").read_bytes() == data, "canonical_fidelity_mismatch")
        else:
            require(args.rl and args.events and args.html, "three_sources_required")
            require(not output.exists(), "output_already_exists")
            data, manifest = convert(args.rl, args.events, args.html)
            output.mkdir(parents=True)
            (output / "cases.jsonl").write_bytes(data)
            (output / "provenance.json").write_bytes(canonical(manifest) + b"\n")
        print(json.dumps({"schema_version": 1, "case_groups": manifest["case_groups"],
                          "records": manifest["records"], "canonical_sha256": manifest["canonical_sha256"],
                          "canonical_bytes": manifest["canonical_bytes"], "checked": args.check,
                          "source_messages": manifest["rl"]["source_messages"],
                          "provided_steps": manifest["events"]["provided_steps"],
                          "unprovided_steps": manifest["events"]["unprovided_steps"],
                          "unresolved_tool_calls": manifest["events"]["unresolved_tool_calls"]}, sort_keys=True))
    except Exception as error:
        code = str(error) if isinstance(error, NormalizationError) else type(error).__name__
        print(json.dumps({"error": code}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
