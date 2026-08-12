import base64
import json
import re
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

import qwen_exo_booster.router as router_module
from qwen_exo_booster.document_ingest import KnowledgeIngestError
from qwen_exo_booster.router import compat_router, router
from qwen_exo_booster.runtime import QwenExoRuntimeState
from qwen_exo_booster.service_config import ServiceConfigStore
from qwen_exo_booster.tensor_bank import TensorBankCompileError
from qwen_exo_booster.model_catalog import ModelCatalogStore


def _write_catalog_model(root: Path, architecture: str) -> None:
    root.mkdir()
    moe = architecture == "Qwen3_5MoeForConditionalGeneration"
    layer_count = 40 if moe else 64
    text = {
        "model_type": "qwen3_5_moe_text" if moe else "qwen3_5_text",
        "head_dim": 256,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "max_position_embeddings": 262144,
        "vocab_size": 248320,
        "full_attention_interval": 4,
        "num_hidden_layers": layer_count,
        "intermediate_size": None if moe else 17408,
        "hidden_size": 2048 if moe else 5120,
        "num_attention_heads": 16 if moe else 24,
        "num_key_value_heads": 2 if moe else 4,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32 if moe else 48,
        "layer_types": [
            "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
            for index in range(layer_count)
        ],
        "attn_output_gate": True,
        "partial_rotary_factor": 0.25,
        "rope_parameters": {"rope_theta": 10_000_000},
    }
    if moe:
        text.update(
            num_experts=256,
            num_experts_per_tok=8,
            moe_intermediate_size=512,
            shared_expert_intermediate_size=512,
        )
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": [architecture],
                "model_type": "qwen3_5_moe" if moe else "qwen3_5",
                "text_config": text,
            }
        ),
        encoding="utf-8",
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 1}, "weight_map": {}}),
        encoding="utf-8",
    )
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        (root / name).write_text(name, encoding="utf-8")


class FakeRuntime:
    def __init__(self):
        self.config = SimpleNamespace(
            telemetry_text_mode="off",
            feature_flags=SimpleNamespace(policy_data=True),
            max_policy_tokens=4096,
        )
        self.state = QwenExoRuntimeState.READY
        self.knowledge = SimpleNamespace(
            snapshot=SimpleNamespace(source_digest="digest")
        )
        self.policy_data = SimpleNamespace(
            snapshot=SimpleNamespace(source_digest="policy-digest")
        )
        self.policy_documents = {"policy.md": "# Policy"}
        self.policy_document_tags = {"policy.md": []}
        self.documents = {"guide.md": "# Guide"}
        self.document_tags = {"guide.md": []}
        self.tensor_bank = object()
        self.telemetry = SimpleNamespace(emit=lambda *args, **kwargs: None)
        self.reflection_organization_calls = 0
        self._reflection_organization_status = {
            "job_id": None,
            "status": "idle",
            "stage": "idle",
            "progress": 0,
            "message": "尚未开始整理",
        }
        self.pending_reflections = {
            "conversation-1": {
                "conversation_key": "conversation-1",
                "trajectory_id": "resp-1",
                "status": "waiting",
            }
        }

    def status(self):
        return {"runtime_state": "ready"}

    def health(self):
        return {"status": "ok", "runtime_state": "ready"}

    def telemetry_events(self, request_id=None, limit=256):
        return [{"event_id": 1, "request_id": request_id or "request-1"}][:limit]

    def knowledge_documents(self, query=""):
        needle = str(query).lower()
        return [
            {
                "relative_path": path,
                "byte_count": len(content.encode()),
                "tags": self.document_tags.get(path, []),
            }
            for path, content in self.documents.items()
            if not needle or needle in content.lower() or needle in path.lower()
        ]

    def knowledge_document(self, relative_path):
        if relative_path not in self.documents:
            raise KeyError(relative_path)
        return {
            "relative_path": relative_path,
            "content": self.documents[relative_path],
            "tags": self.document_tags.get(relative_path, []),
        }

    def upsert_knowledge(self, relative_path, content, tags=None):
        if not relative_path.endswith(".md"):
            raise ValueError("Knowledge sources must be Markdown files")
        self.documents[relative_path] = content
        self.document_tags[relative_path] = list(tags or [])
        return {
            "relative_path": relative_path,
            "content": content,
            "tags": list(tags or []),
        }

    def delete_knowledge(self, relative_path):
        if relative_path not in self.documents:
            raise FileNotFoundError(relative_path)
        del self.documents[relative_path]
        self.document_tags.pop(relative_path, None)

    def reindex_knowledge(self):
        return {"document_count": len(self.documents), "source_digest": "digest"}

    def start_reflection_memory_organization(self):
        self.reflection_organization_calls += 1
        self._reflection_organization_status = {
            "job_id": "reflection-organization-test",
            "status": "queued",
            "stage": "queued",
            "progress": 0,
            "message": "整理任务已进入后台队列",
        }
        return dict(self._reflection_organization_status)

    def reflection_memory_organization_status(self):
        return dict(self._reflection_organization_status)

    def policy_data_documents(self, query=""):
        needle = str(query).lower()
        return [
            {
                "relative_path": path,
                "byte_count": len(content.encode()),
                "tags": self.policy_document_tags.get(path, []),
            }
            for path, content in self.policy_documents.items()
            if not needle or needle in content.lower() or needle in path.lower()
        ]

    def policy_data_document(self, relative_path):
        if relative_path not in self.policy_documents:
            raise KeyError(relative_path)
        return {
            "relative_path": relative_path,
            "content": self.policy_documents[relative_path],
            "tags": self.policy_document_tags.get(relative_path, []),
        }

    def upsert_policy_data(self, relative_path, content, tags=None):
        del tags
        if not relative_path.endswith(".md"):
            raise ValueError("PolicyData sources must be Markdown files")
        if self.policy_documents and relative_path not in self.policy_documents:
            raise RuntimeError(
                "PolicyData already contains its one personality document"
            )
        self.policy_documents[relative_path] = content
        self.policy_document_tags[relative_path] = []
        return {
            "relative_path": relative_path,
            "content": content,
            "tags": [],
        }

    def delete_policy_data(self, relative_path):
        if relative_path not in self.policy_documents:
            raise FileNotFoundError(relative_path)
        del self.policy_documents[relative_path]
        self.policy_document_tags.pop(relative_path, None)

    def reindex_policy_data(self):
        return {
            "document_count": len(self.policy_documents),
            "source_digest": "policy-digest",
        }

    async def reindex_tensor_bank(self):
        return {
            "document_state_count": 3,
            "model_native_documents": 3,
            "complete_gdn_document_states": 3,
        }

    def pending_reflection_memories(self):
        return list(self.pending_reflections.values())

    def start_pending_reflections(self, conversation_keys):
        return {
            "started": list(conversation_keys),
            "started_count": len(conversation_keys),
        }

    def cancel_pending_reflections(self, conversation_keys):
        for key in conversation_keys:
            if key not in self.pending_reflections:
                raise KeyError(key)
            self.pending_reflections.pop(key)
        return {
            "cancelled": list(conversation_keys),
            "cancelled_count": len(conversation_keys),
        }

    def delete_source_documents(self, lane, relative_paths):
        target = self.documents if lane == "knowledge" else self.policy_documents
        for path in relative_paths:
            if path not in target:
                raise FileNotFoundError(path)
        for path in relative_paths:
            target.pop(path)
        return {"deleted": True, "deleted_count": len(relative_paths)}

    async def compile_source_documents(self, lane, relative_paths):
        return {
            "requested_lane": lane,
            "requested_paths": list(relative_paths),
            "requested_count": len(relative_paths),
        }

    async def ingest_knowledge_files(self, files):
        return {
            "hot_updated": True,
            "restart_required": False,
            "source_digest": "updated-digest",
            "document_count": 2,
            "replaced_document_count": 0,
            "files": [
                {
                    "filename": files[0]["filename"],
                    "documents": [
                        {
                            "relative_path": "uploads/guide.md",
                            "token_count": 12,
                        }
                    ],
                }
            ],
            "tensor_bank": {"document_state_count": 4},
        }

    def recall_trace(self, max_turns=100):
        return {
            "schema": "inflight-memory-visualization-v2",
            "bank": {
                "semantics": "associative_global_policy_knowledge_semantic_admission"
            },
            "turns": [{"turn_id": "request-1"}][:max_turns],
        }

    async def clear_recall_trace(self):
        return {"status": "cleared", "turns": 0}


def client(runtime):
    app = FastAPI()
    app.include_router(router)
    app.include_router(compat_router)
    app.state.qwen_exo_runtime = runtime
    return TestClient(app)


def test_telemetry_and_knowledge_metadata_are_read_only_public():
    api = client(FakeRuntime())

    telemetry = api.get("/qwen-exo/telemetry", params={"request_id": "r"})
    knowledge = api.get("/qwen-exo/knowledge")

    assert telemetry.status_code == 200
    assert telemetry.json()["redacted"] is True
    assert telemetry.json()["events"][0]["request_id"] == "r"
    assert knowledge.status_code == 200
    assert "content" not in knowledge.json()["documents"][0]


def test_knowledge_content_and_mutations_are_directly_available():
    runtime = FakeRuntime()
    api = client(runtime)

    content = api.get("/qwen-exo/knowledge/guide.md")
    response = api.put(
        "/qwen-exo/knowledge/new.md",
        json={"content": "# New", "tags": ["sdk", "reviewed"]},
    )

    assert content.status_code == 200
    assert content.json()["content"] == "# Guide"
    assert response.status_code == 200
    assert runtime.documents["new.md"] == "# New"
    assert response.json()["tags"] == ["sdk", "reviewed"]

    deleted = api.delete("/qwen-exo/knowledge/new.md")
    assert deleted.status_code == 200
    assert "new.md" not in runtime.documents


def test_reflection_memory_organization_is_an_explicit_background_job():
    runtime = FakeRuntime()
    api = client(runtime)

    response = api.post("/qwen-exo/reflection-memory/organize")
    status = api.get("/qwen-exo/reflection-memory/organize")

    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    assert response.json()["job_id"] == "reflection-organization-test"
    assert status.status_code == 200
    assert status.json()["stage"] == "queued"
    assert runtime.reflection_organization_calls == 1


def test_pending_reflection_management_supports_list_start_and_cancel():
    runtime = FakeRuntime()
    api = client(runtime)

    listing = api.get("/qwen-exo/reflection-memory/pending")
    started = api.post(
        "/qwen-exo/reflection-memory/pending/reflect",
        json={"conversation_keys": ["conversation-1"]},
    )
    cancelled = api.post(
        "/qwen-exo/reflection-memory/pending/cancel",
        json={"conversation_keys": ["conversation-1"]},
    )

    assert listing.status_code == 200
    assert listing.json()["pending"][0]["trajectory_id"] == "resp-1"
    assert started.status_code == 202
    assert started.json()["started_count"] == 1
    assert cancelled.status_code == 200
    assert runtime.pending_reflections == {}


def test_source_listing_searches_content_and_batch_actions_are_explicit():
    runtime = FakeRuntime()
    runtime.documents["other.md"] = "# Other\nneedle in body"
    api = client(runtime)

    listing = api.get("/qwen-exo/knowledge", params={"q": "needle in body"})
    compiled = api.post(
        "/qwen-exo/sources/compile",
        json={"lane": "knowledge", "relative_paths": ["other.md"]},
    )
    deleted = api.post(
        "/qwen-exo/sources/delete",
        json={"lane": "knowledge", "relative_paths": ["other.md"]},
    )

    assert [item["relative_path"] for item in listing.json()["documents"]] == [
        "other.md"
    ]
    assert compiled.status_code == 200
    assert compiled.json()["requested_paths"] == ["other.md"]
    assert deleted.status_code == 200
    assert deleted.json()["deleted_count"] == 1
    assert "other.md" not in runtime.documents


def test_policy_data_surface_is_singleton_and_has_no_tags():
    runtime = FakeRuntime()
    api = client(runtime)

    listing = api.get("/qwen-exo/policydata")
    content = api.get("/qwen-exo/policydata/policy.md")
    updated = api.put(
        "/qwen-exo/policydata/policy.md",
        json={"content": "# Updated Policy", "tags": ["ignored"]},
    )
    rejected = api.put(
        "/qwen-exo/policydata/new.md",
        json={"content": "# New Policy", "tags": ["ignored"]},
    )

    assert listing.status_code == 200
    assert listing.json()["source_digest"] == "policy-digest"
    assert "content" not in listing.json()["documents"][0]
    assert content.json()["content"] == "# Policy"
    assert updated.status_code == 200
    assert updated.json()["tags"] == []
    assert runtime.policy_documents == {"policy.md": "# Updated Policy"}
    assert rejected.status_code == 409
    assert "one personality document" in rejected.json()["detail"]


def test_trajectory_preview_requires_explicit_create_then_supports_tagged_edit(
    tmp_path: Path,
):
    runtime = FakeRuntime()
    runtime.config.state_directory = tmp_path / "state"
    api = client(runtime)
    messages = [
        {"role": "user", "content": "开始任务"},
        {"role": "assistant", "content": "已完成"},
    ]
    encoded = base64.b64encode(
        json.dumps({"session": {"messages": messages}}, ensure_ascii=False).encode()
    ).decode()

    preview = api.post(
        "/qwen-exo/trajectories/preview",
        json={"filename": "uploaded.json", "content_base64": encoded},
    )

    assert preview.status_code == 200
    assert preview.json()["draft"] is True
    assert preview.json()["suggested_name"] == "uploaded"
    assert api.get("/qwen-exo/trajectories").json()["trajectories"] == []

    missing_name = api.post(
        "/qwen-exo/trajectories",
        json={"name": "", "content": preview.json()["content"], "tags": ["coding"]},
    )
    missing_tags = api.post(
        "/qwen-exo/trajectories",
        json={"name": "custom-run", "content": preview.json()["content"], "tags": []},
    )
    created = api.post(
        "/qwen-exo/trajectories",
        json={
            "name": "custom-run",
            "content": preview.json()["content"],
            "tags": ["coding", "success"],
        },
    )

    assert missing_name.status_code == 422
    assert missing_tags.status_code == 422
    assert created.status_code == 201
    assert created.json()["tags"] == ["coding", "success"]

    document = api.get("/qwen-exo/trajectories/custom-run").json()
    edited_content = json.loads(document["content"])
    edited_content["session"]["messages"].append(
        {"role": "user", "content": "复查结果"}
    )
    edited = api.put(
        "/qwen-exo/trajectories/custom-run",
        json={
            "content": json.dumps(edited_content, ensure_ascii=False),
            "tags": ["reviewed"],
            "name": "custom-run-renamed",
        },
    )
    assert edited.status_code == 200
    assert edited.json()["name"] == "custom-run-renamed"
    assert edited.json()["messages"] == 3
    assert edited.json()["tags"] == ["reviewed"]
    assert api.get("/qwen-exo/trajectories/custom-run").status_code == 404
    renamed = api.get("/qwen-exo/trajectories/custom-run-renamed")
    assert renamed.status_code == 200
    assert renamed.json()["tags"] == ["reviewed"]

    deleted = api.delete("/qwen-exo/trajectories/custom-run-renamed")
    assert deleted.status_code == 200
    assert api.get("/qwen-exo/trajectories").json()["trajectories"] == []


def test_training_selection_controls_membership_and_tracks_trajectory_changes(
    tmp_path: Path,
):
    runtime = FakeRuntime()
    runtime.config.state_directory = tmp_path / "state"
    editors_root = runtime.config.state_directory / "activation-editors"
    editors_root.mkdir(parents=True)
    (editors_root / "active.json").write_text(
        json.dumps({"editor": "combined-trajectories", "strength": 1.0}),
        encoding="utf-8",
    )
    api = client(runtime)

    messages = json.dumps(
        {
            "session": {
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "start"},
                    {
                        "role": "assistant",
                        "content": "first assistant action with enough detail",
                    },
                    {"role": "user", "content": "continue"},
                    {
                        "role": "assistant",
                        "content": "second assistant action with enough detail",
                    },
                ]
            }
        },
        ensure_ascii=False,
    )
    for name, tag in (("first", "alpha"), ("second", "beta")):
        assert (
            api.post(
                "/qwen-exo/trajectories",
                json={"name": name, "content": messages, "tags": [tag]},
            ).status_code
            == 201
        )

    initial = api.get("/qwen-exo/editors/training-selection")
    selected = api.put(
        "/qwen-exo/editors/training-selection",
        json={"names": ["first", "second"]},
    )

    assert initial.status_code == 200
    assert initial.json()["names"] == []
    assert selected.status_code == 200
    assert selected.json()["names"] == ["first", "second"]
    assert [source["name"] for source in selected.json()["sources"]] == [
        "first",
        "second",
    ]
    assert selected.json()["applied"] is False
    assert not (editors_root / "active.json").exists()

    renamed = api.patch("/qwen-exo/trajectories/first", json={"name": "renamed"})
    assert renamed.status_code == 200
    assert api.get("/qwen-exo/editors/training-selection").json()["names"] == [
        "renamed",
        "second",
    ]

    deleted = api.delete("/qwen-exo/trajectories/second")
    assert deleted.status_code == 200
    assert api.get("/qwen-exo/editors/training-selection").json()["names"] == [
        "renamed"
    ]
    assert api.post("/qwen-exo/editors/apply", json={"names": []}).status_code == 404


def test_joint_editor_training_endpoint_queues_selected_sources_and_restarts(
    tmp_path: Path, monkeypatch
):
    runtime = FakeRuntime()
    runtime.config.state_directory = tmp_path / "state"
    config_path = tmp_path / "service-config.json"
    ServiceConfigStore(config_path).ensure([])
    monkeypatch.setenv("QWEN_EXO_SERVICE_CONFIG", str(config_path))
    monkeypatch.setenv("QWEN_EXO_MANAGED_RESTART", "1")
    restart_delays = []
    monkeypatch.setattr(
        router_module,
        "request_managed_restart",
        lambda delay_seconds=1.25: restart_delays.append(delay_seconds),
    )
    api = client(runtime)
    messages = {
        "session": {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "start"},
                {
                    "role": "assistant",
                    "content": "first assistant action with enough detail",
                },
                {"role": "user", "content": "continue"},
                {
                    "role": "assistant",
                    "content": "second assistant action with enough detail",
                },
            ]
        }
    }
    for name in ("first", "second"):
        assert (
            api.post(
                "/qwen-exo/trajectories",
                json={
                    "name": name,
                    "content": json.dumps(messages),
                    "tags": ["training"],
                },
            ).status_code
            == 201
        )
    selected = api.put(
        "/qwen-exo/editors/training-selection",
        json={"names": ["first", "second"]},
    )

    queued = api.post("/qwen-exo/editors/train")
    status = api.get("/qwen-exo/editors/training")
    duplicate = api.post("/qwen-exo/editors/train")

    assert selected.status_code == 200
    assert queued.status_code == 202
    assert queued.json()["status"] == "queued"
    assert queued.json()["job"]["editor"] == "combined-trajectories"
    assert queued.json()["job"]["trajectories"] == ["first", "second"]
    assert queued.json()["restart_requested"] is True
    assert status.json()["job"]["sample_count"] == 4
    assert duplicate.status_code == 409
    assert restart_delays == [2.0]


def test_knowledge_preview_is_draft_until_named_tagged_write():
    runtime = FakeRuntime()
    api = client(runtime)
    before = dict(runtime.documents)
    encoded = base64.b64encode(b"# Uploaded\nDraft body").decode()

    preview = api.post(
        "/qwen-exo/knowledge/preview",
        json={"files": [{"filename": "upload.md", "content_base64": encoded}]},
    )

    assert preview.status_code == 200
    draft = preview.json()["drafts"][0]
    assert draft["suggested_path"] == "uploads/upload.md"
    assert runtime.documents == before

    created = api.put(
        "/qwen-exo/knowledge/custom/reference.md",
        json={"content": draft["content"], "tags": ["sdk", "wfp"]},
    )
    assert created.status_code == 200
    assert created.json()["tags"] == ["sdk", "wfp"]
    assert "custom/reference.md" in runtime.documents

    deleted = api.delete("/qwen-exo/knowledge/custom/reference.md")
    assert deleted.status_code == 200
    assert runtime.documents == before


def test_source_reindex_is_direct_and_verification_route_is_removed():
    api = client(FakeRuntime())

    knowledge = api.post("/qwen-exo/knowledge/reindex")
    policy = api.post("/qwen-exo/policydata/reindex")

    assert api.post("/qwen-exo/admin/verify").status_code == 404
    assert knowledge.json()["document_count"] == 1
    assert knowledge.json()["source_digest"] == "digest"
    assert policy.json()["document_count"] == 1
    assert policy.json()["source_digest"] == "policy-digest"


def test_recall_trace_contract_and_compatibility_page_are_served():
    api = client(FakeRuntime())

    trace = api.get("/qwen-exo/recall-trace")
    page = api.get("/recall-trace")
    cleared = api.delete("/qwen-exo/recall-trace")

    assert trace.status_code == 200
    assert trace.json()["schema"] == "inflight-memory-visualization-v2"
    assert trace.json()["turns"] == [{"turn_id": "request-1"}]
    assert page.status_code == 200
    assert "飞行中召回轨迹" in page.text
    assert cleared.json() == {"status": "cleared", "turns": 0}


def test_single_page_console_and_hashed_assets_are_served():
    api = client(FakeRuntime())

    workspace = api.get("/qwen-exo/")
    admin = api.get("/qwen-exo/admin")
    script_path = re.search(r'src="(/qwen-exo/assets/[^"]+\.js)"', workspace.text)
    style_path = re.search(r'href="(/qwen-exo/assets/[^"]+\.css)"', workspace.text)

    assert workspace.status_code == 200
    assert admin.status_code == 200
    assert workspace.text == admin.text
    assert "QWEN EXO 控制台" in workspace.text
    assert script_path is not None
    assert style_path is not None
    script = api.get(script_path.group(1))
    stylesheet = api.get(style_path.group(1))
    assert script.status_code == 200
    assert stylesheet.status_code == 200
    assert stylesheet.headers["content-type"].startswith("text/css")
    assert "service-config" in script.text
    assert api.get("/qwen-exo/workspace.js").status_code == 404


def test_service_config_update_is_revisioned_and_requests_managed_restart(
    tmp_path: Path, monkeypatch
):
    path = tmp_path / "service-config.json"
    store = ServiceConfigStore(path)
    initial = store.ensure([])
    restart_calls = []
    monkeypatch.setenv("QWEN_EXO_SERVICE_CONFIG", str(path))
    monkeypatch.setenv("QWEN_EXO_MANAGED_RESTART", "1")
    monkeypatch.setattr(
        router_module,
        "request_managed_restart",
        lambda: restart_calls.append(True),
    )
    api = client(FakeRuntime())

    response = api.put(
        "/qwen-exo/service-config",
        json={
            "values": {"qwen_exo_max_candidates": 12},
            "expected_revision": initial["revision"],
        },
    )

    assert response.status_code == 202
    assert response.json()["restart_requested"] is True
    assert response.json()["values"]["qwen_exo_max_candidates"] == 12
    assert response.json()["pending_restart"] is True
    assert restart_calls == [True]


def test_service_config_update_refuses_unmanaged_process(tmp_path: Path, monkeypatch):
    path = tmp_path / "service-config.json"
    initial = ServiceConfigStore(path).ensure([])
    monkeypatch.setenv("QWEN_EXO_SERVICE_CONFIG", str(path))
    monkeypatch.delenv("QWEN_EXO_MANAGED_RESTART", raising=False)

    response = client(FakeRuntime()).put(
        "/qwen-exo/service-config",
        json={
            "values": {"qwen_exo_max_candidates": 12},
            "expected_revision": initial["revision"],
        },
    )

    assert response.status_code == 409
    assert ServiceConfigStore(path).public_document()["revision"] == initial["revision"]


def test_service_config_validation_returns_structured_422_without_restart(
    tmp_path: Path, monkeypatch
):
    path = tmp_path / "service-config.json"
    initial = ServiceConfigStore(path).ensure([])
    restart_calls = []
    monkeypatch.setenv("QWEN_EXO_SERVICE_CONFIG", str(path))
    monkeypatch.setenv("QWEN_EXO_MANAGED_RESTART", "1")
    monkeypatch.setattr(
        router_module,
        "request_managed_restart",
        lambda: restart_calls.append(True),
    )

    response = client(FakeRuntime()).put(
        "/qwen-exo/service-config",
        json={
            "values": {"max_prefill_tokens": 999999},
            "expected_revision": initial["revision"],
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "code": "above_maximum",
        "field": "max_prefill_tokens",
        "message": "不得大于 262144",
    }
    assert restart_calls == []
    assert ServiceConfigStore(path).public_document()["revision"] == initial["revision"]


def test_model_catalog_routes_switch_revisioned_profile_and_restart(
    tmp_path: Path, monkeypatch
):
    models = tmp_path / "models"
    models.mkdir()
    dense = models / "dense"
    moe = models / "moe"
    _write_catalog_model(dense, "Qwen3_5ForConditionalGeneration")
    _write_catalog_model(moe, "Qwen3_5MoeForConditionalGeneration")
    data = tmp_path / "data"
    store = ModelCatalogStore([models], data)
    initial = store.ensure(dense)
    moe_fingerprint = next(
        model["model_fingerprint"]
        for model in store.public_document()["models"]
        if model["model_path"] == str(moe.resolve())
    )
    restart_calls = []
    monkeypatch.setenv("QWEN_EXO_MODEL_CATALOG_ROOTS", str(models))
    monkeypatch.setenv(
        "QWEN_EXO_MODEL_CATALOG_CONFIG", str(data / "model-catalog.json")
    )
    monkeypatch.setenv("QWEN_EXO_MODEL_DATA_ROOT", str(data))
    monkeypatch.setenv("QWEN_EXO_MANAGED_RESTART", "1")
    monkeypatch.setattr(
        router_module,
        "request_managed_restart",
        lambda: restart_calls.append(True),
    )
    runtime = FakeRuntime()
    runtime.model_identity = SimpleNamespace(
        fingerprint=initial["active_model_fingerprint"]
    )
    api = client(runtime)

    listing = api.get("/qwen-exo/models")
    switched = api.put(
        "/qwen-exo/models/active",
        json={
            "model_fingerprint": moe_fingerprint,
            "expected_revision": initial["revision"],
            "clone_sources": True,
        },
    )

    assert listing.status_code == 200
    assert (
        listing.json()["running_model_fingerprint"]
        == initial["active_model_fingerprint"]
    )
    assert switched.status_code == 202
    assert switched.json()["active_model_fingerprint"] == moe_fingerprint
    assert switched.json()["restart_requested"] is True
    assert restart_calls == [True]


def test_model_catalog_route_rejects_stale_revision_without_restart(
    tmp_path: Path, monkeypatch
):
    models = tmp_path / "models"
    models.mkdir()
    dense = models / "dense"
    moe = models / "moe"
    _write_catalog_model(dense, "Qwen3_5ForConditionalGeneration")
    _write_catalog_model(moe, "Qwen3_5MoeForConditionalGeneration")
    data = tmp_path / "data"
    store = ModelCatalogStore([models], data)
    store.ensure(dense)
    moe_fingerprint = next(
        model["model_fingerprint"]
        for model in store.public_document()["models"]
        if model["model_path"] == str(moe.resolve())
    )
    restart_calls = []
    monkeypatch.setenv("QWEN_EXO_MODEL_CATALOG_ROOTS", str(models))
    monkeypatch.setenv(
        "QWEN_EXO_MODEL_CATALOG_CONFIG", str(data / "model-catalog.json")
    )
    monkeypatch.setenv("QWEN_EXO_MODEL_DATA_ROOT", str(data))
    monkeypatch.setenv("QWEN_EXO_MANAGED_RESTART", "1")
    monkeypatch.setattr(
        router_module,
        "request_managed_restart",
        lambda: restart_calls.append(True),
    )

    response = client(FakeRuntime()).put(
        "/qwen-exo/models/active",
        json={
            "model_fingerprint": moe_fingerprint,
            "expected_revision": "stale",
            "clone_sources": True,
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "revision_conflict"
    assert restart_calls == []


def test_tensor_bank_reindex_is_model_native():
    response = client(FakeRuntime()).post("/qwen-exo/tensor-bank/reindex")

    assert response.status_code == 200
    assert response.json() == {
        "document_state_count": 3,
        "model_native_documents": 3,
        "complete_gdn_document_states": 3,
    }


def test_tensor_bank_reindex_returns_actionable_compile_failure():
    runtime = FakeRuntime()

    async def fail_reindex():
        raise TensorBankCompileError(
            "salient_span_budget_exceeded",
            "dense.md",
            "merged spans require 2112 tokens",
            details={"salient_tokens": 2112, "salient_token_budget": 2048},
            hint="Split or simplify the document at a semantic boundary.",
        )

    runtime.reindex_tensor_bank = fail_reindex
    response = client(runtime).post("/qwen-exo/tensor-bank/reindex")

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "code": "salient_span_budget_exceeded",
        "relative_path": "dense.md",
        "message": "merged spans require 2112 tokens",
        "details": {"salient_tokens": 2112, "salient_token_budget": 2048},
        "hint": "Split or simplify the document at a semantic boundary.",
    }


def test_knowledge_ingest_hot_updates_without_restart():
    response = client(FakeRuntime()).post(
        "/qwen-exo/knowledge/ingest",
        json={
            "files": [
                {
                    "filename": "guide.txt",
                    "content_base64": "SGVsbG8=",
                }
            ]
        },
    )

    assert response.status_code == 200
    assert response.json()["hot_updated"] is True
    assert response.json()["restart_required"] is False
    assert response.json()["files"][0]["documents"][0]["relative_path"] == (
        "uploads/guide.md"
    )


def test_knowledge_ingest_returns_structured_cleaning_error():
    runtime = FakeRuntime()

    async def fail_ingest(_files):
        raise KnowledgeIngestError(
            "invalid_json",
            "JSON 解析失败：第 3 行，第 9 列",
            filename="broken.json",
        )

    runtime.ingest_knowledge_files = fail_ingest
    response = client(runtime).post(
        "/qwen-exo/knowledge/ingest",
        json={
            "files": [
                {
                    "filename": "broken.json",
                    "content_base64": "e30=",
                }
            ]
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "code": "invalid_json",
        "message": "JSON 解析失败：第 3 行，第 9 列",
        "details": {},
        "filename": "broken.json",
    }
