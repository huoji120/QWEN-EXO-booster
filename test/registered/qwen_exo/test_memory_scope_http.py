"""Managed-key memory ownership at real HTTP and retained-response boundaries."""

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from qwen_exo_booster.api_keys import ApiKeyStore
from sglang.srt.entrypoints.http_server import (
    _managed_warmup_headers,
    v1_cancel_responses,
    v1_retrieve_responses,
)
from sglang.srt.entrypoints.openai.protocol import ResponsesRequest, ResponsesResponse
from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses
from sglang.srt.utils.auth import add_api_key_middleware


class MemoryScopeHttpTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.keys = ApiKeyStore(Path(self.temporary.name) / "keys.json")
        self.shared = self.keys.create("shared")
        self.private_a = self.keys.create("private-a", "private")
        self.private_b = self.keys.create("private-b", "private")
        self.aborted = []
        self.serving = OpenAIServingResponses.__new__(OpenAIServingResponses)
        self.serving.response_store_lock = asyncio.Lock()
        self.serving.response_store = {}
        self.serving.msg_store = {}
        self.serving.background_tasks = {}
        self.serving.tokenizer_manager = SimpleNamespace(
            abort_request=lambda *, rid: self.aborted.append(rid)
        )
        for name, owner in (("shared", None), ("a", self.private_a["id"]), ("b", self.private_b["id"])):
            request = ResponsesRequest(model="fixture", input="private task", request_id=f"resp_{name}")
            request._qwen_exo_memory_key_id = owner
            request._qwen_exo_memory_scope = "private" if owner else "shared"
            self.serving.response_store[request.request_id] = ResponsesResponse.from_request(
                request, {}, "fixture", 1, [], "in_progress", None
            )
        app = FastAPI()
        app.state.openai_serving_responses = self.serving
        app.add_api_route("/v1/responses/{response_id}", v1_retrieve_responses, methods=["GET"])
        app.add_api_route("/v1/responses/{response_id}/cancel", v1_cancel_responses, methods=["POST"])
        add_api_key_middleware(
            app, api_key=None, admin_api_key=None,
            dynamic_authorizer=self.keys.authorize_request,
            dynamic_identity_resolver=self.keys.resolve_authorization,
        )
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    @staticmethod
    def headers(key):
        return {"Authorization": f"Bearer {key['token']}"}

    def test_private_response_cannot_be_read_or_cancelled_by_other_scopes(self):
        for key in (self.shared, self.private_b):
            with self.subTest(key=key["label"]):
                self.assertEqual(self.client.get("/v1/responses/resp_a", headers=self.headers(key)).status_code, 404)
                self.assertEqual(self.client.post("/v1/responses/resp_a/cancel", headers=self.headers(key)).status_code, 404)
        self.assertEqual(self.serving.response_store["resp_a"].status, "in_progress")
        self.assertEqual(self.aborted, [])
        self.assertEqual(self.client.get("/v1/responses/resp_a", headers=self.headers(self.private_a)).status_code, 200)
        response = self.client.post("/v1/responses/resp_a/cancel", headers=self.headers(self.private_a))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "cancelled")
        self.assertEqual(self.aborted, ["resp_a"])

    def test_mode_change_does_not_expose_old_private_response(self):
        self.keys.update_memory_scope(self.private_a["id"], "shared")
        self.assertEqual(self.client.get("/v1/responses/resp_a", headers=self.headers(self.private_a)).status_code, 404)
        self.assertEqual(self.client.get("/v1/responses/resp_shared", headers=self.headers(self.private_a)).status_code, 200)
        self.keys.update_memory_scope(self.private_a["id"], "private")
        self.assertEqual(self.client.get("/v1/responses/resp_a", headers=self.headers(self.private_a)).status_code, 200)
        self.keys.revoke(self.private_a["id"])
        self.assertEqual(self.client.get("/v1/responses/resp_a", headers=self.headers(self.private_a)).status_code, 401)

    def test_client_cannot_supply_server_memory_identity(self):
        request = ResponsesRequest.model_validate({
            "model": "fixture", "input": "hello",
            "_qwen_exo_memory_key_id": self.private_a["id"],
            "_qwen_exo_memory_scope": "super",
        })
        self.assertIsNone(request._qwen_exo_memory_key_id)
        self.assertEqual(request._qwen_exo_memory_scope, "shared")
        request._qwen_exo_memory_key_id = self.private_a["id"]
        request._qwen_exo_memory_scope = "private"
        self.assertEqual(request.model_copy()._qwen_exo_memory_key_id, self.private_a["id"])
        self.assertNotIn("_qwen_exo_memory_key_id", request.model_dump())
        self.assertNotIn("_qwen_exo_memory_scope", request.model_dump())

    def test_previous_response_replay_requires_same_scope(self):
        self.serving.use_harmony = False
        self.serving.tool_server = None
        self.serving.response_store["resp_a"].status = "completed"
        self.serving.msg_store["resp_a"] = [{"role": "user", "content": "private task"}]
        request = ResponsesRequest(model="fixture", input="continue", previous_response_id="resp_a")
        for owner in (None, self.private_b["id"]):
            request._qwen_exo_memory_key_id = owner
            request._qwen_exo_memory_scope = "private" if owner else "shared"
            result = asyncio.run(self.serving.prevalidate_qwen_exo_request(request))
            self.assertEqual(result.status_code, 404)
        request._qwen_exo_memory_key_id = self.private_a["id"]
        request._qwen_exo_memory_scope = "private"
        self.assertIsNone(asyncio.run(self.serving.prevalidate_qwen_exo_request(request)))

    def test_mode_downgrade_cannot_replay_same_owner_old_response(self):
        for mode in ("private_only", "super"):
            self.keys.update_memory_scope(self.private_a["id"], mode)
            self.assertEqual(self.client.get("/v1/responses/resp_a", headers=self.headers(self.private_a)).status_code, 404)
            self.assertEqual(self.client.post("/v1/responses/resp_a/cancel", headers=self.headers(self.private_a)).status_code, 404)
        self.assertEqual(self.aborted, [])
        self.keys.update_memory_scope(self.private_b["id"], "super")
        self.assertEqual(self.client.get("/v1/responses/resp_a", headers=self.headers(self.private_b)).status_code, 404)
        self.keys.update_memory_scope(self.private_a["id"], "private")
        self.assertEqual(self.client.get("/v1/responses/resp_a", headers=self.headers(self.private_a)).status_code, 200)

    def test_warmup_key_is_valid_only_during_authenticated_request(self):
        args = SimpleNamespace(qwen_exo_api_key_store=str(self.keys.path))
        before = self.keys.listing()["keys"]
        self.assertFalse(self.keys.authorize_request("POST", "/v1/chat/completions", None))
        with _managed_warmup_headers(args, "/v1/chat/completions", {}) as headers:
            authorization = headers["Authorization"]
            self.assertTrue(self.keys.authorize_request("POST", "/v1/chat/completions", authorization))
            self.assertEqual(self.keys.resolve_authorization(authorization)["memory_scope"], "shared")
        self.assertFalse(self.keys.authorize_request("POST", "/v1/chat/completions", authorization))
        self.assertEqual(self.keys.listing()["keys"], before)
        with self.assertRaisesRegex(RuntimeError, "warmup failed"):
            with _managed_warmup_headers(args, "/v1/chat/completions", {}) as headers:
                failed_authorization = headers["Authorization"]
                raise RuntimeError("warmup failed")
        self.assertFalse(self.keys.authorize_request("POST", "/v1/chat/completions", failed_authorization))
        self.assertEqual(self.keys.listing()["keys"], before)
        self.assertTrue(self.keys.authorize_request("POST", "/v1/responses", self.headers(self.private_a)["Authorization"]))


if __name__ == "__main__":
    unittest.main()
