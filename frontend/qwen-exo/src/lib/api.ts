import type {
  ActiveEditor,
  ApiKeyListing,
  ApiKeyInfo,
  CreatedApiKey,
  EditorInfo,
  EditorTrainingStatus,
  HealthStatus,
  ModelCatalog,
  KnowledgeDraft,
  RecallTrace,
  RequestTraceListing,
  PendingReflectionMemory,
  RuntimeStatus,
  ServiceConfig,
  SourceListing,
  TelemetryEvent,
  TrajectoryDetail,
  TrajectoryDraft,
  TrajectoryInfo,
  TrainingSelectionStatus,
} from "@/lib/types";
import { translate as t } from "@/lib/i18n";

const API = "/qwen-exo";
const MANAGED_API_KEY = "qwen-exo-managed-api-key";

export function setManagedApiKey(token: string | null) {
  if (token) window.sessionStorage.setItem(MANAGED_API_KEY, token);
  else window.sessionStorage.removeItem(MANAGED_API_KEY);
}

export function getManagedApiKey() {
  return window.sessionStorage.getItem(MANAGED_API_KEY);
}

function managedAuthorizationHeader(): Record<string, string> {
  const token = getManagedApiKey();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export class ApiError extends Error {
  status: number;
  detail: unknown;

  constructor(message: string, status: number, detail?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

function errorMessage(payload: unknown, fallback: string) {
  if (!payload || typeof payload !== "object") return fallback;
  const object = payload as Record<string, unknown>;
  const detail = object.detail;
  if (typeof detail === "string") return detail;
  if (detail && typeof detail === "object") {
    const message = (detail as Record<string, unknown>).message;
    if (typeof message === "string") return message;
  }
  const error = object.error;
  if (error && typeof error === "object") {
    const message = (error as Record<string, unknown>).message;
    if (typeof message === "string") return message;
  }
  if (typeof object.message === "string") return object.message;
  return fallback;
}

export async function apiFetch(path: string, init: RequestInit = {}) {
  const response = await fetch(`${API}${path}`, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init.body ? { "Content-Type": "application/json" } : {}),
      ...init.headers,
    },
  });
  if (response.ok) return response;
  let payload: unknown = null;
  try {
    payload = await response.json();
  } catch {
    // Preserve the HTTP fallback.
  }
  throw new ApiError(
    errorMessage(
      payload,
      t("请求失败（HTTP {status}）", { status: response.status }),
    ),
    response.status,
    payload,
  );
}

export async function getStatus() {
  return (await (await apiFetch("/status")).json()) as RuntimeStatus;
}

export async function getHealth() {
  return (await (await apiFetch("/health")).json()) as HealthStatus;
}

export async function getModelCatalog() {
  return (await (await apiFetch("/models")).json()) as ModelCatalog;
}

export async function selectActiveModel(
  modelFingerprint: string,
  expectedRevision: string,
) {
  return (await (
    await apiFetch("/models/active", {
      method: "PUT",
      body: JSON.stringify({
        model_fingerprint: modelFingerprint,
        expected_revision: expectedRevision,
      }),
    })
  ).json()) as ModelCatalog & { restart_requested: boolean };
}

export async function getTelemetry(limit = 100, requestId?: string) {
  const params = new URLSearchParams({ limit: String(limit) });
  if (requestId) params.set("request_id", requestId);
  const response = await apiFetch(`/telemetry?${params}`);
  return (await response.json()) as {
    events?: TelemetryEvent[];
    redacted?: boolean;
  };
}

export async function getRequestTraces(limit = 50, q = "") {
  const params = new URLSearchParams({ limit: String(limit) });
  if (q) params.set("q", q);
  return (await (
    await apiFetch(`/request-traces?${params}`)
  ).json()) as RequestTraceListing;
}

export async function getRecallTrace(limit = 10) {
  return (await (
    await apiFetch(`/recall-trace?limit=${limit}`)
  ).json()) as RecallTrace;
}

export async function clearTelemetry() {
  return (await (
    await apiFetch("/telemetry/clear", { method: "POST" })
  ).json()) as { cleared: boolean } & Record<string, unknown>;
}

export async function listSources(
  lane: "knowledge" | "policydata",
  query = "",
) {
  const params = new URLSearchParams();
  if (query.trim()) params.set("q", query.trim());
  const suffix = params.size ? `?${params}` : "";
  return (await (await apiFetch(`/${lane}${suffix}`)).json()) as SourceListing;
}

export type ReflectionOrganizationResult = {
  status: "merged" | "kept_distinct" | "no_high_qk_pairs" | "not_needed";
  document_count?: number;
  document_count_before?: number;
  document_count_after?: number;
  high_qk_pair_count: number;
  review_count?: number;
  merge_operation_count?: number;
  merged_document_count?: number;
  target_document_path?: string;
  merged_document_paths?: string[];
  removed_document_paths?: string[];
  hot_updated?: boolean;
  restart_required?: boolean;
};

export type ReflectionOrganizationJobStatus = {
  job_id: string | null;
  status: "idle" | "queued" | "running" | "succeeded" | "failed";
  stage:
    | "idle"
    | "queued"
    | "scanning"
    | "qk_retrieval"
    | "model_review"
    | "publishing"
    | "completed"
    | "failed";
  progress: number;
  message: string;
  queued_at?: number | null;
  started_at?: number | null;
  updated_at?: number | null;
  finished_at?: number | null;
  details?: Record<string, number | string | boolean | null>;
  result?: ReflectionOrganizationResult | null;
  error?: string | null;
};

export async function startReflectionMemoryOrganization() {
  return (await (
    await apiFetch("/reflection-memory/organize", { method: "POST" })
  ).json()) as ReflectionOrganizationJobStatus;
}

export async function getReflectionMemoryOrganizationStatus() {
  return (await (
    await apiFetch("/reflection-memory/organize")
  ).json()) as ReflectionOrganizationJobStatus;
}

export async function listPendingReflectionMemories() {
  return (await (await apiFetch("/reflection-memory/pending")).json()) as {
    pending: PendingReflectionMemory[];
  };
}

export async function startPendingReflections(conversationKeys: string[]) {
  return (await (
    await apiFetch("/reflection-memory/pending/reflect", {
      method: "POST",
      body: JSON.stringify({ conversation_keys: conversationKeys }),
    })
  ).json()) as { started: string[]; started_count: number };
}

export async function cancelPendingReflections(conversationKeys: string[]) {
  return (await (
    await apiFetch("/reflection-memory/pending/cancel", {
      method: "POST",
      body: JSON.stringify({ conversation_keys: conversationKeys }),
    })
  ).json()) as { cancelled: string[]; cancelled_count: number };
}

export async function compileSourceDocuments(
  lane: "knowledge" | "policydata",
  relativePaths: string[],
) {
  return await (
    await apiFetch("/sources/compile", {
      method: "POST",
      body: JSON.stringify({ lane, relative_paths: relativePaths }),
    })
  ).json();
}

export async function deleteSourceDocuments(
  lane: "knowledge" | "policydata",
  relativePaths: string[],
) {
  return await (
    await apiFetch("/sources/delete", {
      method: "POST",
      body: JSON.stringify({ lane, relative_paths: relativePaths }),
    })
  ).json();
}

export type KnowledgeIngestResult = {
  hot_updated: boolean;
  restart_required: boolean;
  source_digest: string;
  document_count: number;
  replaced_document_count: number;
  files: {
    filename: string;
    documents: {
      relative_path: string;
      token_count: number;
      byte_count: number;
      sha256: string;
      changes: string[];
    }[];
  }[];
  tensor_bank: Record<string, unknown>;
};

export async function ingestKnowledgeFiles(
  files: { filename: string; content_base64: string }[],
) {
  return (await (
    await apiFetch("/knowledge/ingest", {
      method: "POST",
      body: JSON.stringify({ files }),
    })
  ).json()) as KnowledgeIngestResult;
}

export async function previewKnowledgeFiles(
  files: { filename: string; content_base64: string }[],
) {
  return (await (
    await apiFetch("/knowledge/preview", {
      method: "POST",
      body: JSON.stringify({ files }),
    })
  ).json()) as { drafts: KnowledgeDraft[]; persisted: false };
}

function encodeSourcePath(path: string) {
  return path.split("/").map(encodeURIComponent).join("/");
}

export async function getSource(
  lane: "knowledge" | "policydata",
  path: string,
) {
  return (await (
    await apiFetch(`/${lane}/${encodeSourcePath(path)}`)
  ).json()) as {
    relative_path: string;
    content: string;
    tags: string[];
  };
}

export async function saveSource(
  lane: "knowledge" | "policydata",
  path: string,
  content: string,
  tags: string[],
) {
  return await (
    await apiFetch(`/${lane}/${encodeSourcePath(path)}`, {
      method: "PUT",
      body: JSON.stringify({ content, tags }),
    })
  ).json();
}

export async function deleteSource(
  lane: "knowledge" | "policydata",
  path: string,
) {
  await apiFetch(`/${lane}/${encodeSourcePath(path)}`, { method: "DELETE" });
}

export async function reindexSource(lane: "knowledge" | "policydata") {
  return await (await apiFetch(`/${lane}/reindex`, { method: "POST" })).json();
}

export async function reindexTensorBank() {
  return await (
    await apiFetch("/tensor-bank/reindex", { method: "POST" })
  ).json();
}

export async function listEditors() {
  return (await (await apiFetch("/editors")).json()) as {
    editors: EditorInfo[];
    active: ActiveEditor | null;
  };
}

export async function getTrainingSelection() {
  return (await (
    await apiFetch("/editors/training-selection")
  ).json()) as TrainingSelectionStatus;
}

export async function updateTrainingSelection(names: string[]) {
  const normalized = names
    .map((name) => name.trim().toLowerCase())
    .filter(Boolean);
  return (await (
    await apiFetch("/editors/training-selection", {
      method: "PUT",
      body: JSON.stringify({ names: normalized }),
    })
  ).json()) as TrainingSelectionStatus;
}

export async function getEditorTraining() {
  return (await (
    await apiFetch("/editors/training")
  ).json()) as EditorTrainingStatus;
}

export async function trainSelectedTrajectories() {
  return (await (
    await apiFetch("/editors/train", { method: "POST" })
  ).json()) as EditorTrainingStatus;
}

export async function listTrajectories() {
  return (await (await apiFetch("/trajectories")).json()) as {
    trajectories: TrajectoryInfo[];
  };
}

export async function previewTrajectory(
  filename: string,
  contentBase64: string,
) {
  return (await (
    await apiFetch("/trajectories/preview", {
      method: "POST",
      body: JSON.stringify({ filename, content_base64: contentBase64 }),
    })
  ).json()) as TrajectoryDraft;
}

export async function createTrajectory(
  name: string,
  content: string,
  tags: string[],
) {
  return (await (
    await apiFetch("/trajectories", {
      method: "POST",
      body: JSON.stringify({ name, content, tags }),
    })
  ).json()) as {
    name: string;
    messages: number;
    bytes: number;
    tags: string[];
  };
}

export async function getTrajectory(name: string) {
  return (await (
    await apiFetch(`/trajectories/${encodeURIComponent(name)}`)
  ).json()) as TrajectoryDetail;
}

export async function saveTrajectory(
  name: string,
  content: string,
  tags: string[],
  newName?: string,
) {
  const payload: Record<string, unknown> = { content, tags };
  if (newName && newName !== name) payload.name = newName;
  return (await (
    await apiFetch(`/trajectories/${encodeURIComponent(name)}`, {
      method: "PUT",
      body: JSON.stringify(payload),
    })
  ).json()) as {
    name: string;
    messages: number;
    bytes: number;
    tags: string[];
  };
}

export async function renameTrajectory(name: string, newName: string) {
  return (await (
    await apiFetch(`/trajectories/${encodeURIComponent(name)}`, {
      method: "PATCH",
      body: JSON.stringify({ name: newName }),
    })
  ).json()) as TrajectoryDetail;
}

export async function deleteTrajectory(name: string) {
  await apiFetch(`/trajectories/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
}

export async function getServiceConfig() {
  return (await (await apiFetch("/service-config")).json()) as ServiceConfig;
}

export async function updateServiceConfig(
  values: Record<string, boolean | number | string>,
  expectedRevision: string,
) {
  return (await (
    await apiFetch("/service-config", {
      method: "PUT",
      body: JSON.stringify({ values, expected_revision: expectedRevision }),
    })
  ).json()) as ServiceConfig;
}

export async function getApiKeys() {
  return (await (await apiFetch("/api-keys")).json()) as ApiKeyListing;
}

export async function createApiKey(label: string) {
  return (await (
    await apiFetch("/api-keys", {
      method: "POST",
      body: JSON.stringify({ label }),
    })
  ).json()) as CreatedApiKey;
}

export async function revokeApiKey(keyId: string) {
  return (await (
    await apiFetch(`/api-keys/${encodeURIComponent(keyId)}`, {
      method: "DELETE",
    })
  ).json()) as ApiKeyInfo;
}

export async function getModelId() {
  const response = await fetch("/v1/models", {
    headers: { Accept: "application/json", ...managedAuthorizationHeader() },
  });
  if (!response.ok) throw new ApiError(t("无法读取模型列表"), response.status);
  const payload = (await response.json()) as { data?: { id?: string }[] };
  return payload.data?.[0]?.id || "default";
}

export type StreamHandlers = {
  onEvent: (event: string, payload: Record<string, any>) => void;
};

const TERMINAL_STREAM_EVENTS = new Set([
  "response.completed",
  "response.incomplete",
  "response.failed",
  "error",
]);

function consumeFrame(
  frame: string,
  handler: StreamHandlers["onEvent"],
): boolean {
  let type = "message";
  const data: string[] = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) type = line.slice(6).trim();
    if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
  }
  if (!data.length) return false;
  if (data[0] === "[DONE]") return true;
  const payload = JSON.parse(data.join("\n")) as Record<string, any>;
  const eventType = String(payload.type || type);
  handler(eventType, payload);
  return TERMINAL_STREAM_EVENTS.has(eventType);
}

export async function streamResponse(
  body: Record<string, unknown>,
  signal: AbortSignal,
  handlers: StreamHandlers,
) {
  const response = await fetch("/v1/responses", {
    method: "POST",
    headers: {
      Accept: "text/event-stream",
      "Content-Type": "application/json",
      ...managedAuthorizationHeader(),
    },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) {
    let payload: unknown = null;
    try {
      payload = await response.json();
    } catch {
      // Keep HTTP fallback.
    }
    throw new ApiError(
      errorMessage(
        payload,
        t("生成请求失败（HTTP {status}）", { status: response.status }),
      ),
      response.status,
      payload,
    );
  }
  if (!response.body) throw new Error(t("当前浏览器无法读取流式响应"));
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder
        .decode(value || new Uint8Array(), { stream: !done })
        .replaceAll("\r\n", "\n");
      let boundary = buffer.indexOf("\n\n");
      while (boundary >= 0) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        if (frame.trim() && consumeFrame(frame, handlers.onEvent)) return;
        boundary = buffer.indexOf("\n\n");
      }
      if (done) break;
    }
    if (buffer.trim()) consumeFrame(buffer, handlers.onEvent);
  } finally {
    try {
      await reader.cancel();
    } catch {
      // The stream may already be closed or aborted.
    }
    reader.releaseLock();
  }
}
