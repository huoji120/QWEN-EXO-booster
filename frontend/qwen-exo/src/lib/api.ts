import type {
  ActiveEditor,
  ApiKeyDeletion,
  ApiKeyListing,
  ApiKeyInfo,
  CreatedApiKey,
  EditorInfo,
  EditorTrainingStatus,
  HealthStatus,
  ModelCatalog,
  DocumentCategory,
  DreamRunDetail,
  DreamStatus,
  KnowledgeDraft,
  KnowledgeFile,
  KnowledgeFilePage,
  KnowledgeMoveResult,
  KnowledgeReadResponse,
  KnowledgeSearchResponse,
  KnowledgeSummaryRecord,
  KnowledgeSummaryStatus,
  KnowledgeTree,
  RequestTraceListing,
  MemoryToolCall,
  PendingReflectionMemory,
  ReflectionEvidenceEventResponse,
  ReflectionSourceDetail,
  ReflectionMemoryRecord,
  ReflectionMemoryListing,
  ReflectionMemoryListQuery,
  ReflectionRegenerationJobStatus,
  RuntimeStatus,
  ServiceConfig,
  PolicyDataInjection,
  SourceListing,
  TelemetryEvent,
  TrajectoryDetail,
  TrajectoryDraft,
  TrajectoryInfo,
  TrainingSelectionStatus,
  ServerSessionDeletion,
  ServerSessionListing,
  WorkTrashEntry,
} from "@/lib/types";
import { translate as t } from "@/lib/i18n";

const API = "/qwen-exo";

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

export async function getTelemetry(
  limit = 100,
  requestId?: string,
  signal?: AbortSignal,
) {
  const params = new URLSearchParams({ limit: String(limit) });
  if (requestId) params.set("request_id", requestId);
  const response = await apiFetch(`/telemetry?${params}`, { signal });
  return (await response.json()) as {
    events?: TelemetryEvent[];
    redacted?: boolean;
  };
}

export async function getRequestTraces(
  limit = 50,
  q = "",
  signal?: AbortSignal,
) {
  const params = new URLSearchParams({ limit: String(limit) });
  if (q) params.set("q", q);
  return (await (
    await apiFetch(`/request-traces?${params}`, { signal })
  ).json()) as RequestTraceListing;
}

export async function listDocumentCategories(signal?: AbortSignal) {
  return (await (
    await apiFetch("/knowledge/categories", { signal })
  ).json()) as {
    categories: DocumentCategory[];
  };
}

export async function createDocumentCategory(
  categoryId: string,
  title: string,
  parentId: string | null,
) {
  return (await (
    await apiFetch("/knowledge/categories", {
      method: "POST",
      body: JSON.stringify({
        category_id: categoryId,
        title,
        parent_id: parentId,
      }),
    })
  ).json()) as DocumentCategory;
}

export async function clearTelemetry() {
  return (await (
    await apiFetch("/telemetry/clear", { method: "POST" })
  ).json()) as { cleared: boolean } & Record<string, unknown>;
}

export async function listSources(
  lane: "knowledge" | "policydata",
  query = "",
  signal?: AbortSignal,
) {
  const params = new URLSearchParams();
  if (query.trim()) params.set("q", query.trim());
  const suffix = params.size ? `?${params}` : "";
  return (await (
    await apiFetch(`/${lane}${suffix}`, { signal })
  ).json()) as SourceListing;
}

/** Turn persona injection on/off; applies immediately, no restart. */
export async function setPolicyDataInjection(enabled: boolean) {
  return (await (
    await apiFetch("/policydata-injection", {
      method: "PUT",
      body: JSON.stringify({ enabled }),
    })
  ).json()) as PolicyDataInjection;
}

export async function listReflectionMemories(
  { limit, offset, q }: ReflectionMemoryListQuery,
  signal?: AbortSignal,
) {
  const params = new URLSearchParams({
    limit: String(limit),
    offset: String(offset),
  });
  if (q) params.set("q", q);
  return (await (
    await apiFetch(`/reflection-memory?${params}`, { signal })
  ).json()) as ReflectionMemoryListing;
}

export async function getReflectionMemory(
  sourceDigest: string,
  signal?: AbortSignal,
) {
  return (await (
    await apiFetch(`/reflection-memory/${encodeURIComponent(sourceDigest)}`, {
      signal,
    })
  ).json()) as { reflection: ReflectionMemoryRecord };
}

export async function getReflectionSource(
  sourceDigest: string,
  signal?: AbortSignal,
) {
  return (await (
    await apiFetch(
      `/reflection-memory/${encodeURIComponent(sourceDigest)}/source`,
      { signal },
    )
  ).json()) as ReflectionSourceDetail;
}

export async function getReflectionEvidence(
  eventId: string,
  signal?: AbortSignal,
) {
  return (await (
    await apiFetch(
      `/reflection-memory/evidence/${encodeURIComponent(eventId)}`,
      { signal },
    )
  ).json()) as ReflectionEvidenceEventResponse;
}

export async function getReflectionRegenerationStatus(signal?: AbortSignal) {
  return (await (
    await apiFetch("/reflection-memory/regeneration", { signal })
  ).json()) as ReflectionRegenerationJobStatus;
}

export async function regenerateReflectionMemory(
  sourceDigest: string,
  verifierFeedback: string,
  expectedDocumentSha256: string,
) {
  return (await (
    await apiFetch(
      `/reflection-memory/${encodeURIComponent(sourceDigest)}/regenerate`,
      {
        method: "POST",
        body: JSON.stringify({
          verifier_feedback: verifierFeedback,
          expected_document_sha256: expectedDocumentSha256,
        }),
      },
    )
  ).json()) as ReflectionRegenerationJobStatus;
}

export async function listPendingReflectionMemories(signal?: AbortSignal) {
  return (await (
    await apiFetch("/reflection-memory/pending", { signal })
  ).json()) as {
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
};

export async function ingestKnowledgeFiles(
  files: {
    filename: string;
    content_base64: string;
    retrieval_category?: string;
  }[],
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
    source_kind?: string;
    retrieval_category?: string | null;
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

export async function getMemoryToolCalls(
  requestId: string,
  signal?: AbortSignal,
) {
  const params = new URLSearchParams({ request_id: requestId });
  return (await (
    await apiFetch(`/memory-tool-calls?${params}`, { signal })
  ).json()) as { calls: MemoryToolCall[] };
}

export async function listKnowledgeFiles(
  query: {
    folder?: string;
    nested?: boolean;
    q?: string;
    area?: "all" | "imported" | "work";
    limit?: number;
    offset?: number;
  } = {},
  signal?: AbortSignal,
) {
  const params = new URLSearchParams();
  if (query.folder) params.set("folder", query.folder);
  if (query.nested === false) params.set("nested", "false");
  if (query.q) params.set("q", query.q);
  if (query.area && query.area !== "all") params.set("area", query.area);
  if (query.limit != null) params.set("limit", String(query.limit));
  if (query.offset) params.set("offset", String(query.offset));
  const suffix = params.size ? `?${params}` : "";
  return (await (
    await apiFetch(`/knowledge/files${suffix}`, { signal })
  ).json()) as KnowledgeFilePage;
}

export async function uploadKnowledgeFile(
  relativePath: string,
  contentBase64: string,
) {
  return (await (
    await apiFetch("/knowledge/files", {
      method: "POST",
      body: JSON.stringify({
        relative_path: relativePath,
        content_base64: contentBase64,
      }),
    })
  ).json()) as { file: KnowledgeFile };
}

/** Raw-byte upload of one text file (no base64, no client-side size limit;
 * the server answers 507 only when its disk would run low). */
export async function uploadKnowledgeFileRaw(relativePath: string, file: Blob) {
  return (await (
    await apiFetch(`/knowledge/upload?${new URLSearchParams({ path: relativePath })}`, {
      method: "POST",
      body: file,
      headers: { "Content-Type": "application/octet-stream" },
    })
  ).json()) as { file: KnowledgeFile };
}

/** Raw-byte ZIP upload; the server stores each text member as its own file. */
export async function uploadKnowledgeArchiveRaw(file: Blob, targetDirectory: string) {
  const params = new URLSearchParams({ target_directory: targetDirectory });
  return (await (
    await apiFetch(`/knowledge/upload-archive?${params}`, {
      method: "POST",
      body: file,
      headers: { "Content-Type": "application/octet-stream" },
    })
  ).json()) as {
    files: KnowledgeFile[];
    skipped: { path: string; reason: string }[];
  };
}

export async function deleteKnowledgeFile(relativePath: string) {
  await apiFetch(`/knowledge/files/${encodeSourcePath(relativePath)}`, {
    method: "DELETE",
  });
}

export async function getKnowledgeTree(signal?: AbortSignal) {
  return (await (
    await apiFetch("/knowledge/tree", { signal })
  ).json()) as KnowledgeTree;
}

export async function createKnowledgeFolder(path: string) {
  return (await (
    await apiFetch("/knowledge/folders", {
      method: "POST",
      body: JSON.stringify({ path }),
    })
  ).json()) as { path: string };
}

/** Removes a folder and everything inside it. */
export async function deleteKnowledgeFolder(path: string) {
  return (await (
    await apiFetch(`/knowledge/folders/${encodeSourcePath(path)}`, {
      method: "DELETE",
    })
  ).json()) as { deleted: boolean; path?: string; files: number };
}

/** Renames or moves a file or folder; summaries follow it. */
export async function moveKnowledgeEntry(source: string, destination: string) {
  return (await (
    await apiFetch("/knowledge/move", {
      method: "POST",
      body: JSON.stringify({ source, destination }),
    })
  ).json()) as KnowledgeMoveResult;
}

export async function getSummaryStatus(signal?: AbortSignal) {
  return (await (
    await apiFetch("/knowledge/summaries", { signal })
  ).json()) as KnowledgeSummaryStatus;
}

/** Returns ``null`` when the file has no summary yet. */
export async function getKnowledgeSummary(path: string) {
  try {
    return (await (
      await apiFetch(`/knowledge/summary?${new URLSearchParams({ path })}`)
    ).json()) as KnowledgeSummaryRecord;
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) return null;
    throw error;
  }
}

/** Stores a hand-edited summary (origin manual; never auto-overwritten). */
export async function saveKnowledgeSummary(path: string, summary: string) {
  return (await (
    await apiFetch("/knowledge/summary", {
      method: "PUT",
      body: JSON.stringify({ path, summary }),
    })
  ).json()) as KnowledgeSummaryRecord;
}

export async function deleteKnowledgeSummary(path: string) {
  return (await (
    await apiFetch(`/knowledge/summary?${new URLSearchParams({ path })}`, {
      method: "DELETE",
    })
  ).json()) as { deleted: boolean };
}

/** Queues summaries for ``paths`` (empty = every file still missing one). */
export async function generateKnowledgeSummaries(paths: string[], force = false) {
  return (await (
    await apiFetch("/knowledge/summaries/generate", {
      method: "POST",
      body: JSON.stringify({ paths, force }),
    })
  ).json()) as KnowledgeSummaryStatus;
}

export async function searchKnowledge(options: {
  pattern: string;
  pathGlob?: string;
  caseSensitive?: boolean;
  regex?: boolean;
  maxResults?: number;
  context?: number;
}) {
  const params = new URLSearchParams({
    pattern: options.pattern,
    path_glob: options.pathGlob || "**/*",
    case_sensitive: String(Boolean(options.caseSensitive)),
    regex: String(Boolean(options.regex)),
    max_results: String(options.maxResults ?? 200),
    context_before: String(options.context ?? 1),
    context_after: String(options.context ?? 1),
  });
  return (await (
    await apiFetch(`/knowledge/search?${params}`)
  ).json()) as KnowledgeSearchResponse;
}

export async function getDreamStatus(signal?: AbortSignal) {
  return (await (
    await apiFetch("/memory/dream", { signal })
  ).json()) as DreamStatus;
}

/** Start a manual dream run; 409 when one is running or dream is off. */
export async function runDream() {
  return (await (
    await apiFetch("/memory/dream/run", { method: "POST" })
  ).json()) as DreamStatus;
}

export async function getDreamRun(id: number, signal?: AbortSignal) {
  return (await (
    await apiFetch(`/memory/dream/runs/${encodeURIComponent(String(id))}`, { signal })
  ).json()) as DreamRunDetail;
}

export async function getWorkTrash(signal?: AbortSignal) {
  return (await (
    await apiFetch("/memory/work-trash", { signal })
  ).json()) as { entries: WorkTrashEntry[] };
}

/** Restore one deleted work note; 400 when the original path exists again. */
export async function restoreWorkTrash(batch: string, path: string) {
  return (await (
    await apiFetch("/memory/work-trash/restore", {
      method: "POST",
      body: JSON.stringify({ batch, path }),
    })
  ).json()) as { restored: string };
}

export async function readKnowledge(path: string, offset = 0, limit = 262_144) {
  const params = new URLSearchParams({
    path,
    offset: String(offset),
    limit: String(limit),
  });
  return (await (
    await apiFetch(`/knowledge/read?${params}`)
  ).json()) as KnowledgeReadResponse;
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

export async function deleteApiKeys(ids: string[]) {
  return (await (
    await apiFetch("/api-keys/delete", {
      method: "POST",
      body: JSON.stringify({ ids }),
    })
  ).json()) as ApiKeyDeletion;
}

export async function getModelId() {
  const response = await fetch(`${API}/console/v1/models`, {
    headers: { Accept: "application/json" },
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
  const response = await fetch(`${API}/console/v1/responses`, {
    method: "POST",
    headers: {
      Accept: "text/event-stream",
      "Content-Type": "application/json",
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

export async function listServerSessions(
  limit = 25,
  offset = 0,
  q = "",
  signal?: AbortSignal,
) {
  const params = new URLSearchParams({
    limit: String(limit),
    offset: String(offset),
  });
  if (q) params.set("q", q);
  return (await (
    await apiFetch(`/server-sessions?${params}`, { signal })
  ).json()) as ServerSessionListing;
}

export async function deleteServerSessions(
  selection: { conversation_keys: string[] } | { all: true },
) {
  return (await (
    await apiFetch("/server-sessions/delete", {
      method: "POST",
      body: JSON.stringify(selection),
    })
  ).json()) as ServerSessionDeletion;
}
