export type RuntimeStatus = {
  project?: string;
  enabled: boolean;
  runtime_state: string;
  model_path?: string;
  tp_size?: number;
  max_running_requests?: number;
  observer_mode?: string;
  features?: Record<string, boolean>;
  hybrid_state?: {
    tp_size?: number;
    dtype?: string;
    mamba_state_dtype?: string;
    mamba_strategy?: string;
    page_size?: number;
    atomic_full_gdn_lifecycle?: boolean;
  };
  model?: {
    model_path?: string;
    revision?: string;
    fingerprint?: string;
    [key: string]: unknown;
  } | null;
  knowledge?: { source_digest?: string; document_count?: number };
  policy_data?: {
    source_digest?: string;
    document_count?: number;
    enabled?: boolean;
    route?: string;
    max_tokens?: number;
  };
  cognition?: {
    source_digest?: string;
    document_count?: number;
    always_on?: boolean;
    route?: string;
  };
  internal_services?: Record<string, boolean>;
  telemetry?: {
    event_count?: number;
    observer_mode?: string;
    persistence?: { status?: string; [key: string]: unknown };
  };
  [key: string]: unknown;
};

export type CatalogModel = {
  model_fingerprint: string;
  name: string;
  model_path: string;
  architecture: string;
  model_type: string;
  variant: string;
  layer_count: number;
  full_attention_layers: number;
  linear_attention_layers: number;
  max_position_embeddings: number;
  weight_bytes: number;
  checkpoint_quantization: string | null;
  checkpoint_quantization_bits: number | null;
  checkpoint_quantization_group_size: number | null;
  checkpoint_quantization_exclusions: string[];
  runtime_quantization: string | null;
  active: boolean;
  running: boolean;
  profile_initialized: boolean;
  profile_root: string;
  knowledge_document_count: number;
  policy_document_count: number;
  cognition_document_count: number;
  native_bank_ready: boolean;
};

export type ModelCatalog = {
  schema: number;
  revision: string;
  active_model_fingerprint: string;
  applied_model_fingerprint: string | null;
  healthy_model_fingerprint: string | null;
  previous_model_fingerprint: string | null;
  last_failed_model_fingerprint: string | null;
  last_rollback_at: string | null;
  running_model_fingerprint: string | null;
  managed_restart: boolean;
  models: CatalogModel[];
  catalog_roots: string[];
  profiles_root: string;
  source_root: string;
  sources_shared: boolean;
};

export type HealthStatus = {
  project: string;
  status: string;
  runtime_state: string;
  telemetry_persistence?: Record<string, unknown>;
};

export type TelemetryEvent = {
  event_id?: number;
  timestamp?: string | number;
  request_id?: string;
  category?: string;
  event?: string;
  type?: string;
  payload?: Record<string, unknown>;
  [key: string]: unknown;
};

export type SourceDocument = {
  relative_path: string;
  byte_count?: number;
  token_count?: number;
  source_digest?: string;
  title?: string;
  tags?: string[];
  source_kind?: string;
  document_group?: string | null;
  retrieval_diversity_bucket?: string;
  canonical?: boolean;
  quality?: number;
  ingested_at?: number | null;
  updated_at?: number | null;
  [key: string]: unknown;
};

export type SourceListing = {
  source_digest?: string;
  document_count?: number;
  documents: SourceDocument[];
  /** Persona lane only: injected into every request's instructions (hot switch). */
  enabled?: boolean;
  always_on?: boolean;
  /** Persona lane only: full token size of the injected document. */
  attached_tokens?: number | null;
};

export type PolicyDataInjection = {
  enabled: boolean;
  always_on: boolean;
  route: string;
  attached_tokens: number | null;
};

export type DocumentCategory = {
  category_id: string;
  title: string;
  parent_id: string | null;
  origin: "system" | "user" | "observed";
  document_count: number;
};

export type PendingReflectionMemory = {
  conversation_key: string;
  trajectory_id: string;
  original_task: string;
  status: "waiting" | "running" | "failed";
  event_count: number;
  trajectory_row_count: number;
  capsule_count: number;
  source_token_count: number;
  source_digest: string;
  last_activity_at: number;
  scheduled_at: number;
  due_at: number;
  timeout_remaining_seconds: number;
  started_at?: number | null;
  error?: string | null;
  coverage?: ReflectionCoverage | null;
};

export type ReflectionTrajectorySource = {
  source_digest: string;
  trajectory_id: string;
  conversation_key: string;
  captured_at: number;
  supersedes_source_digest?: string | null;
  source_event_count: number;
  source_token_count: number;
  trajectory_row_count: number;
  capsule_count: number;
  verifier_feedback_present: boolean;
};

export type ReflectionEvidenceRef = {
  event_id: string;
  quote: string;
};

export type ReflectionCausalEntry = {
  entry_id: string;
  version: number;
  title: string;
  scope: string;
  problem: string;
  action: string;
  observation: string;
  mechanism: string;
  alternatives: string[];
  counterevidence: string[];
  missing_evidence: string[];
  causal_status: "verified" | "supported" | "unresolved";
  admission_status: "active" | "candidate" | "retired";
  rule: string;
  next_check: string;
  evidence_refs: ReflectionEvidenceRef[];
  verification?: { method?: string; evidence_refs?: ReflectionEvidenceRef[] };
  reason?: string;
  versions?: Array<Omit<ReflectionCausalEntry, "versions">>;
};

export type ReflectionCoverage = {
  status?: "complete" | "partial" | "failed" | "no_lesson" | string;
  provided_events?: number;
  analyzed_events?: number;
  pending_events?: number;
  failed_segments?: number;
  segments?: Array<{
    segment_id: string;
    event_ids?: string[];
    status: string;
    reason?: string;
  }>;
  [key: string]: unknown;
};

export type ReflectionMemorySummary = {
  causal_schema?: number;
  trajectory_id: string;
  conversation_key: string;
  source_digest: string;
  source_snapshot_digest?: string | null;
  title: string;
  outcome: "success" | "failure" | "mixed" | "uncertain";
  source_event_count: number;
  source_token_count: number;
  created_at: number;
  document_path?: string | null;
  document_sha256?: string | null;
  native_source_digest?: string | null;
  publication_status: string;
  hot_updated: boolean;
  source_available: boolean;
  trajectory_source?: ReflectionTrajectorySource | null;
  analysis_status?: string;
};

export type ReflectionMemoryRecord = ReflectionMemorySummary & {
  causal_entries?: ReflectionCausalEntry[];
  coverage?: ReflectionCoverage | null;
  entry_changes?: Array<Record<string, unknown>>;
  reflection: string;
  evidence: string;
  causal_analysis: string;
  conflict_resolution: string;
  reusable_experience: string;
  avoid: string;
  next_time: string;
};

export type ReflectionMemoryListing = {
  reflections: ReflectionMemorySummary[];
  total: number;
  limit: number;
  offset: number;
};

export type ReflectionMemoryListQuery = {
  limit: number;
  offset: number;
  q: string;
};
export type ReflectionEvidenceEvent = {
  event_id: string;
  kind?: string;
  request_id?: string;
  tool_name?: string;
  call_id?: string;
  content?: string;
  [key: string]: unknown;
};

export type ReflectionSourceDetail = {
  reflection: ReflectionMemoryRecord;
  source: ReflectionTrajectorySource & {
    original_task: string;
    trajectory_history: Array<Record<string, unknown>>;
    capsule_history: Array<Record<string, unknown>>;
    verifier_feedback: string;
    source_audit: Record<string, unknown>;
    coverage?: ReflectionCoverage | null;
    causal_entries?: ReflectionCausalEntry[];
    entry_changes?: Array<Record<string, unknown>>;
  };
};

export type ReflectionEvidenceEventResponse = ReflectionEvidenceEvent;

export type ReflectionRegenerationJobStatus = {
  job_id: string | null;
  status: "idle" | "queued" | "running" | "succeeded" | "failed";
  stage:
    | "idle"
    | "queued"
    | "loading_source"
    | "similar_retrieval"
    | "evidence_extraction"
    | "causal_review"
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
  result?: {
    source_digest: string;
    supersedes_source_digest: string;
    trajectory_id: string;
    document_path?: string | null;
    document_sha256?: string | null;
    native_source_digest?: string | null;
    publication_status: string;
    hot_updated: boolean;
  } | null;
  error?: string | null;
};
export type ServiceSetting = {
  key: string;
  group: string;
  label: string;
  description: string;
  type: "boolean" | "integer" | "number" | "string";
  default: boolean | number | string;
  minimum?: number;
  maximum?: number;
  step?: number;
  choices?: string[];
  choice_labels?: Record<string, string>;
  unit?: string;
  restart_required: boolean;
};

export type ServiceConfig = {
  schema: number;
  revision: string;
  applied_revision: string | null;
  pending_restart: boolean;
  updated_at: string | null;
  applied_at: string | null;
  healthy_revision: string | null;
  healthy_at: string | null;
  boot_attempts: number;
  last_failed_revision: string | null;
  last_rollback_at: string | null;
  managed_restart: boolean;
  groups: { id: string; label: string; description: string }[];
  settings: ServiceSetting[];
  values: Record<string, boolean | number | string>;
  restart_requested?: boolean;
};

export type ApiKeyInfo = {
  id: string;
  label: string;
  created_at: string;
  revoked_at: string | null;
};

export type ApiKeyDeletion = {
  deleted: ApiKeyInfo[];
  missing: string[];
  revision: string;
};

export type ApiKeyListing = {
  schema: number;
  revision: string;
  updated_at: string | null;
  keys: ApiKeyInfo[];
};

export type CreatedApiKey = ApiKeyInfo & {
  token: string;
  revision: string;
};

export type ToolCall = {
  id: string;
  name: string;
  arguments: string;
  done: boolean;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  reasoning: string;
  tools: ToolCall[];
  createdAt: string;
  status: "in_progress" | "completed" | "incomplete" | "cancelled" | "failed";
  error?: boolean;
};

export type ChatSession = {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  lastResponseId: string | null;
  messages: ChatMessage[];
};

export type EditorSource = {
  name: string;
  sha256: string;
};

export type EditorInfo = {
  name: string;
  layer: number;
  rank: number;
  window: number;
  hidden_size: number;
  bytes: number;
  modified_ns: number;
  valid: boolean;
  tags?: string[];
  sources: EditorSource[];
};

export type EditorTrainingSource = {
  name: string;
  sha256: string;
  message_count: number;
  sample_count: number;
};

export type EditorQualityMetrics = {
  baseline_nll: number;
  random_editor_nll: number;
  trained_editor_nll: number;
};

export type EditorTrainingJob = {
  schema: number;
  job_id: string;
  status: "queued" | "running" | "succeeded" | "failed";
  stage: string;
  editor: string;
  trajectories: string[];
  sources: EditorTrainingSource[];
  message_count: number;
  sample_count: number;
  config: {
    layer: number;
    rank: number;
    window: number;
    epochs: number;
  };
  requested_at: string;
  started_at: string | null;
  completed_at: string | null;
  metrics:
    | (EditorQualityMetrics & {
        sources: Array<EditorQualityMetrics & { name: string }>;
      })
    | null;
  error: string | null;
};

export type EditorTrainingStatus = {
  status: "idle" | EditorTrainingJob["status"];
  job: EditorTrainingJob | null;
  restart_requested?: boolean;
};

export type ActiveEditor = {
  editor: string;
  strength: number | null;
  applied_at: string | null;
};

export type TrainingSelectionStatus = {
  names: string[];
  updated_at: string | null;
  sources: EditorTrainingSource[];
  editor: EditorInfo | null;
  up_to_date: boolean;
  applied: boolean;
};

export type TrajectoryInfo = {
  name: string;
  messages: number;
  bytes: number;
  modified_ns: number;
  valid: boolean;
  tags: string[];
};

export type TrajectoryDetail = {
  name: string;
  content: string;
  messages: number;
  bytes: number;
  tags: string[];
};

export type TrajectoryDraft = {
  suggested_name: string;
  content: string;
  tags: string[];
  messages: number;
  bytes: number;
};

export type KnowledgeDraft = {
  original_filename: string;
  suggested_path: string;
  content: string;
  tags: string[];
  retrieval_category: string;
  source_kind: string;
  byte_count: number;
  changes: string[];
};

/** One file in the server-side knowledge base (searched by memory tools). */
export type KnowledgeFile = {
  relative_path: string;
  sha256: string;
  byte_count: number;
  mtime_ns: number;
  encoding: string;
  line_count: number;
  indexed_ns: number;
  schema_version?: number;
  kind: KnowledgeFileKind;
  /** Auto-detected title (or the summary title once one exists). */
  title: string;
  summary: KnowledgeFileSummaryInfo | null;
};

export type KnowledgeFileKind =
  | "summary"
  | "note"
  | "code"
  | "other"
  | "data"
  | "log"
  | "session"
  | "trajectory"
  | "work";

/** Summary metadata attached to each knowledge file row. */
export type KnowledgeFileSummaryInfo = {
  title: string;
  origin: "auto" | "manual";
  sampled: boolean;
  updated_ns: number;
  /** The file changed after the summary was written. */
  stale: boolean;
};

export type KnowledgeDirectoryCount = {
  files: number;
  bytes: number;
};

export type KnowledgeTree = {
  /** Every folder (including empty ones), relative POSIX paths, sorted. */
  directories: string[];
  /** Recursive file totals keyed by folder. ``""`` is the whole corpus. */
  counts: Record<string, KnowledgeDirectoryCount>;
  areas: {
    imported: KnowledgeDirectoryCount;
    work: KnowledgeDirectoryCount;
  };
};

/** One server page of knowledge files. File bytes and summary text are absent. */
export type KnowledgeFilePage = {
  files: KnowledgeFile[];
  total: number;
  total_bytes: number;
  offset: number;
  limit: number;
};

export type KnowledgeMoveResult = {
  from: string;
  to: string;
  files: number;
};

/** Background summary worker state; only ``available`` exists when it is off. */
export type KnowledgeSummaryStatus = {
  available: boolean;
  queued?: string[];
  current?: {
    path: string;
    chunks: number;
    done: number;
    sampled: boolean;
  } | null;
  missing?: number;
  completed?: number;
  failed?: { path: string; error: string }[];
};

export type KnowledgeSummaryRecord = {
  relative_path: string;
  sha256: string;
  title: string;
  summary: string;
  sampled: boolean;
  origin: "auto" | "manual";
  updated_ns: number;
};

export type KnowledgeSearchResult = {
  path: string;
  line: number;
  line_start_offset: number;
  /** Matched substring; ``line_text`` is the whole (clipped) line. */
  text: string;
  line_text: string;
  before: string[];
  after: string[];
};

export type KnowledgeSearchResponse = {
  results: KnowledgeSearchResult[];
  count: number;
  truncated: boolean;
};

export type KnowledgeReadResponse = {
  path: string;
  offset: number;
  end_offset: number;
  total_bytes: number;
  text: string;
  eof: boolean;
};

export type RequestTrace = {
  request_id: string;
  started_at: number | null;
  duration_seconds: number | null;
  input_text: string;
  output_text: string;
  reasoning_text?: string;
  output_tokens: number | null;
  /** Server-executed memory tool calls (qwen_exo_memory_search/read). */
  memory_tool_calls: number;
  memory_tool_failures: number;
  memory_tools: string[];
  event_types: string[];
};

/** One server-executed memory tool call, from the tool-call journal. */
export type MemoryToolCall = {
  request_id: string;
  round: number;
  call_id: string;
  name: string;
  arguments: string;
  ok: boolean;
  output?: string;
  output_chars: number;
  duration_ms: number;
  created_at: number;
};

export type RequestTraceListing = {
  requests: RequestTrace[];
  total_requests: number;
  text_mode?: string;
};
export type ServerSession = {
  conversation_key: string;
  updated_at: number;
  event_count: number;
  raw_bytes: number;
  source_count: number;
  active: boolean;
  pending_reflection: boolean;
};

export type ServerSessionListing = {
  sessions: ServerSession[];
  total: number;
  limit: number;
  offset: number;
};

export type ServerSessionDeletion = {
  deleted: string[];
  skipped_active: string[];
  missing: string[];
  deleted_count: number;
};

export type DreamTrigger = "nightly" | "manual";

export type DreamRunStatus = "running" | "completed" | "stopped" | "failed" | "cancelled";

export type DreamActionKind = "write" | "move" | "delete" | "mkdir";

/** One tool action the dream agent applied to the work memory. */
export type DreamAction = {
  step: number;
  action: DreamActionKind | string;
  target: string;
  to?: string | null;
  ok: boolean;
  message?: string | null;
};

export type DreamRunSummary = {
  id: number;
  trigger: DreamTrigger | string;
  started_ns: number;
  finished_ns: number | null;
  status: DreamRunStatus | string;
  steps: number;
  summary: string;
  action_count: number;
};

export type DreamRunDetail = Omit<DreamRunSummary, "action_count"> & {
  action_count?: number;
  actions: DreamAction[];
};

/** Nightly work-memory tidy-up agent; only ``available`` exists when it is off. */
export type DreamStatus = {
  available: boolean;
  running?: boolean;
  current?: {
    run_id: number;
    step: number;
    last_action: DreamAction | null;
  } | null;
  next_run_in_seconds?: number;
  runs?: DreamRunSummary[];
};

export type WorkTrashEntry = {
  /** Deletion batch, e.g. ``dream-12``. */
  batch: string;
  /** Path relative to the knowledge root, e.g. ``work-memory/ctf/x.md``. */
  path: string;
  bytes: number;
};
