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
  tensor_bank?: Record<string, unknown> | null;
  internal_services?: Record<string, boolean>;
  telemetry?: {
    event_count?: number;
    observer_mode?: string;
    persistence?: { status?: string; [key: string]: unknown };
  };
  adaptive_retrieval?: Record<string, unknown> | null;
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

export type RecallTurn = {
  turn_id?: string;
  request_id?: string;
  outcome?: string;
  strategy?: string;
  source?: string;
  knowledge_admitted?: boolean;
  policy_data_active?: boolean;
  self_ask_seconds?: number;
  exact_replay_seconds?: number;
  total_seconds?: number;
  [key: string]: unknown;
};

export type RecallTrace = {
  schema?: string;
  bank?: Record<string, unknown>;
  turns: RecallTurn[];
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
  canonical?: boolean;
  quality?: number;
  compiled?: boolean;
  compile_status?: "compiled" | "uncompiled";
  compiled_at?: number | null;
  ingested_at?: number | null;
  updated_at?: number | null;
  [key: string]: unknown;
};

export type SourceListing = {
  source_digest?: string;
  document_count?: number;
  documents: SourceDocument[];
};

export type PendingReflectionMemory = {
  conversation_key: string;
  trajectory_id: string;
  original_task: string;
  status: "waiting" | "running";
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
  source_kind: string;
  byte_count: number;
  changes: string[];
};

export type TraceCandidate = {
  relative_path: string;
  lane: string;
  tensor_score: number | null;
  lexical_score: number | null;
  excerpt: string;
};

export type RequestTrace = {
  request_id: string;
  started_at: number | null;
  duration_seconds: number | null;
  input_text: string;
  output_text: string;
  output_tokens: number | null;
  candidates: TraceCandidate[];
  native_restore: {
    lane: string;
    tokens: number;
    selection_reason: string;
  } | null;
  cognition_active: boolean;
  attached_tokens: number | null;
  self_ask: {
    event_type: string;
    question: string;
    answer: string;
    status: string;
  }[];
  latent_transplant: { artifact: string; strength: number | null } | null;
  event_types: string[];
};

export type RequestTraceListing = {
  requests: RequestTrace[];
  total_requests: number;
  text_mode?: string;
};
