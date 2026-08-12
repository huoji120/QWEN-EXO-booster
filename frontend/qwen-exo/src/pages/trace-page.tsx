import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BrainCircuit,
  CheckCircle2,
  ChevronRight,
  CircleDot,
  Copy,
  Database,
  Eraser,
  FileSearch,
  FileText,
  Hash,
  LoaderCircle,
  RefreshCw,
  Search,
  Sparkles,
  XCircle,
} from "lucide-react";
import { toast } from "sonner";
import { EmptyState } from "@/components/empty-state";
import { PageHeader } from "@/components/page-header";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  ApiError,
  clearTelemetry,
  getRequestTraces,
  getServiceConfig,
  getSource,
  getTelemetry,
} from "@/lib/api";
import { currentLanguage, translate as t } from "@/lib/i18n";
import type {
  RequestTrace,
  RequestTraceListing,
  TelemetryEvent,
} from "@/lib/types";
import {
  isTraceDefaultScope,
  type TraceDefaultScope,
} from "@/lib/trace-preferences";
import { cn, formatDuration, formatNumber, shortHash } from "@/lib/utils";

type JsonObject = Record<string, unknown>;
type RequestStatus = "running" | "completed" | "cancelled" | "failed";
type TraceFilter = TraceDefaultScope | "proposed" | "running" | "integrity";

type CandidateEvidence = {
  candidateId: string;
  documentId: string;
  path: string;
  lane: string;
  score: number | null;
  tensorScore: number | null;
  lexicalScore: number | null;
  selected: boolean;
  decisionStatus: string;
  judgeMethod: string;
  origin: string;
  pageIds: number[];
  sourcePositionCount: number;
  sourcePositionMin: number | null;
  sourcePositionMax: number | null;
  topAttributionScore: number | null;
  excerpt: string;
};

type QkScoredDocument = {
  lane: string;
  documentId: string;
  path: string;
  score: number | null;
  passedScore: boolean;
  rejectionReason: string;
};

type QkRecallAudit = {
  status: string;
  reason: string;
  preset: string;
  minTensorScore: number | null;
  minDocumentMargin: number | null;
  bestScore: number | null;
  secondScore: number | null;
  observedMargin: number | null;
  consideredDocuments: number | null;
  candidateCount: number | null;
  scoredDocuments: QkScoredDocument[];
};

type ReplayLossEvidence = {
  candidateId: string | null;
  documentId: string | null;
  nll: number | null;
  gainVsBaseline: number | null;
  observedTokenKl: number | null;
  observationTokens: number | null;
};

type CausalReplayEvidence = {
  eventType: string;
  eventId: number | null;
  timestamp: string | number | null;
  decision: string;
  maybeDecision: string;
  winnerCandidateId: string | null;
  winnerDocumentId: string | null;
  winnerGain: number | null;
  winnerKl: number | null;
  scheduledNextTurn: boolean | null;
  latencySeconds: number | null;
  errorType: string | null;
  losses: ReplayLossEvidence[];
};

type MaybeEvidence = {
  eventId: number | null;
  timestamp: string | number | null;
  status: string;
  decision: string;
  scheduledNextTurn: boolean | null;
};

type TraceEvidence = {
  requestStatus: RequestStatus;
  promptTokens: number | null;
  queryTokens: number | null;
  cognitionProbeTokens: number | null;
  probeLatencySeconds: number | null;
  attachedTokens: number;
  nativeRestore: {
    active: boolean;
    lane: string;
    tokens: number;
    reason: string;
  };
  selectedKnowledgeIds: string[];
  selectedPolicyIds: string[];
  retrievalSeconds: number | null;
  judgeSeconds: number | null;
  judgeQuestionTruncated: boolean;
  judgeQuestionOriginalTokens: number | null;
  judgeQuestionReviewTokens: number | null;
  candidates: CandidateEvidence[];
  rejectedCount: number;
  fullyJudged: boolean;
  qkRecall: QkRecallAudit;
  selfAsk: RequestTrace["self_ask"];
  causalReplay: CausalReplayEvidence[];
  maybe: MaybeEvidence[];
  events: TelemetryEvent[];
};

const EVENT_TYPE_LABELS: Record<string, string> = {
  "request.started": "请求开始",
  "request.completed": "请求完成",
  "request.cancelled": "请求取消",
  "request.failed": "请求失败",
  "query_probe.started": "检索探针开始",
  "query_probe.completed": "检索探针完成",
  "tensor.candidates_proposed": "Tensor Bank 提出候选",
  "semantic_judge.completed": "语义 Judge 完成",
  "memory.prepared": "记忆准备完成",
  "refresh.started": "Self-Ask 刷新开始",
  "refresh.completed": "Self-Ask 刷新完成",
  "self_ask.routed": "Self-Ask 路由",
  "self_ask.next_turn_context_restored": "Self-Ask 上下文恢复",
  "post_tool_recall.completed": "工具后召回完成",
  "context_integrity.started": "上下文完整性审查开始",
  "context_integrity.completed": "上下文完整性审查完成",
  "context_integrity.applied": "上下文更正已准入",
  "reflection_memory.scheduled": "反思记忆已排队",
  "reflection_memory.started": "反思记忆开始",
  "reflection_memory.qk_retrieval.started": "反思记忆 Q×K 检索开始",
  "reflection_memory.qk_retrieval.completed": "反思记忆 Q×K 候选完成",
  "reflection_memory.qk_retrieval_failed_closed": "反思记忆 Q×K 失败关闭",
  "reflection_memory.consolidation_decided": "反思记忆合并决策完成",
  "reflection_memory.attempt_failed": "反思记忆工具调用重试",
  "reflection_memory.completed": "反思记忆完成",
  "reflection_memory.published": "反思记忆已热写入知识库",
  "reflection_memory.failed_closed": "反思记忆失败关闭",
  "adaptive.transition": "自适应状态迁移",
  "observer.decode_summary": "Observer 解码观测",
  "score_bias.prompt_scored": "Score Bias 提示评分",
  "score_bias.applied": "Score Bias 已应用",
  "score_bias.decode_abstained": "Score Bias 解码弃权",
  "tensor_bank.prefix_cache": "原生前缀缓存",
  "capsule.updated": "执行胶囊更新",
  "request.stage_summary": "请求阶段汇总",
  "causal_replay.completed": "因果回放完成",
  "causal_replay.failed_closed": "因果回放失败关闭",
  "maybe.completed": "Maybe 门禁完成",
};

const FILTERS: { value: TraceFilter; label: string }[] = [
  { value: "activity", label: "召回 / 审查" },
  { value: "integrity", label: "上下文完整性" },
  { value: "actual", label: "仅已召回" },
  { value: "proposed", label: "有候选" },
  { value: "all", label: "全部" },
  { value: "running", label: "进行中" },
];

const UNKNOWN_DOCUMENT_SOURCE = "未知文档";

const QUERY_SOURCE_LABELS: Record<string, string> = {
  attention_q_request_start: "请求开始 Attention-Q",
  attention_q_local_window: "局部 Attention-Q 窗口",
  post_tool_factual_question: "工具后事实问题",
  self_ask_skipped: "Self-Ask 已跳过",
};

const QUERY_PROBE_STATUS_LABELS: Record<string, string> = {
  ready: "就绪",
  empty_query: "查询为空",
  failed_closed: "失败关闭",
  no_q_signal: "无 Q 信号",
  unavailable: "不可用",
  not_requested: "未请求",
};

const SELF_ASK_STATUS_LABELS: Record<string, string> = {
  ready_for_safe_replay: "可安全回放",
  policy_reflection_ready: "策略反思就绪",
  context_evidence_ready: "上下文证据就绪",
  context_integrity_ready: "完整性更正就绪",
  no_eligible_reference: "无可用参考",
  no_answering_evidence: "无作答证据",
  self_ask_skipped: "已跳过",
  replay_rejected: "回放已拒绝",
  semantic_ready: "语义就绪",
  observer_shadow: "影子观测",
  replay_admitted: "回放已准入",
  reject_fail_closed: "失败关闭拒绝",
  admit_maybe: "Maybe 已准入",
  pending: "等待判定",
  not_requested: "未请求",
};

const CAUSAL_REPLAY_DECISION_LABELS: Record<string, string> = {
  shadow_would_switch: "候选分支胜出",
  reject_no_challenger: "没有可比较候选",
  reject_no_semantic_candidate: "没有通过语义准入的候选",
  reject_insufficient_gain: "损失增益不足",
  reject_switch_margin: "切换边际不足",
  insufficient_future_observation: "未来观测 token 不足",
  reject_empty_prefix: "回放前缀为空",
  reject_empty_candidate_state: "候选状态为空",
  failed_closed: "失败关闭",
};

const MAYBE_DECISION_LABELS: Record<string, string> = {
  admit_maybe: "Maybe 已准入",
  reject_maybe_kl: "KL 超限，拒绝 Maybe",
  not_compiled: "未编译 Maybe",
  not_recorded: "未记录",
};

const CONTEXT_INTEGRITY_REASON_LABELS: Record<string, string> = {
  conversation_refresh_budget_exhausted: "对话刷新预算已用尽",
  no_eligible_reference_cooldown: "无可用参考，仍在冷却",
  mode_off: "完整性检查已关闭",
  unexpected_correction_without_corrected_status: "更正状态异常",
  correction_without_prior_conflict: "更正前未确认冲突",
  correction_requires_more_evidence: "更正需要更多证据",
  correction_missing_exact_evidence_quote: "更正缺少精确证据引文",
  correction_evidence_quote_not_in_current_tool_result:
    "更正引文不在当前工具结果中",
  correction_text_is_invalid: "更正文无效",
  correction_confidence_below_admission_threshold: "更正置信度低于准入门槛",
};

const TIME_FORMATTERS = {
  "zh-CN": new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }),
  "en-US": new Intl.DateTimeFormat("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }),
};

const DATE_TIME_FORMATTERS = {
  "zh-CN": new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }),
  "en-US": new Intl.DateTimeFormat("en-US", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }),
};

const FIXED_DECIMAL_FORMATTERS = {
  "zh-CN": {
    two: new Intl.NumberFormat("zh-CN", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }),
    three: new Intl.NumberFormat("zh-CN", {
      minimumFractionDigits: 3,
      maximumFractionDigits: 3,
    }),
  },
  "en-US": {
    two: new Intl.NumberFormat("en-US", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }),
    three: new Intl.NumberFormat("en-US", {
      minimumFractionDigits: 3,
      maximumFractionDigits: 3,
    }),
  },
};

function formatFixedNumber(value: number, digits: 2 | 3) {
  const precision = digits === 2 ? "two" : "three";
  return FIXED_DECIMAL_FORMATTERS[currentLanguage()][precision].format(value);
}

function translatedCodeLabel(
  value: unknown,
  labels: Record<string, string>,
  fallback = "未知",
) {
  const code = String(value || "").trim();
  return t(labels[code] || code || fallback);
}

function selfAskStatusLabel(status: string) {
  return status.startsWith("failed_closed")
    ? t("失败关闭")
    : translatedCodeLabel(status, SELF_ASK_STATUS_LABELS, "状态未记录");
}

function asObject(value: unknown): JsonObject {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as JsonObject)
    : {};
}

function asObjects(value: unknown): JsonObject[] {
  return Array.isArray(value) ? value.map(asObject) : [];
}

function strings(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];
}

function numbers(value: unknown): number[] {
  return Array.isArray(value)
    ? value.filter((item): item is number => typeof item === "number")
    : [];
}

function numberOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function booleanOrNull(value: unknown): boolean | null {
  return typeof value === "boolean" ? value : null;
}

function eventType(event: TelemetryEvent): string {
  return String(event.event_type || event.event || event.type || "unknown");
}

function eventPayload(event: TelemetryEvent | undefined): JsonObject {
  return asObject(event?.payload);
}

function timeOnly(value?: string | number | null) {
  if (value === null || value === undefined) return "—";
  const date =
    typeof value === "number"
      ? new Date(value > 1e12 ? value : value * 1000)
      : new Date(value);
  return Number.isNaN(date.getTime())
    ? "—"
    : TIME_FORMATTERS[currentLanguage()].format(date);
}

function dateTime(value?: string | number | null) {
  if (value === null || value === undefined) return t("时间未知");
  const date =
    typeof value === "number"
      ? new Date(value > 1e12 ? value : value * 1000)
      : new Date(value);
  return Number.isNaN(date.getTime())
    ? t("时间未知")
    : DATE_TIME_FORMATTERS[currentLanguage()].format(date);
}

function hasActualKnowledge(trace: RequestTrace) {
  return (
    Number(trace.attached_tokens || 0) > 0 ||
    trace.native_restore?.lane === "knowledge"
  );
}

function hasPolicyDataRestore(trace: RequestTrace) {
  return trace.native_restore?.lane === "policydata";
}

function contextIntegrityEventCount(trace: RequestTrace) {
  return trace.event_types.filter((type) =>
    type.startsWith("context_integrity."),
  ).length;
}

function causalReplayEventCount(trace: RequestTrace) {
  return trace.event_types.filter(
    (type) => type.startsWith("causal_replay.") || type === "maybe.completed",
  ).length;
}

function contextIntegrityListStatus(trace: RequestTrace) {
  if (trace.event_types.includes("context_integrity.applied")) {
    return { label: t("完整性：已干预"), variant: "default" as const };
  }
  if (
    trace.event_types.includes("context_integrity.completed") ||
    trace.event_types.includes("context_integrity.skipped") ||
    trace.event_types.includes("context_integrity.failed_closed")
  ) {
    return { label: t("完整性：未干预"), variant: "secondary" as const };
  }
  if (trace.event_types.includes("context_integrity.started")) {
    return { label: t("完整性：审查中"), variant: "outline" as const };
  }
  return null;
}

function lastContextIntegrityEvent(events: TelemetryEvent[], type: string) {
  return [...events].reverse().find((event) => eventType(event) === type);
}

function contextIntegrityVerdict(events: TelemetryEvent[]) {
  const integrityEvents = events.filter((event) =>
    eventType(event).startsWith("context_integrity."),
  );
  if (!integrityEvents.length) return null;
  const applied = lastContextIntegrityEvent(
    integrityEvents,
    "context_integrity.applied",
  );
  const completed = lastContextIntegrityEvent(
    integrityEvents,
    "context_integrity.completed",
  );
  const skipped = lastContextIntegrityEvent(
    integrityEvents,
    "context_integrity.skipped",
  );
  const failed = lastContextIntegrityEvent(
    integrityEvents,
    "context_integrity.failed_closed",
  );
  const completedPayload = eventPayload(completed);
  const appliedPayload = eventPayload(applied);
  const status = String(
    appliedPayload.status || completedPayload.status || "unknown",
  );
  const confidence = numberOrNull(
    appliedPayload.confidence ?? completedPayload.confidence,
  );
  const invalidClaimCount = Array.isArray(completedPayload.invalid_claims)
    ? completedPayload.invalid_claims.length
    : 0;

  if (applied) {
    return {
      label: t("已干预"),
      title: t("已写入上下文更正"),
      description: t(
        "最新工具结果证明历史上下文存在实质冲突；服务端已将带原文证据的更正写入内部 think 上下文。",
      ),
      action: t("写入内部 think 更正；未追加为用户文本"),
      status,
      confidence,
      invalidClaimCount,
      icon: CheckCircle2,
      badgeVariant: "default" as const,
      tone: "border-primary/35 bg-primary/[0.05]",
    };
  }
  if (failed) {
    return {
      label: t("未干预"),
      title: t("完整性检查失败关闭"),
      description: t("检查发生错误，服务端按失败关闭策略没有写入任何更正。"),
      action: t("未修改上下文"),
      status: "failed_closed",
      confidence,
      invalidClaimCount,
      icon: XCircle,
      badgeVariant: "destructive" as const,
      tone: "border-destructive/35 bg-destructive/[0.05]",
    };
  }
  if (status === "consistent") {
    return {
      label: t("无需干预"),
      title: t("上下文与最新工具结果一致"),
      description: t(
        "审查没有确认任何实质冲突，因此服务端没有生成或写入上下文更正。",
      ),
      action: t("无需处理；未修改上下文"),
      status,
      confidence,
      invalidClaimCount,
      icon: CheckCircle2,
      badgeVariant: "success" as const,
      tone: "border-emerald-500/35 bg-emerald-500/[0.05]",
    };
  }
  if (status === "corrected") {
    const reason = translatedCodeLabel(
      eventPayload(skipped).reason,
      CONTEXT_INTEGRITY_REASON_LABELS,
      "更正未获得准入",
    );
    return {
      label: t("未干预"),
      title: t("发现冲突，但没有写入更正"),
      description: t(
        "模型形成了有证据的更正，但服务端没有应用。原因：{reason}。",
        { reason },
      ),
      action: t("未修改上下文"),
      status,
      confidence,
      invalidClaimCount,
      icon: AlertTriangle,
      badgeVariant: "warning" as const,
      tone: "border-amber-500/35 bg-amber-500/[0.05]",
    };
  }
  if (status === "uncertain") {
    const reason = translatedCodeLabel(
      completedPayload.reason,
      CONTEXT_INTEGRITY_REASON_LABELS,
      "证据不完整或存在歧义",
    );
    return {
      label: t("未干预"),
      title: t("证据不足，没有修改上下文"),
      description: t("完整性检查无法确认需要更正。原因：{reason}。", {
        reason,
      }),
      action: t("未修改上下文"),
      status,
      confidence,
      invalidClaimCount,
      icon: AlertTriangle,
      badgeVariant: "warning" as const,
      tone: "border-amber-500/35 bg-amber-500/[0.05]",
    };
  }
  if (completed || skipped) {
    return {
      label: t("未干预"),
      title: t("审查完成，没有修改上下文"),
      description: t(
        "服务端完成了完整性检查，但当前遥测没有可识别的更正准入结论。",
      ),
      action: t("未修改上下文"),
      status,
      confidence,
      invalidClaimCount,
      icon: AlertTriangle,
      badgeVariant: "warning" as const,
      tone: "border-amber-500/35 bg-amber-500/[0.05]",
    };
  }
  return {
    label: t("审查中"),
    title: t("正在核对最新工具结果"),
    description: t("服务端尚未给出是否需要干预的最终结论。"),
    action: t("尚未修改上下文"),
    status,
    confidence,
    invalidClaimCount,
    icon: LoaderCircle,
    badgeVariant: "outline" as const,
    tone: "border-border bg-muted/20",
  };
}

function contextIntegrityStatusLabel(status: string) {
  if (status === "consistent") return t("上下文一致");
  if (status === "corrected") return t("已形成更正");
  if (status === "uncertain") return t("无法确认");
  if (status === "failed_closed") return t("失败关闭");
  return status === "unknown" ? t("检查中") : status;
}

function requestStatus(events: TelemetryEvent[]): RequestStatus {
  const types = new Set(events.map(eventType));
  if (types.has("request.failed") || types.has("system.error")) return "failed";
  if (types.has("request.cancelled")) return "cancelled";
  if (types.has("request.completed")) return "completed";
  return "running";
}

function statusMeta(status: RequestStatus) {
  if (status === "completed") {
    return {
      label: t("已完成"),
      variant: "success" as const,
      dot: "bg-emerald-500",
    };
  }
  if (status === "cancelled") {
    return {
      label: t("已取消"),
      variant: "warning" as const,
      dot: "bg-amber-500",
    };
  }
  if (status === "failed") {
    return {
      label: t("失败"),
      variant: "destructive" as const,
      dot: "bg-red-500",
    };
  }
  return {
    label: t("推理中"),
    variant: "default" as const,
    dot: "bg-blue-500",
  };
}

function laneLabel(lane: string) {
  if (lane === "knowledge") return t("知识");
  if (lane === "policydata") return t("人格");
  if (lane === "cognition") return t("认知");
  return lane || t("未知");
}

function decisionIsRejected(status: string) {
  return ["false", "rejected", "ineligible", "unsupported"].includes(
    status.toLowerCase(),
  );
}

function deduplicateSelfAsk(rows: RequestTrace["self_ask"]) {
  const seen = new Set<string>();
  return rows.filter((row) => {
    const key = `${row.question}\u0000${row.answer}\u0000${row.status}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function buildEvidence(
  trace: RequestTrace,
  events: TelemetryEvent[],
): TraceEvidence {
  const ordered = [...events].sort(
    (left, right) => Number(left.event_id || 0) - Number(right.event_id || 0),
  );
  const reversed = [...ordered].reverse();
  const memory = eventPayload(
    reversed.find((event) => eventType(event) === "memory.prepared"),
  );
  const probeStarted = eventPayload(
    ordered.find((event) => eventType(event) === "query_probe.started"),
  );
  const probeCompleted = eventPayload(
    reversed.find((event) => eventType(event) === "query_probe.completed"),
  );
  const memoryProbe = asObject(memory.query_probe);
  const policyData = asObject(memory.policy_data);
  const nativeRestore = asObject(memory.native_prefix_restore);
  const qkRetrieval = asObject(memory.qk_retrieval);
  const candidateProposal = eventPayload(
    reversed.find((event) => eventType(event) === "tensor.candidates_proposed"),
  );
  const requestJudgeEvent = reversed.find(
    (event) =>
      eventType(event) === "semantic_judge.completed" &&
      String(eventPayload(event).purpose || "") === "request_start_admission",
  );
  const requestJudge = eventPayload(requestJudgeEvent);
  const memoryRankAudit = asObject(qkRetrieval.audit);
  const proposalRankAudit = asObject(candidateProposal.rank_audit);
  const rankAudit = Object.keys(memoryRankAudit).length
    ? memoryRankAudit
    : proposalRankAudit;
  const rawSelectedKnowledgeIds = strings(memory.selected_document_ids);
  const rawSelectedPolicyIds = strings(policyData.document_ids);
  const selectedDocumentIds = new Set([
    ...rawSelectedKnowledgeIds,
    ...rawSelectedPolicyIds,
  ]);
  const decisions = new Map(
    asObjects(memory.semantic_decisions).map((decision) => [
      String(decision.candidate_id || ""),
      decision,
    ]),
  );
  const excerpts = new Map(
    trace.candidates.map((candidate) => [
      `${candidate.lane}:${candidate.relative_path}`,
      candidate.excerpt,
    ]),
  );
  const rawCandidates = asObjects(memory.proposed_candidates);
  const candidateRows: JsonObject[] = rawCandidates.length
    ? rawCandidates
    : trace.candidates.map((candidate) => ({
        relative_path: candidate.relative_path,
        lane: candidate.lane,
        tensor_score: candidate.tensor_score,
        lexical_score: candidate.lexical_score,
      }));
  const candidates: CandidateEvidence[] = candidateRows.map((candidate) => {
    const candidateId = String(candidate.candidate_id || "");
    const documentId = String(candidate.document_id || "");
    const path = String(candidate.relative_path || UNKNOWN_DOCUMENT_SOURCE);
    const lane = String(candidate.lane || "knowledge");
    const decision = decisions.get(candidateId) || {};
    const decisionStatus = String(decision.status || "");
    const positions = numbers(candidate.source_positions);
    const attributions = asObjects(candidate.token_attributions);
    const attributionScores = attributions
      .map((item) => numberOrNull(item.score))
      .filter((item): item is number => item !== null);
    return {
      candidateId,
      documentId,
      path,
      lane,
      score: numberOrNull(candidate.score),
      tensorScore: numberOrNull(candidate.tensor_score),
      lexicalScore: numberOrNull(candidate.lexical_score),
      selected:
        selectedDocumentIds.has(documentId) &&
        !decisionIsRejected(decisionStatus),
      decisionStatus,
      judgeMethod: String(decision.judge_method || ""),
      origin: String(candidate.candidate_origin || ""),
      pageIds: numbers(candidate.page_ids),
      sourcePositionCount: positions.length,
      sourcePositionMin: positions.length ? Math.min(...positions) : null,
      sourcePositionMax: positions.length ? Math.max(...positions) : null,
      topAttributionScore: attributionScores.length
        ? Math.max(...attributionScores)
        : null,
      excerpt: excerpts.get(`${lane}:${path}`) || "",
    };
  });
  const rejectedDocumentIds = new Set(
    candidates
      .filter((candidate) => decisionIsRejected(candidate.decisionStatus))
      .map((candidate) => candidate.documentId),
  );
  const nonRejectedSelectedDocumentIds = new Set(
    candidates
      .filter((candidate) => candidate.selected)
      .map((candidate) => candidate.documentId),
  );
  const selectedIsNotRejected = (documentId: string) =>
    !rejectedDocumentIds.has(documentId) ||
    nonRejectedSelectedDocumentIds.has(documentId);
  const selectedKnowledgeIds = rawSelectedKnowledgeIds.filter(
    selectedIsNotRejected,
  );
  const selectedPolicyIds = rawSelectedPolicyIds.filter(selectedIsNotRejected);
  candidates.sort((left, right) => {
    if (left.selected !== right.selected) return left.selected ? -1 : 1;
    return (
      Number(right.tensorScore || right.score || 0) -
      Number(left.tensorScore || left.score || 0)
    );
  });
  const rejectedCount = candidates.filter((candidate) =>
    decisionIsRejected(candidate.decisionStatus),
  ).length;
  const causalReplay: CausalReplayEvidence[] = ordered
    .filter((event) => eventType(event).startsWith("causal_replay."))
    .map((event) => {
      const payload = eventPayload(event);
      const winnerCandidateId = payload.winner_candidate_id;
      const winnerDocumentId = payload.winner_document_id;
      const errorType = payload.error_type;
      return {
        eventType: eventType(event),
        eventId: numberOrNull(event.event_id),
        timestamp: event.timestamp ?? null,
        decision: String(
          payload.replay_decision ||
            (eventType(event) === "causal_replay.failed_closed"
              ? "failed_closed"
              : "not_recorded"),
        ),
        maybeDecision: String(payload.maybe_gate_decision || "not_recorded"),
        winnerCandidateId:
          typeof winnerCandidateId === "string" && winnerCandidateId
            ? winnerCandidateId
            : null,
        winnerDocumentId:
          typeof winnerDocumentId === "string" && winnerDocumentId
            ? winnerDocumentId
            : null,
        winnerGain: numberOrNull(payload.winner_gain),
        winnerKl: numberOrNull(payload.winner_kl),
        scheduledNextTurn: booleanOrNull(payload.scheduled_next_turn),
        latencySeconds: numberOrNull(payload.latency_seconds),
        errorType:
          typeof errorType === "string" && errorType ? errorType : null,
        losses: asObjects(payload.losses).map((loss) => ({
          candidateId:
            typeof loss.candidate_id === "string" && loss.candidate_id
              ? loss.candidate_id
              : null,
          documentId:
            typeof loss.document_id === "string" && loss.document_id
              ? loss.document_id
              : null,
          nll: numberOrNull(loss.nll),
          gainVsBaseline: numberOrNull(loss.gain_vs_baseline),
          observedTokenKl: numberOrNull(loss.observed_token_kl),
          observationTokens: numberOrNull(loss.observation_tokens),
        })),
      };
    });
  const maybe: MaybeEvidence[] = ordered
    .filter((event) => eventType(event) === "maybe.completed")
    .map((event) => {
      const payload = eventPayload(event);
      return {
        eventId: numberOrNull(event.event_id),
        timestamp: event.timestamp ?? null,
        status: String(payload.status || "not_recorded"),
        decision: String(
          payload.maybe_decision ||
            payload.maybe_gate_decision ||
            "not_recorded",
        ),
        scheduledNextTurn:
          booleanOrNull(payload.maybe_scheduled_next_turn) ??
          booleanOrNull(payload.scheduled_next_turn),
      };
    });
  return {
    requestStatus: requestStatus(ordered),
    promptTokens:
      numberOrNull(probeStarted.prompt_tokens) ??
      numberOrNull(memoryProbe.prompt_tokens),
    queryTokens: numberOrNull(probeStarted.query_tokens),
    cognitionProbeTokens: numberOrNull(probeStarted.cognition_tokens),
    probeLatencySeconds: numberOrNull(probeCompleted.latency_seconds),
    attachedTokens: Number(
      memory.attached_tokens || trace.attached_tokens || 0,
    ),
    nativeRestore: {
      active: Boolean(nativeRestore.active || trace.native_restore),
      lane: String(nativeRestore.lane || trace.native_restore?.lane || ""),
      tokens: Number(nativeRestore.tokens || trace.native_restore?.tokens || 0),
      reason: String(
        nativeRestore.selection_reason ||
          trace.native_restore?.selection_reason ||
          "",
      ),
    },
    selectedKnowledgeIds,
    selectedPolicyIds,
    retrievalSeconds: numberOrNull(memory.retrieval_latency_seconds),
    judgeSeconds: numberOrNull(memory.judge_latency_seconds),
    judgeQuestionTruncated: Boolean(requestJudge.question_truncated),
    judgeQuestionOriginalTokens: numberOrNull(
      requestJudge.question_original_tokens,
    ),
    judgeQuestionReviewTokens: numberOrNull(
      requestJudge.question_review_tokens,
    ),
    candidates,
    rejectedCount,
    fullyJudged: candidates.length > 0 && rejectedCount === candidates.length,
    qkRecall: {
      status: String(rankAudit.status || "not_run"),
      reason: String(rankAudit.reason || "audit_unavailable"),
      preset: String(rankAudit.preset || qkRetrieval.preset || "balanced"),
      minTensorScore: numberOrNull(rankAudit.min_tensor_score),
      minDocumentMargin: numberOrNull(rankAudit.min_document_margin),
      bestScore: numberOrNull(rankAudit.top_score),
      secondScore: numberOrNull(rankAudit.runner_up_score),
      observedMargin: numberOrNull(rankAudit.observed_margin),
      consideredDocuments: numberOrNull(rankAudit.considered_documents),
      candidateCount: numberOrNull(rankAudit.candidate_count),
      scoredDocuments: asObjects(rankAudit.scored_documents).map(
        (document) => ({
          lane: String(document.lane || "knowledge"),
          documentId: String(document.document_id || ""),
          path: String(
            document.relative_path ||
              document.document_id ||
              UNKNOWN_DOCUMENT_SOURCE,
          ),
          score: numberOrNull(document.tensor_score),
          passedScore: Boolean(document.passed_score),
          rejectionReason: String(document.rejection_reason || ""),
        }),
      ),
    },
    selfAsk: deduplicateSelfAsk(trace.self_ask),
    causalReplay,
    maybe,
    events: ordered,
  };
}

function formatScore(value: number | null) {
  return value === null ? "—" : formatFixedNumber(value, 3);
}

function qkPresetLabel(preset: string) {
  if (preset === "broad") return t("高召回");
  if (preset === "strict") return t("高精度");
  return t("标准");
}

function qkAuditExplanation(audit: QkRecallAudit) {
  if (audit.reason === "top_score_below_threshold") {
    return t(
      "最高文档分 {best}，没有达到本档要求的 {minimum}，因此在进入语义审计前被拒绝。",
      {
        best: formatScore(audit.bestScore),
        minimum: formatScore(audit.minTensorScore),
      },
    );
  }
  if (audit.reason === "document_margin_too_small") {
    return t(
      "第一名只领先第二名 {margin}，低于要求的 {minimum}。系统没有在两份难分胜负的文档中强行选择。",
      {
        margin: formatScore(audit.observedMargin),
        minimum: formatScore(audit.minDocumentMargin),
      },
    );
  }
  if (audit.reason === "candidates_ready") {
    return t(
      "Q×K 第一门已通过，产生 {count} 份候选；它们仍需经过语义审计，候选不等于实际注入。",
      { count: formatNumber(audit.candidateCount ?? 0) },
    );
  }
  if (audit.reason === "all_scores_below_threshold") {
    return t("所有文档页都低于本档分数要求，没有候选进入语义审计。");
  }
  if (audit.reason === "no_attention_query") {
    return t("本次请求没有可用的 Attention-Q 窗口，因此未执行 Q×K 排名。");
  }
  if (audit.reason === "no_finite_attention_query") {
    return t("Attention-Q 窗口没有可用于计算的有限数值，排名已安全跳过。");
  }
  if (audit.reason === "tensor_bank_not_ready") {
    return t("Tensor Bank 尚未就绪，Q×K 排名没有运行。");
  }
  if (audit.reason === "query_probe_unavailable") {
    return t("请求开始时没有取得检索探针，Q×K 排名没有运行。");
  }
  return t("当前遥测没有保存 Q×K 第一门的详细审计；旧请求可能没有该字段。");
}

function injectionFailureExplanation(evidence: TraceEvidence) {
  if (evidence.qkRecall.status === "rejected") {
    return t("Q×K 门禁拒绝：{reason}", {
      reason: qkAuditExplanation(evidence.qkRecall),
    });
  }
  if (evidence.selectedKnowledgeIds.length) {
    return evidence.nativeRestore.reason
      ? t("候选已经通过语义准入，但原生状态没有绑定。服务端原因：{reason}。", {
          reason: evidence.nativeRestore.reason,
        })
      : t(
          "候选已经通过语义准入，但既没有文本 token，也没有可验证的原生状态绑定；请下钻原始事件检查预算、状态摘要或前缀绑定失败。",
        );
  }
  if (evidence.rejectedCount) {
    return t(
      "Semantic Judge 已拒绝 {count} 份候选；本次没有 Knowledge 准入、文本注入或 Knowledge 原生恢复。",
      { count: formatNumber(evidence.rejectedCount) },
    );
  }
  if (evidence.candidates.length && evidence.fullyJudged) {
    return t(
      "{count} 份候选全部被 Semantic Judge 判为不相关，因此系统按失败关闭策略保持 0 注入。",
      { count: formatNumber(evidence.candidates.length) },
    );
  }
  if (evidence.candidates.length) {
    return t(
      "候选已经提出，但遥测中没有通过语义准入的文档；候选不会自动等同于注入。",
    );
  }
  return qkAuditExplanation(evidence.qkRecall);
}

function selfAskStatusExplanation(status: string) {
  if (status === "ready_for_safe_replay") {
    return t("已找到能回答自问的相关证据，并通过安全回放门禁。");
  }
  if (status === "policy_reflection_ready") {
    return t(
      "策略反思已经形成，仅作为模型内部上下文使用，不会追加为用户指令。",
    );
  }
  if (status === "context_evidence_ready") {
    return t("工具结果中的上下文证据已经通过语义门禁，可供当前推理使用。");
  }
  if (status === "context_integrity_ready") {
    return t(
      "最新工具结果证明历史上下文存在冲突；更正已作为内部 think 上下文传递，不会追加为用户指令。",
    );
  }
  if (status === "no_eligible_reference") {
    return t(
      "没有参考通过语义相关性门禁；系统失败关闭，不生成也不注入 Self-Answer。",
    );
  }
  if (status === "no_answering_evidence") {
    return t("存在相关参考，但它不能回答这次自问；系统拒绝注入不完整答案。");
  }
  if (status === "self_ask_skipped") {
    return t("本次 Self-Ask 因预算、去重或调度条件被跳过，没有产生注入。");
  }
  if (status === "replay_rejected") {
    return t("候选答案在同一未来 token 上没有取得足够增益，因果回放拒绝采用。");
  }
  if (status.startsWith("failed_closed")) {
    return t("生成或审计发生错误，服务端按失败关闭策略拒绝注入。");
  }
  return status
    ? t("服务端状态：{status}。未记录更细的解释，请结合原始事件判断。", {
        status,
      })
    : t("服务端没有记录 Self-Ask 状态。");
}

function selfAskSucceeded(status: string) {
  return [
    "ready_for_safe_replay",
    "policy_reflection_ready",
    "context_evidence_ready",
    "context_integrity_ready",
  ].includes(status);
}

function causalReplayDecisionLabel(decision: string) {
  return decision.startsWith("failed_closed")
    ? t("失败关闭")
    : translatedCodeLabel(
        decision,
        CAUSAL_REPLAY_DECISION_LABELS,
        "状态未记录",
      );
}

function maybeDecisionLabel(decision: string) {
  return translatedCodeLabel(decision, MAYBE_DECISION_LABELS, "未记录");
}

function causalReplayExplanation(replay: CausalReplayEvidence) {
  if (replay.eventType === "causal_replay.failed_closed") {
    return replay.errorType
      ? t("回放评分发生 {error}；服务端失败关闭，没有安排下一轮状态。", {
          error: replay.errorType,
        })
      : t("回放评分异常；服务端失败关闭，没有安排下一轮状态。");
  }
  if (replay.decision === "shadow_would_switch") {
    return replay.scheduledNextTurn
      ? t(
          "候选分支在同一组未来 token 上优于基线，并通过 KL 门禁；已安排为下一轮可恢复状态。",
        )
      : t(
          "候选分支优于基线，但没有通过 Maybe/KL 门禁；本轮输出未被改写，也没有安排下一轮状态。",
        );
  }
  if (replay.decision === "reject_insufficient_gain") {
    return t("候选分支相对基线的 NLL 改善不足，回放拒绝切换。");
  }
  if (replay.decision === "reject_switch_margin") {
    return t("新候选没有超过当前候选所需的切换边际，回放保持现状。");
  }
  if (replay.decision === "reject_no_semantic_candidate") {
    return t("没有候选通过语义准入，回放未执行候选分支评分。");
  }
  if (replay.decision === "insufficient_future_observation") {
    return t("触发点后的真实未来 token 不足，无法进行同窗口因果比较。");
  }
  if (replay.decision === "reject_empty_prefix") {
    return t("没有可复用的父请求前缀，回放按失败关闭策略停止。");
  }
  if (replay.decision === "reject_empty_candidate_state") {
    return t("候选没有可评分的原生状态或参考 token，回放没有启动分支比较。");
  }
  if (replay.decision === "reject_no_challenger") {
    return t("回放只有基线分支，没有可比较的候选分支。");
  }
  return t("服务端记录了回放终态；可展开原始事件检查完整字段。");
}

function eventSummary(event: TelemetryEvent) {
  const type = eventType(event);
  const payload = eventPayload(event);
  if (type === "tensor.candidates_proposed") {
    return t("{count} 份候选 · {source}", {
      count: formatNumber(asObjects(payload.candidates).length),
      source: translatedCodeLabel(
        payload.query_source,
        QUERY_SOURCE_LABELS,
        "来源未知",
      ),
    });
  }
  if (type === "semantic_judge.completed") {
    return t("{candidates} 份候选 · {eligible} 份通过", {
      candidates: Number(payload.candidate_count || 0),
      eligible: Number(payload.eligible_count || 0),
    });
  }
  if (type === "memory.prepared") {
    const policy = asObject(payload.policy_data);
    return t("{count} 份准入 · {tokens} 文本 token", {
      count: formatNumber(
        strings(payload.selected_document_ids).length +
          strings(policy.document_ids).length,
      ),
      tokens: formatNumber(Number(payload.attached_tokens || 0)),
    });
  }
  if (type === "observer.decode_summary") {
    return `token ${formatNumber(Number(payload.token_count || 0))} · surprisal ${formatFixedNumber(Number(payload.ema_surprisal || 0), 3)}`;
  }
  if (type === "adaptive.transition") {
    return `${String(payload.from ?? "—")} → ${String(payload.to ?? "—")}${payload.decision ? ` · ${String(payload.decision)}` : ""}`;
  }
  if (type === "context_integrity.started") {
    return t("正在核对最新工具结果与历史上下文");
  }
  if (type === "context_integrity.completed") {
    const status = String(payload.status || "unknown");
    if (status === "consistent") return t("无需干预 · 未修改上下文");
    if (status === "corrected") return t("发现冲突 · 等待更正准入");
    if (status === "uncertain") return t("证据不足 · 未修改上下文");
    return t("检查结论 {status} · 未确认干预", { status });
  }
  if (type === "context_integrity.applied") {
    return t("已干预 · 内部 think 更正已准入 · 未追加用户文本");
  }
  if (type === "context_integrity.skipped") {
    return t("未干预 · {reason}", {
      reason: translatedCodeLabel(
        payload.reason,
        CONTEXT_INTEGRITY_REASON_LABELS,
        "更正未获得准入",
      ),
    });
  }
  if (type === "context_integrity.failed_closed") {
    return t("检查失败关闭 · 未修改上下文");
  }
  if (type === "causal_replay.completed") {
    return t("{decision} · 增益 {gain} · {schedule}", {
      decision: causalReplayDecisionLabel(
        String(payload.replay_decision || ""),
      ),
      gain: formatScore(numberOrNull(payload.winner_gain)),
      schedule: payload.scheduled_next_turn
        ? t("已安排下一轮")
        : t("未安排下一轮"),
    });
  }
  if (type === "causal_replay.failed_closed") {
    return t("失败关闭 · {error}", {
      error: String(payload.error_type || t("未知错误")),
    });
  }
  if (type === "maybe.completed") {
    return t("{status} · {decision}", {
      status: selfAskStatusLabel(String(payload.status || "")),
      decision: maybeDecisionLabel(
        String(payload.maybe_decision || payload.maybe_gate_decision || ""),
      ),
    });
  }
  if (type.startsWith("refresh")) {
    return String(
      payload.question || payload.status || payload.purpose || t("刷新事件"),
    );
  }
  if (type === "query_probe.started") {
    return t("{tokens} token · {spans} 个查询片段", {
      tokens: formatNumber(Number(payload.prompt_tokens || 0)),
      spans: formatNumber(Number(payload.span_count || 0)),
    });
  }
  if (type === "query_probe.completed") {
    return t("{status} · {latency}s", {
      status: translatedCodeLabel(
        payload.status,
        QUERY_PROBE_STATUS_LABELS,
        "完成",
      ),
      latency: formatFixedNumber(Number(payload.latency_seconds || 0), 2),
    });
  }
  if (type === "request.started")
    return String(payload.input || t("请求已开始"));
  if (type === "request.completed") {
    return t("{count} 输出 token", {
      count: formatNumber(Number(payload.output_tokens || 0)),
    });
  }
  const scalar = Object.entries(payload)
    .filter(([, value]) =>
      ["string", "number", "boolean"].includes(typeof value),
    )
    .slice(0, 3)
    .map(([key, value]) => `${key}=${String(value)}`);
  return scalar.join(" · ") || t("查看原始负载");
}

function MetricTile({
  label,
  value,
  detail,
  icon: Icon,
}: {
  label: string;
  value: string;
  detail: string;
  icon: typeof Database;
}) {
  return (
    <div className="min-w-0 border-l-2 border-border px-3 py-1 first:border-l-0 first:pl-0 sm:first:border-l-2 sm:first:pl-3">
      <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
        <Icon className="h-3.5 w-3.5" />
        {label}
      </div>
      <div className="mt-1 font-mono text-xl font-semibold tracking-tight">
        {value}
      </div>
      <div
        className="mt-0.5 truncate text-[11px] text-muted-foreground"
        title={detail}
      >
        {detail}
      </div>
    </div>
  );
}

function PipelineStage({
  index,
  title,
  value,
  detail,
  state,
}: {
  index: number;
  title: string;
  value: string;
  detail: string;
  state: "success" | "warning" | "neutral";
}) {
  return (
    <div className="relative min-w-0 flex-1 px-3 py-3 first:pl-0 last:pr-0">
      <div className="flex items-center gap-2">
        <span
          className={cn(
            "grid h-6 w-6 shrink-0 place-items-center rounded-full border font-mono text-[10px] font-semibold",
            state === "success" &&
              "border-emerald-500/40 bg-emerald-500/10 text-emerald-600 dark:text-emerald-300",
            state === "warning" &&
              "border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-300",
            state === "neutral" && "bg-muted text-muted-foreground",
          )}
        >
          {index}
        </span>
        <div className="min-w-0">
          <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-muted-foreground">
            {title}
          </div>
          <div className="truncate text-sm font-semibold">{value}</div>
        </div>
      </div>
      <p className="mt-2 line-clamp-2 text-[11px] leading-5 text-muted-foreground">
        {detail}
      </p>
      {index < 4 ? (
        <ChevronRight className="absolute -right-2 top-5 hidden h-4 w-4 text-border xl:block" />
      ) : null}
    </div>
  );
}

function RequestRow({
  trace,
  selected,
  onSelect,
}: {
  trace: RequestTrace;
  selected: boolean;
  onSelect: () => void;
}) {
  const running = trace.duration_seconds === null;
  const actualKnowledge = hasActualKnowledge(trace);
  const policyDataRestored = hasPolicyDataRestore(trace);
  const integrityStatus = contextIntegrityListStatus(trace);
  const replayEventCount = causalReplayEventCount(trace);
  return (
    <button
      type="button"
      data-request-id={trace.request_id}
      onClick={onSelect}
      className={cn(
        "group relative w-full border-b px-4 py-3 text-left transition-colors last:border-b-0 hover:bg-muted/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring",
        selected && "bg-primary/[0.06] hover:bg-primary/[0.08]",
      )}
    >
      <span
        className={cn(
          "absolute inset-y-2 left-0 w-0.5 rounded-full bg-transparent",
          selected && "bg-primary",
        )}
      />
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 font-mono text-[11px] text-muted-foreground">
          <span
            className={cn(
              "h-1.5 w-1.5 rounded-full",
              running ? "animate-pulse bg-blue-500" : "bg-slate-400",
            )}
          />
          {timeOnly(trace.started_at)}
        </div>
        <span className="font-mono text-[10px] text-muted-foreground">
          {running ? t("推理中") : formatDuration(trace.duration_seconds)}
        </span>
      </div>
      <p className="mt-2 line-clamp-2 text-[13px] font-medium leading-5">
        {trace.input_text || t("该事件片段缺少请求输入")}
      </p>
      <div className="mt-2.5 flex items-center gap-1.5">
        {actualKnowledge ? (
          <Badge variant="success" className="px-1.5 py-0 text-[10px]">
            {t("知识已注入")}
          </Badge>
        ) : trace.candidates.length ? (
          <Badge variant="warning" className="px-1.5 py-0 text-[10px]">
            {t("候选 {count}", {
              count: formatNumber(trace.candidates.length),
            })}
          </Badge>
        ) : (
          <Badge variant="secondary" className="px-1.5 py-0 text-[10px]">
            {t("无候选")}
          </Badge>
        )}
        {policyDataRestored ? (
          <Badge variant="outline" className="px-1.5 py-0 text-[10px]">
            {t("PolicyData 原生恢复")}
          </Badge>
        ) : null}
        {trace.self_ask.length ? (
          <span className="text-[10px] text-muted-foreground">
            Self-Ask ×{formatNumber(trace.self_ask.length)}
          </span>
        ) : null}
        {replayEventCount ? (
          <Badge variant="outline" className="px-1.5 py-0 text-[10px]">
            {t("回放 ×{count}", { count: formatNumber(replayEventCount) })}
          </Badge>
        ) : null}
        {integrityStatus ? (
          <Badge
            variant={integrityStatus.variant}
            className="px-1.5 py-0 text-[10px]"
          >
            {integrityStatus.label}
          </Badge>
        ) : null}
        <span className="ml-auto font-mono text-[9px] text-muted-foreground/70">
          {shortHash(trace.request_id, 9)}
        </span>
      </div>
    </button>
  );
}

function CandidateRow({
  candidate,
  rank,
  onOpenDocument,
}: {
  candidate: CandidateEvidence;
  rank: number;
  onOpenDocument: (candidate: CandidateEvidence) => void;
}) {
  const rejected = decisionIsRejected(candidate.decisionStatus);
  const status = rejected
    ? "rejected"
    : candidate.selected
      ? "injected"
      : "proposed";
  return (
    <div
      data-candidate-id={candidate.candidateId}
      data-recall-status={status}
      className={cn(
        "relative overflow-hidden rounded-lg border bg-card",
        status === "injected" && "border-emerald-500/40",
      )}
    >
      <div
        className={cn(
          "absolute inset-y-0 left-0 w-1 bg-muted",
          status === "injected" && "bg-emerald-500",
          status === "rejected" && "bg-amber-400",
          status === "proposed" && "bg-blue-400",
        )}
      />
      <div className="p-3.5 pl-5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="flex min-w-0 flex-1 items-start gap-3">
            <span className="grid h-7 w-7 shrink-0 place-items-center rounded-md bg-muted font-mono text-[11px] font-semibold text-muted-foreground">
              {formatNumber(rank)}
            </span>
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-2">
                <span className="max-w-full break-all font-mono text-xs font-semibold">
                  {candidate.path === UNKNOWN_DOCUMENT_SOURCE
                    ? t(UNKNOWN_DOCUMENT_SOURCE)
                    : candidate.path}
                </span>
                <Badge variant="outline" className="px-1.5 py-0 text-[10px]">
                  {laneLabel(candidate.lane)}
                </Badge>
                {status === "injected" ? (
                  <Badge variant="success" className="px-1.5 py-0 text-[10px]">
                    <CheckCircle2 className="h-3 w-3" /> {t("已准入并注入")}
                  </Badge>
                ) : status === "rejected" ? (
                  <Badge variant="warning" className="px-1.5 py-0 text-[10px]">
                    <XCircle className="h-3 w-3" /> {t("Judge 拒绝")}
                  </Badge>
                ) : (
                  <Badge
                    variant="secondary"
                    className="px-1.5 py-0 text-[10px]"
                  >
                    {t("仅提出")}
                  </Badge>
                )}
              </div>
              <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 font-mono text-[10px] text-muted-foreground">
                <span>tensor {formatScore(candidate.tensorScore)}</span>
                <span>
                  {t("综合 {score}", { score: formatScore(candidate.score) })}
                </span>
                <span>
                  page{" "}
                  {candidate.pageIds.length
                    ? candidate.pageIds
                        .map((page) => formatNumber(page))
                        .join(", ")
                    : "—"}
                </span>
                <span>
                  {t("源 token")} {formatNumber(candidate.sourcePositionMin)}–
                  {formatNumber(candidate.sourcePositionMax)}
                  {candidate.sourcePositionCount
                    ? t(" · {count} 位", {
                        count: formatNumber(candidate.sourcePositionCount),
                      })
                    : ""}
                </span>
                <span>
                  {t("QK 峰值 {score}", {
                    score: formatScore(candidate.topAttributionScore),
                  })}
                </span>
              </div>
            </div>
          </div>
          <Button
            variant="ghost"
            size="sm"
            className="shrink-0"
            onClick={() => onOpenDocument(candidate)}
          >
            <FileSearch />
            {t("查看文档")}
          </Button>
        </div>
        {candidate.excerpt ? (
          <div className="mt-3 rounded-md bg-muted/45 px-3 py-2.5">
            <div className="mb-1 text-[9px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
              {t("文档预览 · 命中范围见上方 token 证据")}
            </div>
            <p className="line-clamp-3 whitespace-pre-wrap text-xs leading-5 text-muted-foreground">
              {candidate.excerpt}
            </p>
          </div>
        ) : null}
        <div className="mt-2 flex flex-wrap gap-x-4 text-[10px] text-muted-foreground">
          <span>
            {t("来源：{source}", { source: candidate.origin || t("未记录") })}
          </span>
          <span>
            {t("Judge：{method}", {
              method: candidate.judgeMethod || t("未执行/未记录"),
            })}
          </span>
        </div>
      </div>
    </div>
  );
}

function EventRow({ event }: { event: TelemetryEvent }) {
  const type = eventType(event);
  return (
    <details className="group border-b last:border-b-0">
      <summary className="grid cursor-pointer list-none grid-cols-[54px_minmax(150px,240px)_minmax(0,1fr)] items-center gap-3 px-3 py-2.5 hover:bg-muted/50 [&::-webkit-details-marker]:hidden">
        <span className="font-mono text-[10px] text-muted-foreground">
          #{event.event_id ?? "—"}
        </span>
        <span className="truncate text-xs font-semibold" title={type}>
          {t(EVENT_TYPE_LABELS[type] ?? type)}
        </span>
        <span className="truncate text-[11px] text-muted-foreground">
          {eventSummary(event)}
        </span>
      </summary>
      <div className="border-t bg-slate-950 p-3 text-slate-200">
        <div className="mb-2 flex items-center justify-between font-mono text-[10px] text-slate-400">
          <span>{type}</span>
          <span>{dateTime(event.timestamp)}</span>
        </div>
        <pre className="max-h-80 overflow-auto whitespace-pre-wrap break-words font-mono text-[11px] leading-5">
          {JSON.stringify(event.payload || {}, null, 2)}
        </pre>
      </div>
    </details>
  );
}

function EvidencePanel({
  trace,
  evidence,
  loading,
}: {
  trace: RequestTrace;
  evidence: TraceEvidence | null;
  loading: boolean;
}) {
  const [showAllCandidates, setShowAllCandidates] = useState(false);
  const [documentCandidate, setDocumentCandidate] =
    useState<CandidateEvidence | null>(null);
  const [documentContent, setDocumentContent] = useState("");
  const [documentLoading, setDocumentLoading] = useState(false);

  useEffect(() => {
    setShowAllCandidates(false);
    setDocumentCandidate(null);
    setDocumentContent("");
  }, [trace.request_id]);

  const openDocument = async (candidate: CandidateEvidence) => {
    setDocumentCandidate(candidate);
    setDocumentContent("");
    setDocumentLoading(true);
    try {
      const lane = candidate.lane === "policydata" ? "policydata" : "knowledge";
      const result = await getSource(lane, candidate.path);
      setDocumentContent(result.content);
    } catch (error) {
      toast.error(t("文档读取失败"), {
        description: error instanceof Error ? error.message : t("未知错误"),
      });
    } finally {
      setDocumentLoading(false);
    }
  };

  if (loading || !evidence) {
    return (
      <div className="grid min-h-[560px] place-items-center rounded-xl border bg-card">
        <div className="text-center">
          <LoaderCircle className="mx-auto h-5 w-5 animate-spin text-muted-foreground" />
          <p className="mt-3 text-sm text-muted-foreground">
            {t("正在归并请求证据…")}
          </p>
        </div>
      </div>
    );
  }

  const status = statusMeta(evidence.requestStatus);
  const hasKnowledgeInjection =
    evidence.selectedKnowledgeIds.length > 0 ||
    evidence.attachedTokens > 0 ||
    (evidence.nativeRestore.active &&
      evidence.nativeRestore.lane === "knowledge");
  const policyDataRestored =
    evidence.selectedPolicyIds.length > 0 ||
    (evidence.nativeRestore.active &&
      evidence.nativeRestore.lane === "policydata");
  const visibleCandidates = showAllCandidates
    ? evidence.candidates
    : evidence.candidates.slice(0, 6);
  const contextIntegrityEvents = evidence.events.filter((event) =>
    eventType(event).startsWith("context_integrity."),
  );
  const causalReplayTelemetryCount =
    evidence.causalReplay.length + evidence.maybe.length;
  const observerTriggered = evidence.events.some((event) => {
    const type = eventType(event);
    if (type === "observer.mid_think_triggered") return true;
    return (
      type === "observer.decode_summary" &&
      Boolean(eventPayload(event).triggered)
    );
  });
  const integrityVerdict = contextIntegrityVerdict(contextIntegrityEvents);
  const ContextIntegrityIcon = integrityVerdict?.icon ?? CircleDot;
  const verdict = hasKnowledgeInjection
    ? {
        title: t("已实际注入 {count} 份知识", {
          count: formatNumber(
            Math.max(1, evidence.selectedKnowledgeIds.length),
          ),
        }),
        description: t(
          "下方绿色条目标记服务端已准入并绑定到本次请求的文档；这是实际注入证据，不是候选排名。",
        ),
        icon: CheckCircle2,
        tone: "success" as const,
      }
    : evidence.rejectedCount
      ? {
          title: t("0 份知识进入本次模型上下文"),
          description: t(
            "Semantic Judge 已拒绝 {count} 份候选；本次没有 Knowledge 准入、文本注入或 Knowledge 原生恢复。",
            { count: formatNumber(evidence.rejectedCount) },
          ),
          icon: XCircle,
          tone: "warning" as const,
        }
      : evidence.candidates.length
        ? {
            title: t("0 份知识进入本次模型上下文"),
            description: t(
              "{count} 份候选已提出，但当前事件中没有实际准入或注入证据。",
              { count: formatNumber(evidence.candidates.length) },
            ),
            icon: AlertTriangle,
            tone: "warning" as const,
          }
        : {
            title: t("本次请求没有知识候选"),
            description: qkAuditExplanation(evidence.qkRecall),
            icon: CircleDot,
            tone: "neutral" as const,
          };
  const VerdictIcon = verdict.icon;

  return (
    <div className="min-w-0 overflow-hidden rounded-xl border bg-card shadow-sm">
      <div className="border-b px-5 py-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant={status.variant}>
                <span className={cn("h-1.5 w-1.5 rounded-full", status.dot)} />
                {status.label}
              </Badge>
              <span className="text-xs text-muted-foreground">
                {dateTime(trace.started_at)} ·{" "}
                {formatDuration(trace.duration_seconds)}
              </span>
            </div>
            <h2 className="mt-3 whitespace-pre-wrap text-base font-semibold leading-7">
              {trace.input_text || t("该请求片段没有记录输入文本")}
            </h2>
          </div>
          <Button
            variant="outline"
            size="sm"
            className="font-mono text-[10px]"
            onClick={() => {
              void navigator.clipboard.writeText(trace.request_id);
              toast.success(t("请求 ID 已复制"));
            }}
          >
            <Copy />
            {shortHash(trace.request_id, 14)}
          </Button>
        </div>
      </div>

      <div
        className={cn(
          "border-b px-5 py-4",
          verdict.tone === "success" && "bg-emerald-500/[0.06]",
          verdict.tone === "warning" && "bg-amber-500/[0.07]",
          verdict.tone === "neutral" && "bg-muted/35",
        )}
      >
        <div className="flex items-start gap-3">
          <div
            className={cn(
              "mt-0.5 grid h-9 w-9 shrink-0 place-items-center rounded-full",
              verdict.tone === "success" &&
                "bg-emerald-500/15 text-emerald-600 dark:text-emerald-300",
              verdict.tone === "warning" &&
                "bg-amber-500/15 text-amber-700 dark:text-amber-300",
              verdict.tone === "neutral" && "bg-muted text-muted-foreground",
            )}
          >
            <VerdictIcon className="h-4 w-4" />
          </div>
          <div>
            {!hasKnowledgeInjection && evidence.rejectedCount ? (
              <Badge variant="warning" className="mb-1 px-1.5 py-0 text-[10px]">
                {t("Judge 拒绝")}
              </Badge>
            ) : null}
            <div className="text-sm font-semibold">{verdict.title}</div>
            <p className="mt-1 text-xs leading-5 text-muted-foreground">
              {verdict.description}
            </p>
            {policyDataRestored ? (
              <p className="mt-1 text-[11px] text-muted-foreground">
                {t(
                  "PolicyData 单独恢复 {tokens} token；它是始终生效的策略状态，不等于 Knowledge 候选通过或知识注入。",
                  { tokens: formatNumber(evidence.nativeRestore.tokens) },
                )}
              </p>
            ) : null}
            {evidence.nativeRestore.active &&
            evidence.nativeRestore.lane === "cognition" ? (
              <p className="mt-1 text-[11px] text-muted-foreground">
                {t(
                  "Cognition 基础状态恢复 {tokens} token；它不等于知识文档召回。",
                  { tokens: formatNumber(evidence.nativeRestore.tokens) },
                )}
              </p>
            ) : null}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-y-4 border-b px-5 py-4 sm:grid-cols-4">
        <MetricTile
          label={t("检索上下文")}
          value={`${formatNumber(evidence.promptTokens)} tok`}
          detail={`query ${formatNumber(evidence.queryTokens)} + cognition ${formatNumber(evidence.cognitionProbeTokens)}`}
          icon={Search}
        />
        <MetricTile
          label={t("文本注入")}
          value={`${formatNumber(evidence.attachedTokens)} tok`}
          detail={t("{count} 份知识文档", {
            count: formatNumber(evidence.selectedKnowledgeIds.length),
          })}
          icon={FileText}
        />
        <MetricTile
          label={t("原生恢复")}
          value={`${formatNumber(evidence.nativeRestore.tokens)} tok`}
          detail={
            evidence.nativeRestore.active
              ? laneLabel(evidence.nativeRestore.lane)
              : t("未恢复")
          }
          icon={Database}
        />
        <MetricTile
          label={t("模型输出")}
          value={`${formatNumber(trace.output_tokens)} tok`}
          detail={
            evidence.requestStatus === "running" ? t("仍在生成") : t("本次响应")
          }
          icon={Activity}
        />
      </div>

      {trace.output_text ? (
        <details className="border-b px-5 py-3">
          <summary className="cursor-pointer text-xs font-semibold text-muted-foreground hover:text-foreground">
            {t("查看模型输出文本")}
          </summary>
          <pre className="mt-3 max-h-64 overflow-auto whitespace-pre-wrap rounded-md bg-muted/50 p-3 text-xs leading-5">
            {trace.output_text}
          </pre>
        </details>
      ) : null}

      <Tabs defaultValue="recall" className="px-5 py-4">
        <TabsList className="grid w-full grid-cols-2 sm:w-auto sm:inline-grid sm:grid-cols-5">
          <TabsTrigger value="recall">
            {t("召回判定 · {count}", {
              count: formatNumber(evidence.candidates.length),
            })}
          </TabsTrigger>
          <TabsTrigger value="self-ask">
            Self-Ask · {formatNumber(evidence.selfAsk.length)}
          </TabsTrigger>
          <TabsTrigger value="causal-replay">
            {t("因果回放 · {count}", {
              count: formatNumber(causalReplayTelemetryCount),
            })}
          </TabsTrigger>
          <TabsTrigger value="integrity">
            {t("完整性 · {count}", {
              count: formatNumber(contextIntegrityEvents.length),
            })}
          </TabsTrigger>
          <TabsTrigger value="events">
            {t("原始事件 · {count}", {
              count: formatNumber(evidence.events.length),
            })}
          </TabsTrigger>
        </TabsList>

        <TabsContent value="recall" className="space-y-4">
          <div
            className={cn(
              "rounded-lg border p-4",
              evidence.qkRecall.status === "ready" &&
                "border-emerald-200 bg-emerald-500/[0.05]",
              evidence.qkRecall.status === "rejected" &&
                "border-amber-200 bg-amber-500/[0.06]",
            )}
          >
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="text-sm font-semibold">
                    {t("Q×K 召回第一道门")}
                  </h3>
                  <Badge
                    variant={
                      evidence.qkRecall.status === "ready"
                        ? "success"
                        : evidence.qkRecall.status === "rejected"
                          ? "warning"
                          : "outline"
                    }
                  >
                    {evidence.qkRecall.status === "ready"
                      ? t("已提出候选")
                      : evidence.qkRecall.status === "rejected"
                        ? t("已拒绝")
                        : t("未运行")}
                  </Badge>
                </div>
                <p className="mt-1 max-w-3xl text-xs leading-5 text-muted-foreground">
                  {qkAuditExplanation(evidence.qkRecall)}
                </p>
              </div>
              <Badge variant="outline">
                {t("{preset}档", {
                  preset: qkPresetLabel(evidence.qkRecall.preset),
                })}
              </Badge>
            </div>
            <div className="mt-3 grid gap-2 text-xs sm:grid-cols-4">
              <div className="rounded-md bg-background/70 p-2.5">
                <div className="text-muted-foreground">
                  {t("最高分 / 最低要求")}
                </div>
                <div className="mt-1 font-mono font-semibold">
                  {formatScore(evidence.qkRecall.bestScore)} /{" "}
                  {formatScore(evidence.qkRecall.minTensorScore)}
                </div>
              </div>
              <div className="rounded-md bg-background/70 p-2.5">
                <div className="text-muted-foreground">
                  {t("领先差距 / 最低要求")}
                </div>
                <div className="mt-1 font-mono font-semibold">
                  {formatScore(evidence.qkRecall.observedMargin)} /{" "}
                  {formatScore(evidence.qkRecall.minDocumentMargin)}
                </div>
              </div>
              <div className="rounded-md bg-background/70 p-2.5">
                <div className="text-muted-foreground">{t("参与排名文档")}</div>
                <div className="mt-1 font-mono font-semibold">
                  {formatNumber(evidence.qkRecall.consideredDocuments)}
                </div>
              </div>
              <div className="rounded-md bg-background/70 p-2.5">
                <div className="text-muted-foreground">{t("拒绝代码")}</div>
                <div className="mt-1 truncate font-mono font-semibold">
                  {evidence.qkRecall.reason}
                </div>
              </div>
            </div>
            {evidence.qkRecall.scoredDocuments.length ? (
              <div className="mt-3 overflow-hidden rounded-md border bg-background/70">
                <div className="border-b px-3 py-2 text-[11px] font-semibold text-muted-foreground">
                  {t("门前文档排名（尚未进入 Semantic Judge）")}
                </div>
                <div className="divide-y">
                  {evidence.qkRecall.scoredDocuments
                    .slice(0, 6)
                    .map((document, index) => (
                      <div
                        key={`${document.lane}:${document.documentId}:${index}`}
                        className="grid gap-2 px-3 py-2.5 text-xs sm:grid-cols-[32px_minmax(0,1fr)_90px_96px] sm:items-center"
                      >
                        <span className="font-mono text-muted-foreground">
                          #{formatNumber(index + 1)}
                        </span>
                        <div className="min-w-0">
                          <div className="truncate font-medium">
                            {document.path === UNKNOWN_DOCUMENT_SOURCE
                              ? t(UNKNOWN_DOCUMENT_SOURCE)
                              : document.path}
                          </div>
                          <div className="mt-0.5 text-[10px] text-muted-foreground">
                            {laneLabel(document.lane)}
                          </div>
                        </div>
                        <span className="font-mono">
                          {formatScore(document.score)}
                        </span>
                        <Badge
                          variant={document.passedScore ? "success" : "warning"}
                        >
                          {document.passedScore ? t("分数通过") : t("低于门槛")}
                        </Badge>
                      </div>
                    ))}
                </div>
              </div>
            ) : null}
          </div>
          <div className="grid divide-y rounded-lg border px-4 sm:grid-cols-4 sm:divide-x sm:divide-y-0">
            <PipelineStage
              index={1}
              title="Attention-Q"
              value={`${formatNumber(evidence.promptTokens)} token`}
              detail={t("探针 {latency}", {
                latency:
                  evidence.probeLatencySeconds === null
                    ? "—"
                    : `${formatFixedNumber(evidence.probeLatencySeconds, 2)}s`,
              })}
              state={evidence.promptTokens ? "success" : "neutral"}
            />
            <PipelineStage
              index={2}
              title={t("候选生成")}
              value={t("{count} 份文档", {
                count: formatNumber(evidence.candidates.length),
              })}
              detail={t("检索 {latency}", {
                latency:
                  evidence.retrievalSeconds === null
                    ? "—"
                    : `${formatFixedNumber(evidence.retrievalSeconds, 2)}s`,
              })}
              state={evidence.candidates.length ? "success" : "neutral"}
            />
            <PipelineStage
              index={3}
              title="Semantic Judge"
              value={t("{passed} 通过 / {rejected} 拒绝", {
                passed: formatNumber(evidence.selectedKnowledgeIds.length),
                rejected: formatNumber(evidence.rejectedCount),
              })}
              detail={[
                `Judge ${
                  evidence.judgeSeconds === null
                    ? "—"
                    : `${formatFixedNumber(evidence.judgeSeconds, 2)}s`
                }`,
                evidence.judgeQuestionTruncated
                  ? t("Review 输入 {review}/{original} token", {
                      review: formatNumber(
                        evidence.judgeQuestionReviewTokens || 0,
                      ),
                      original: formatNumber(
                        evidence.judgeQuestionOriginalTokens || 0,
                      ),
                    })
                  : "",
              ]
                .filter(Boolean)
                .join(" · ")}
              state={
                evidence.selectedKnowledgeIds.length
                  ? "success"
                  : evidence.rejectedCount
                    ? "warning"
                    : "neutral"
              }
            />
            <PipelineStage
              index={4}
              title={t("知识实际注入")}
              value={hasKnowledgeInjection ? t("已绑定") : t("未注入")}
              detail={t("文本 {tokens} · 原生 {lane}", {
                tokens: formatNumber(evidence.attachedTokens),
                lane:
                  evidence.nativeRestore.active &&
                  evidence.nativeRestore.lane === "knowledge"
                    ? laneLabel(evidence.nativeRestore.lane)
                    : t("无"),
              })}
              state={
                hasKnowledgeInjection
                  ? "success"
                  : evidence.candidates.length
                    ? "warning"
                    : "neutral"
              }
            />
          </div>

          {!hasKnowledgeInjection ? (
            <div className="flex items-start gap-3 rounded-lg border border-amber-200 bg-amber-500/[0.06] p-4">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-700 dark:text-amber-300" />
              <div>
                <div className="text-sm font-semibold">
                  {t("为什么没有知识注入")}
                </div>
                <p className="mt-1 text-xs leading-5 text-muted-foreground">
                  {injectionFailureExplanation(evidence)}
                </p>
              </div>
            </div>
          ) : null}

          <div className="flex items-end justify-between gap-3">
            <div>
              <h3 className="text-sm font-semibold">{t("候选与准入证据")}</h3>
              <p className="mt-1 text-[11px] text-muted-foreground">
                {t("黄色表示仅提出后被拒绝；绿色才表示实际进入本次请求。")}
              </p>
            </div>
            <span className="font-mono text-[10px] text-muted-foreground">
              tensor / page / source token / {t("QK 峰值")}
            </span>
          </div>

          {evidence.candidates.length ? (
            <div className="space-y-2.5">
              {visibleCandidates.map((candidate, index) => (
                <CandidateRow
                  key={`${candidate.candidateId || candidate.path}-${index}`}
                  candidate={candidate}
                  rank={index + 1}
                  onOpenDocument={(item) => void openDocument(item)}
                />
              ))}
              {evidence.candidates.length > 6 ? (
                <Button
                  variant="outline"
                  className="w-full"
                  onClick={() => setShowAllCandidates((current) => !current)}
                >
                  {showAllCandidates
                    ? t("收起候选")
                    : t("显示其余 {count} 份候选", {
                        count: formatNumber(evidence.candidates.length - 6),
                      })}
                </Button>
              ) : null}
            </div>
          ) : (
            <div className="rounded-lg border border-dashed p-8 text-center text-sm text-muted-foreground">
              {t("没有候选文档。")}
            </div>
          )}
        </TabsContent>

        <TabsContent value="self-ask">
          {evidence.selfAsk.length ? (
            <div className="space-y-3">
              {evidence.selfAsk.map((row, index) => (
                <div
                  key={`${row.question}-${row.status}-${index}`}
                  className="rounded-lg border p-4"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-2 text-sm font-semibold">
                      <Sparkles className="h-4 w-4 text-primary" />
                      Self-Ask {formatNumber(index + 1)}
                    </div>
                    <Badge
                      variant={
                        selfAskSucceeded(row.status) ? "success" : "warning"
                      }
                    >
                      {selfAskStatusLabel(row.status)}
                    </Badge>
                  </div>
                  <p className="mt-2 text-xs leading-5 text-muted-foreground">
                    {selfAskStatusExplanation(row.status)}
                  </p>
                  {row.question ? (
                    <div className="mt-3">
                      <div className="eyebrow mb-1">{t("自问内容")}</div>
                      <p className="whitespace-pre-wrap text-sm leading-6">
                        {row.question}
                      </p>
                    </div>
                  ) : null}
                  <div className="mt-3 rounded-md bg-muted/50 p-3">
                    <div className="eyebrow mb-1">
                      {t("参考答案 / 注入内容")}
                    </div>
                    <p className="whitespace-pre-wrap text-xs leading-5 text-muted-foreground">
                      {row.answer ||
                        t("没有生成可注入答案；该次 Self-Ask 已失败关闭。")}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="rounded-lg border border-dashed p-10 text-center">
              <BrainCircuit className="mx-auto h-5 w-5 text-muted-foreground" />
              <p className="mt-3 text-sm font-medium">
                {t("本次请求没有 Self-Ask 记录")}
              </p>
              <p className="mt-1 text-xs text-muted-foreground">
                {t("Observer 未触发，或没有满足刷新条件。")}
              </p>
            </div>
          )}
        </TabsContent>

        <TabsContent value="causal-replay">
          {causalReplayTelemetryCount ? (
            <div className="space-y-3">
              {evidence.causalReplay.map((replay, index) => {
                const admitted = replay.scheduledNextTurn === true;
                const failedClosed =
                  replay.eventType === "causal_replay.failed_closed" ||
                  replay.decision.startsWith("failed_closed");
                return (
                  <div
                    key={`${replay.eventId ?? "replay"}-${replay.eventType}-${index}`}
                    className={cn(
                      "overflow-hidden rounded-lg border",
                      admitted && "border-emerald-200 bg-emerald-500/[0.05]",
                      !admitted &&
                        !failedClosed &&
                        "border-amber-200 bg-amber-500/[0.05]",
                      failedClosed && "border-red-200 bg-red-500/[0.05]",
                    )}
                  >
                    <div className="p-4">
                      <div className="flex flex-wrap items-start justify-between gap-3">
                        <div className="flex min-w-0 items-start gap-3">
                          <div
                            className={cn(
                              "mt-0.5 grid h-8 w-8 shrink-0 place-items-center rounded-full",
                              admitted &&
                                "bg-emerald-500/15 text-emerald-600 dark:text-emerald-300",
                              !admitted &&
                                !failedClosed &&
                                "bg-amber-500/15 text-amber-700 dark:text-amber-300",
                              failedClosed &&
                                "bg-red-500/15 text-red-700 dark:text-red-300",
                            )}
                          >
                            {admitted ? (
                              <CheckCircle2 className="h-4 w-4" />
                            ) : failedClosed ? (
                              <XCircle className="h-4 w-4" />
                            ) : (
                              <Activity className="h-4 w-4" />
                            )}
                          </div>
                          <div className="min-w-0">
                            <div className="flex flex-wrap items-center gap-2">
                              <span className="text-sm font-semibold">
                                {t("因果回放 {index}", {
                                  index: formatNumber(index + 1),
                                })}
                              </span>
                              <Badge
                                variant={
                                  admitted
                                    ? "success"
                                    : failedClosed
                                      ? "warning"
                                      : "outline"
                                }
                              >
                                {causalReplayDecisionLabel(replay.decision)}
                              </Badge>
                            </div>
                            <div className="mt-1 font-mono text-[10px] text-muted-foreground">
                              #{replay.eventId ?? "—"} ·{" "}
                              {timeOnly(replay.timestamp)} · {replay.eventType}
                            </div>
                          </div>
                        </div>
                        <Badge variant={admitted ? "success" : "outline"}>
                          {admitted ? t("已安排下一轮") : t("未安排下一轮")}
                        </Badge>
                      </div>

                      <p className="mt-3 text-xs leading-5 text-muted-foreground">
                        {causalReplayExplanation(replay)}
                      </p>

                      <div className="mt-4 grid grid-cols-2 gap-3 border-t pt-3 sm:grid-cols-4">
                        <div>
                          <div className="eyebrow">{t("胜出增益")}</div>
                          <div className="mt-1 font-mono text-xs font-semibold">
                            {formatScore(replay.winnerGain)}
                          </div>
                        </div>
                        <div>
                          <div className="eyebrow">{t("观测 KL")}</div>
                          <div className="mt-1 font-mono text-xs font-semibold">
                            {formatScore(replay.winnerKl)}
                          </div>
                        </div>
                        <div>
                          <div className="eyebrow">{t("Maybe 门禁")}</div>
                          <div className="mt-1 text-xs font-semibold">
                            {maybeDecisionLabel(replay.maybeDecision)}
                          </div>
                        </div>
                        <div>
                          <div className="eyebrow">{t("回放耗时")}</div>
                          <div className="mt-1 font-mono text-xs font-semibold">
                            {replay.latencySeconds === null
                              ? "—"
                              : formatDuration(replay.latencySeconds)}
                          </div>
                        </div>
                      </div>

                      {replay.winnerCandidateId || replay.winnerDocumentId ? (
                        <div className="mt-3 rounded-md bg-muted/50 px-3 py-2 font-mono text-[10px] text-muted-foreground">
                          {t("胜出候选")}{" "}
                          {shortHash(replay.winnerCandidateId, 16)} ·{" "}
                          {t("文档")} {shortHash(replay.winnerDocumentId, 16)}
                        </div>
                      ) : null}
                    </div>

                    {replay.losses.length ? (
                      <div className="overflow-x-auto border-t">
                        <div className="min-w-[620px]">
                          <div className="grid grid-cols-[minmax(150px,1.5fr)_repeat(4,minmax(80px,1fr))] gap-3 bg-muted/45 px-3 py-2 text-[9px] font-semibold uppercase tracking-[0.12em] text-muted-foreground">
                            <span>{t("分支")}</span>
                            <span>NLL</span>
                            <span>{t("相对基线增益")}</span>
                            <span>KL</span>
                            <span>{t("观测 token")}</span>
                          </div>
                          {replay.losses.map((loss, lossIndex) => (
                            <div
                              key={`${loss.candidateId ?? "baseline"}-${lossIndex}`}
                              className="grid grid-cols-[minmax(150px,1.5fr)_repeat(4,minmax(80px,1fr))] gap-3 border-t px-3 py-2.5 text-[11px]"
                            >
                              <span className="min-w-0 truncate font-medium">
                                {loss.candidateId
                                  ? `${t("候选")} ${shortHash(loss.candidateId, 12)}`
                                  : t("基线")}
                              </span>
                              <span className="font-mono">
                                {formatScore(loss.nll)}
                              </span>
                              <span className="font-mono">
                                {formatScore(loss.gainVsBaseline)}
                              </span>
                              <span className="font-mono">
                                {formatScore(loss.observedTokenKl)}
                              </span>
                              <span className="font-mono">
                                {loss.observationTokens === null
                                  ? "—"
                                  : formatNumber(loss.observationTokens)}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : null}
                  </div>
                );
              })}

              {evidence.maybe.length ? (
                <div className="overflow-hidden rounded-lg border">
                  <div className="border-b bg-muted/45 px-3 py-2 text-[10px] font-semibold text-muted-foreground">
                    {t("Maybe 终态")}
                  </div>
                  {evidence.maybe.map((item, index) => (
                    <div
                      key={`${item.eventId ?? "maybe"}-${index}`}
                      className="flex flex-wrap items-center justify-between gap-3 border-b px-3 py-3 last:border-b-0"
                    >
                      <div>
                        <div className="text-xs font-semibold">
                          {selfAskStatusLabel(item.status)}
                        </div>
                        <div className="mt-1 font-mono text-[10px] text-muted-foreground">
                          #{item.eventId ?? "—"} · {timeOnly(item.timestamp)}
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge variant="outline">
                          {maybeDecisionLabel(item.decision)}
                        </Badge>
                        <Badge
                          variant={
                            item.scheduledNextTurn ? "success" : "outline"
                          }
                        >
                          {item.scheduledNextTurn
                            ? t("已安排下一轮")
                            : t("未安排下一轮")}
                        </Badge>
                      </div>
                    </div>
                  ))}
                </div>
              ) : null}
            </div>
          ) : (
            <div className="rounded-lg border border-dashed p-10 text-center">
              <Activity className="mx-auto h-5 w-5 text-muted-foreground" />
              <p className="mt-3 text-sm font-medium">
                {t("本次请求没有 Causal Replay 终态记录")}
              </p>
              <p className="mt-1 text-xs leading-5 text-muted-foreground">
                {observerTriggered
                  ? t(
                      "Observer 已触发，但后续没有进入回放终态；触发不等于执行了 Causal Replay。",
                    )
                  : t("没有持续不确定性触发，或语义候选未满足回放前置条件。")}
              </p>
            </div>
          )}
        </TabsContent>

        <TabsContent value="integrity">
          {integrityVerdict ? (
            <div className="space-y-3">
              <div
                className={cn("rounded-lg border p-4", integrityVerdict.tone)}
              >
                <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-start">
                  <div className="flex min-w-0 items-start gap-3">
                    <ContextIntegrityIcon
                      className={cn(
                        "mt-0.5 h-5 w-5 shrink-0",
                        integrityVerdict.icon === LoaderCircle &&
                          "animate-spin",
                      )}
                    />
                    <div>
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="text-sm font-semibold">
                          {integrityVerdict.title}
                        </span>
                        <Badge variant={integrityVerdict.badgeVariant}>
                          {integrityVerdict.label}
                        </Badge>
                      </div>
                      <p className="mt-1 text-xs leading-5 text-muted-foreground">
                        {integrityVerdict.description}
                      </p>
                    </div>
                  </div>
                </div>
                <div className="mt-4 grid gap-3 border-t pt-3 sm:grid-cols-3">
                  <div>
                    <div className="eyebrow">{t("实际动作")}</div>
                    <div className="mt-1 text-xs font-medium">
                      {integrityVerdict.action}
                    </div>
                  </div>
                  <div>
                    <div className="eyebrow">{t("检查结论")}</div>
                    <div className="mt-1 text-xs font-medium">
                      {contextIntegrityStatusLabel(integrityVerdict.status)}
                    </div>
                  </div>
                  <div>
                    <div className="eyebrow">{t("置信度 / 失效声明")}</div>
                    <div className="mt-1 text-xs font-medium">
                      {integrityVerdict.confidence === null
                        ? t("未记录")
                        : `${formatNumber(Math.round(integrityVerdict.confidence * 100))}%`}
                      {` / ${formatNumber(integrityVerdict.invalidClaimCount)}`}
                    </div>
                  </div>
                </div>
              </div>
              <div className="overflow-hidden rounded-lg border">
                <div className="border-b bg-muted/40 px-3 py-2 text-[10px] font-semibold text-muted-foreground">
                  {t("审查事件")}
                </div>
                {contextIntegrityEvents.map((event, index) => (
                  <EventRow
                    key={`${event.event_id ?? index}-${eventType(event)}`}
                    event={event}
                  />
                ))}
              </div>
            </div>
          ) : (
            <div className="rounded-lg border border-dashed p-10 text-center">
              <CheckCircle2 className="mx-auto h-5 w-5 text-muted-foreground" />
              <p className="mt-3 text-sm font-medium">
                {t("本次请求没有完整性审查")}
              </p>
              <p className="mt-1 text-xs text-muted-foreground">
                {t("没有新的工具结果，或完整性检查未启用。")}
              </p>
            </div>
          )}
        </TabsContent>

        <TabsContent value="events">
          <div className="overflow-hidden rounded-lg border">
            <div className="grid grid-cols-[54px_minmax(150px,240px)_minmax(0,1fr)] gap-3 border-b bg-muted/45 px-3 py-2 text-[9px] font-semibold uppercase tracking-[0.13em] text-muted-foreground">
              <span>ID</span>
              <span>{t("阶段")}</span>
              <span>{t("摘要 · 点击展开 JSON")}</span>
            </div>
            {evidence.events.map((event, index) => (
              <EventRow
                key={`${event.event_id ?? index}-${eventType(event)}`}
                event={event}
              />
            ))}
          </div>
        </TabsContent>
      </Tabs>

      <Dialog
        open={Boolean(documentCandidate)}
        onOpenChange={(open) => {
          if (!open) setDocumentCandidate(null);
        }}
      >
        <DialogContent className="max-w-4xl">
          <DialogHeader>
            <DialogTitle className="break-all font-mono text-sm">
              {documentCandidate?.path === UNKNOWN_DOCUMENT_SOURCE
                ? t(UNKNOWN_DOCUMENT_SOURCE)
                : documentCandidate?.path}
            </DialogTitle>
            <DialogDescription>
              {t(
                "完整文档原文。QK 命中页与源 token 范围显示在候选卡片中；此处不把文档开头冒充精确命中片段。",
              )}
            </DialogDescription>
          </DialogHeader>
          {documentLoading ? (
            <div className="grid h-64 place-items-center">
              <LoaderCircle className="animate-spin text-muted-foreground" />
            </div>
          ) : (
            <pre className="max-h-[65vh] overflow-auto whitespace-pre-wrap rounded-md border bg-muted/35 p-4 font-mono text-xs leading-6">
              {documentContent || t("文档内容不可用。")}
            </pre>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}

export function TracePage() {
  const [listing, setListing] = useState<RequestTraceListing | null>(null);
  const [loading, setLoading] = useState(true);
  const [unavailable, setUnavailable] = useState(false);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<TraceFilter>("activity");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<{
    requestId: string;
    evidence: TraceEvidence;
  } | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [updatedAt, setUpdatedAt] = useState<number | null>(null);
  const [clearAllOpen, setClearAllOpen] = useState(false);
  const [clearingAll, setClearingAll] = useState(false);
  const detailSequence = useRef(0);

  useEffect(() => {
    let cancelled = false;
    const loadDefaultScope = async () => {
      try {
        const config = await getServiceConfig();
        const value = String(
          config.values.qwen_exo_console_trace_default_scope || "activity",
        );
        if (!cancelled && isTraceDefaultScope(value)) setFilter(value);
      } catch {
        // 保留页面默认筛选；配置不可读不应阻断轨迹查看。
      }
    };
    void loadDefaultScope();
    return () => {
      cancelled = true;
    };
  }, []);

  const loadListing = useCallback(async (silent = false) => {
    if (!silent) setLoading(true);
    try {
      setListing(await getRequestTraces(100));
      setUnavailable(false);
      setUpdatedAt(Date.now());
    } catch (error) {
      if (error instanceof ApiError && error.status === 404) {
        setUnavailable(true);
      } else if (!silent) {
        toast.error(t("请求轨迹加载失败"), {
          description: error instanceof Error ? error.message : t("未知错误"),
        });
      }
    } finally {
      if (!silent) setLoading(false);
    }
  }, []);

  const loadDetail = useCallback(
    async (trace: RequestTrace, silent = false) => {
      const sequence = ++detailSequence.current;
      if (!silent) setDetailLoading(true);
      try {
        const payload = await getTelemetry(1000, trace.request_id);
        if (sequence !== detailSequence.current) return;
        setDetail({
          requestId: trace.request_id,
          evidence: buildEvidence(trace, payload.events || []),
        });
      } catch (error) {
        if (!silent) {
          toast.error(t("请求证据加载失败"), {
            description: error instanceof Error ? error.message : t("未知错误"),
          });
        }
      } finally {
        if (sequence === detailSequence.current && !silent) {
          setDetailLoading(false);
        }
      }
    },
    [],
  );

  useEffect(() => {
    void loadListing();
  }, [loadListing]);

  const requests = listing?.requests ?? [];
  const filteredRequests = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    return requests.filter((trace) => {
      if (
        normalizedQuery &&
        !trace.request_id.toLowerCase().includes(normalizedQuery) &&
        !trace.input_text.toLowerCase().includes(normalizedQuery) &&
        !trace.output_text.toLowerCase().includes(normalizedQuery) &&
        !trace.candidates.some((candidate) =>
          candidate.relative_path.toLowerCase().includes(normalizedQuery),
        )
      ) {
        return false;
      }
      if (filter === "activity") {
        return (
          hasActualKnowledge(trace) ||
          trace.candidates.length > 0 ||
          trace.self_ask.length > 0 ||
          causalReplayEventCount(trace) > 0 ||
          contextIntegrityEventCount(trace) > 0
        );
      }
      if (filter === "integrity") {
        return contextIntegrityEventCount(trace) > 0;
      }
      if (filter === "actual") return hasActualKnowledge(trace);
      if (filter === "proposed") return trace.candidates.length > 0;
      if (filter === "running") return trace.duration_seconds === null;
      return true;
    });
  }, [filter, query, requests]);

  useEffect(() => {
    if (!filteredRequests.length) {
      setSelectedId(null);
      return;
    }
    if (
      !selectedId ||
      !filteredRequests.some((trace) => trace.request_id === selectedId)
    ) {
      setSelectedId(filteredRequests[0].request_id);
    }
  }, [filteredRequests, selectedId]);

  const selectedTrace = useMemo(
    () => requests.find((trace) => trace.request_id === selectedId) || null,
    [requests, selectedId],
  );

  useEffect(() => {
    if (selectedTrace && detail?.requestId !== selectedTrace.request_id) {
      void loadDetail(selectedTrace);
    }
  }, [detail?.requestId, loadDetail, selectedTrace]);

  const refresh = async () => {
    await loadListing();
    if (selectedTrace) await loadDetail(selectedTrace);
  };

  const clearAll = async () => {
    setClearingAll(true);
    try {
      await clearTelemetry();
      setClearAllOpen(false);
      setListing({ requests: [], total_requests: 0 });
      setSelectedId(null);
      setDetail(null);
      toast.success(t("遥测已清理"));
    } catch (error) {
      toast.error(t("遥测清理失败"), {
        description: error instanceof Error ? error.message : t("未知错误"),
      });
    } finally {
      setClearingAll(false);
    }
  };

  return (
    <div className="page-frame">
      <PageHeader
        eyebrow={t("可观测性 / 证据")}
        title={t("召回轨迹")}
        description={t(
          "先看实际注入结论，再下钻候选、Judge、Self-Ask 与原始事件。候选不再冒充已召回文档。",
        )}
        actions={
          <>
            <Button
              variant="outline"
              size="sm"
              onClick={() => void refresh()}
              disabled={loading}
            >
              <RefreshCw className={loading ? "animate-spin" : ""} />
              {t("刷新")}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="text-destructive hover:bg-destructive/10 hover:text-destructive"
              onClick={() => setClearAllOpen(true)}
            >
              <Eraser />
              {t("清理")}
            </Button>
          </>
        }
      />

      {unavailable ? (
        <EmptyState
          icon={Activity}
          title={t("请求轨迹端点不可用")}
          description={t("当前服务没有暴露请求级轨迹接口。")}
        />
      ) : loading && !listing ? (
        <EmptyState
          icon={LoaderCircle}
          title={t("正在归并请求证据")}
          description={t("正在加载请求、召回候选和遥测事件。")}
        />
      ) : !requests.length && !query ? (
        <EmptyState
          icon={Activity}
          title={t("暂无请求轨迹")}
          description={t("新的 Responses 请求开始后，会在这里出现。")}
        />
      ) : (
        <>
          <div className="mb-4 flex flex-col gap-3 rounded-xl border bg-card p-3 shadow-sm lg:flex-row lg:items-center">
            <div className="relative min-w-0 flex-1">
              <Search className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder={t("搜索请求、响应或文档路径…")}
                className="pl-9"
              />
            </div>
            <div className="flex flex-wrap items-center gap-1 rounded-lg bg-muted p-1">
              {FILTERS.map((item) => (
                <button
                  type="button"
                  key={item.value}
                  onClick={() => setFilter(item.value)}
                  className={cn(
                    "rounded-md px-3 py-1.5 text-xs font-medium text-muted-foreground transition-colors",
                    filter === item.value &&
                      "bg-background text-foreground shadow-sm",
                  )}
                >
                  {t(item.label)}
                </button>
              ))}
            </div>
            <div className="flex shrink-0 items-center gap-3 px-1 text-[10px] text-muted-foreground">
              <span className="flex items-center gap-1.5">
                <RefreshCw className="h-3 w-3" />
                {t("仅手动刷新")}
              </span>
              <span>
                {t("{count} 请求", {
                  count: formatNumber(listing?.total_requests),
                })}
                {updatedAt ? ` · ${timeOnly(updatedAt)}` : ""}
              </span>
            </div>
          </div>

          <div className="grid items-start gap-4 lg:grid-cols-[320px_minmax(0,1fr)]">
            <aside className="overflow-hidden rounded-xl border bg-card shadow-sm lg:sticky lg:top-4">
              <div className="flex items-center justify-between border-b px-4 py-3">
                <div>
                  <div className="text-sm font-semibold">{t("请求索引")}</div>
                  <div className="mt-0.5 text-[10px] text-muted-foreground">
                    {t("{visible} / {total} 条", {
                      visible: formatNumber(filteredRequests.length),
                      total: formatNumber(requests.length),
                    })}
                  </div>
                </div>
                <Hash className="h-4 w-4 text-muted-foreground" />
              </div>
              <ScrollArea className="h-[calc(100vh-285px)] min-h-[520px]">
                {filteredRequests.length ? (
                  filteredRequests.map((trace) => (
                    <RequestRow
                      key={trace.request_id}
                      trace={trace}
                      selected={trace.request_id === selectedId}
                      onSelect={() => setSelectedId(trace.request_id)}
                    />
                  ))
                ) : (
                  <div className="px-5 py-12 text-center">
                    <FileSearch className="mx-auto h-5 w-5 text-muted-foreground" />
                    <p className="mt-3 text-sm font-medium">
                      {t("没有匹配请求")}
                    </p>
                    <p className="mt-1 text-xs text-muted-foreground">
                      {t("清除搜索或切换筛选条件。")}
                    </p>
                  </div>
                )}
              </ScrollArea>
            </aside>

            {selectedTrace ? (
              <EvidencePanel
                trace={selectedTrace}
                evidence={
                  detail?.requestId === selectedTrace.request_id
                    ? detail.evidence
                    : null
                }
                loading={
                  detailLoading ||
                  detail?.requestId !== selectedTrace.request_id
                }
              />
            ) : (
              <div className="grid min-h-[560px] place-items-center rounded-xl border border-dashed">
                <div className="text-center">
                  <Activity className="mx-auto h-5 w-5 text-muted-foreground" />
                  <p className="mt-3 text-sm text-muted-foreground">
                    {t("选择一个请求查看证据。")}
                  </p>
                </div>
              </div>
            )}
          </div>
        </>
      )}

      <Dialog
        open={clearAllOpen}
        onOpenChange={(open) => {
          if (!clearingAll) setClearAllOpen(open);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t("清空全部遥测？")}</DialogTitle>
            <DialogDescription>
              {t(
                "将删除全部请求事件与召回轨迹历史。不会重启服务，但此操作无法撤销。",
              )}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setClearAllOpen(false)}
              disabled={clearingAll}
            >
              {t("取消")}
            </Button>
            <Button
              variant="destructive"
              onClick={() => void clearAll()}
              disabled={clearingAll}
            >
              {clearingAll ? <LoaderCircle className="animate-spin" /> : null}
              {t("确认清空")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
