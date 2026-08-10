import { useCallback, useEffect, useState } from "react";
import {
  Activity,
  BookOpen,
  BrainCircuit,
  Layers3,
  MessageSquareText,
  RefreshCw,
  ShieldCheck,
} from "lucide-react";
import { PageHeader } from "@/components/page-header";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { getRequestTraces, listEditors, listTrajectories } from "@/lib/api";
import { runtimeStateSource, translate as t } from "@/lib/i18n";
import type { ActiveEditor, RequestTrace, RuntimeStatus } from "@/lib/types";
import {
  formatDuration,
  formatNumber,
  formatTime,
  shortHash,
} from "@/lib/utils";

type CapabilityState = "active" | "partial" | "off" | "loading";

function booleanState(value: unknown): CapabilityState {
  if (typeof value !== "boolean") return "loading";
  return value ? "active" : "off";
}

function modeState(value: string | null): CapabilityState {
  if (value === null) return "loading";
  return value === "off" ? "off" : "active";
}

function combineStates(...states: CapabilityState[]): CapabilityState {
  if (states.some((state) => state === "loading")) return "loading";
  if (states.every((state) => state === "active")) return "active";
  if (states.every((state) => state === "off")) return "off";
  return "partial";
}

function runtimeMode(status: RuntimeStatus | null, key: string) {
  const value = status?.[key];
  return typeof value === "string" ? value : null;
}

function observerModeSource(value: string | null) {
  if (value === "active") return "主动干预";
  if (value === "shadow") return "仅观测";
  if (value === "off") return "关闭";
  return value || "读取中";
}

function scoreBiasModeSource(value: string | null) {
  if (value === "trajectory_active") return "Score Bias 施加偏置";
  if (value === "trajectory_shadow") return "仅打分";
  if (value === "off") return "关闭";
  return value || "读取中";
}

function qkPresetSource(value: string | null) {
  if (value === "broad") return "高召回";
  if (value === "balanced") return "标准";
  if (value === "strict") return "高精度";
  return value || "读取中";
}

function memoryLaneSource(value: string) {
  if (value === "knowledge") return "知识";
  if (value === "policydata") return "人格";
  if (value === "cognition") return "认知";
  return value;
}

function hasEvent(trace: RequestTrace | null, event: string) {
  return Boolean(trace?.event_types.includes(event));
}

function hasEventPrefix(trace: RequestTrace | null, prefix: string) {
  return Boolean(trace?.event_types.some((event) => event.startsWith(prefix)));
}

export function OverviewPage({
  status,
  onRefresh,
}: {
  status: RuntimeStatus | null;
  onRefresh: () => Promise<void>;
}) {
  const [refreshing, setRefreshing] = useState(false);
  const [latestTrace, setLatestTrace] = useState<RequestTrace | null>(null);
  const [trajectoryCount, setTrajectoryCount] = useState<number | null>(null);
  const [activeEditor, setActiveEditor] = useState<ActiveEditor | null>(null);
  const [overviewDataLoaded, setOverviewDataLoaded] = useState(false);
  const [overviewDataError, setOverviewDataError] = useState(false);

  const loadOverviewData = useCallback(async () => {
    const results = await Promise.allSettled([
      getRequestTraces(1),
      listTrajectories(),
      listEditors(),
    ]);
    const [traceResult, trajectoryResult, editorResult] = results;
    if (traceResult.status === "fulfilled") {
      setLatestTrace(traceResult.value.requests[0] || null);
    }
    if (trajectoryResult.status === "fulfilled") {
      setTrajectoryCount(trajectoryResult.value.trajectories.length);
    }
    if (editorResult.status === "fulfilled") {
      setActiveEditor(editorResult.value.active);
    }
    setOverviewDataLoaded(true);
    setOverviewDataError(
      results.some((result) => result.status === "rejected"),
    );
  }, []);

  useEffect(() => {
    void loadOverviewData();
    const timer = window.setInterval(() => void loadOverviewData(), 5000);
    return () => window.clearInterval(timer);
  }, [loadOverviewData]);

  const ready = status?.runtime_state === "ready";
  const observerMode =
    runtimeMode(status, "observer_mode") || status?.observer_mode || null;
  const scoreBiasMode = runtimeMode(status, "score_bias_mode");
  const contextEvidenceMode = runtimeMode(status, "context_evidence_mode");
  const contextIntegrityMode = runtimeMode(status, "context_integrity_mode");
  const reflectionMode = runtimeMode(status, "reflection_memory_mode");
  const compactionMode = runtimeMode(status, "response_compaction_mode");
  const qkPreset = runtimeMode(status, "qk_recall_preset");
  const storageDtype =
    typeof status?.tensor_bank?.storage_dtype === "string"
      ? status.tensor_bank.storage_dtype
      : "—";

  const observerState = combineStates(
    modeState(observerMode),
    booleanState(status?.features?.adaptive_refresh),
  );
  const toolEvidenceState = combineStates(
    modeState(contextEvidenceMode),
    modeState(contextIntegrityMode),
  );
  const capabilities: Array<{
    label: string;
    detail: string;
    state: CapabilityState;
  }> = [
    {
      label: t("知识记忆"),
      detail: t("{count} 篇 · Native Tensor Bank", {
        count: formatNumber(status?.knowledge?.document_count),
      }),
      state: booleanState(status?.features?.external_memory),
    },
    {
      label: t("人格记忆"),
      detail: t("{count} 篇 · 原生人格前缀", {
        count: formatNumber(status?.policy_data?.document_count),
      }),
      state: booleanState(status?.features?.policy_data),
    },
    {
      label: t("混合状态"),
      detail: "Full-Attention KV + GDN",
      state: booleanState(status?.features?.hybrid_prefix),
    },
    {
      label: t("语义准入"),
      detail: `Reference Judge · ${t(qkPresetSource(qkPreset))}`,
      state: booleanState(status?.features?.reference_judge),
    },
    {
      label: t("解码观测"),
      detail: `${t(observerModeSource(observerMode))} · ${t("自适应刷新")}`,
      state: observerState,
    },
    {
      label: t("轨迹学习"),
      detail: `${t(scoreBiasModeSource(scoreBiasMode))} · ${
        activeEditor
          ? `${activeEditor.editor} ${formatNumber(activeEditor.strength || 1)}×`
          : overviewDataLoaded
            ? t("激活编辑器未加载")
            : t("读取编辑器")
      }`,
      state: modeState(scoreBiasMode),
    },
    {
      label: t("工具后证据"),
      detail: t("临时证据 · 完整性检查"),
      state: toolEvidenceState,
    },
    {
      label: t("反思记忆"),
      detail: t("空闲轨迹提炼并热更新"),
      state: modeState(reflectionMode),
    },
    {
      label: t("上下文压缩"),
      detail: t("摘要 + 原生状态复用"),
      state: modeState(compactionMode),
    },
    {
      label: t("执行胶囊"),
      detail: t("跨轮执行状态连续性"),
      state: booleanState(status?.features?.capsule),
    },
  ];

  const enabledCapabilities = capabilities.filter(
    (capability) => capability.state === "active",
  ).length;
  const metrics = [
    {
      label: t("运行状态"),
      value: ready
        ? "READY"
        : t(runtimeStateSource(status?.runtime_state, "连接中")),
      detail: ready ? t("调度器与 Native Bank 可用") : t("等待运行时屏障"),
      icon: Activity,
    },
    {
      label: t("模型拓扑"),
      value: `TP ${formatNumber(status?.tp_size ?? status?.hybrid_state?.tp_size)}`,
      detail: t("{dtype} 状态 · {storage} Bank", {
        dtype: status?.hybrid_state?.dtype || "—",
        storage: storageDtype,
      }),
      icon: Layers3,
    },
    {
      label: t("知识文档"),
      value: formatNumber(status?.knowledge?.document_count),
      detail: t("知识 {count} · Q×K 候选库", {
        count: formatNumber(status?.knowledge?.document_count || 0),
      }),
      icon: BookOpen,
    },
    {
      label: t("人格文档"),
      value: formatNumber(status?.policy_data?.document_count),
      detail: status?.policy_data?.enabled
        ? t("原生人格状态已启用")
        : t("当前未启用"),
      icon: ShieldCheck,
    },
    {
      label: t("轨迹记忆"),
      value: formatNumber(trajectoryCount),
      detail: scoreBiasMode
        ? t(scoreBiasModeSource(scoreBiasMode))
        : t("读取中"),
      icon: BrainCircuit,
    },
  ];

  const scoreBiasApplied =
    hasEvent(latestTrace, "score_bias.applied") ||
    hasEvent(latestTrace, "score_bias.decode_selected");
  const scoreBiasAbstained = hasEvent(
    latestTrace,
    "score_bias.decode_abstained",
  );
  const observerActive = hasEvent(latestTrace, "observer.decode_summary");
  const pathStages: Array<{
    label: string;
    detail: string;
    status: string;
    variant: "success" | "warning" | "secondary";
    icon: typeof Activity;
  }> = [
    {
      label: "Responses",
      detail: latestTrace
        ? latestTrace.duration_seconds === null
          ? t("流式请求正在运行")
          : `${formatDuration(latestTrace.duration_seconds)} · ${formatNumber(latestTrace.output_tokens)} tokens`
        : t("暂无请求遥测"),
      status: latestTrace
        ? latestTrace.duration_seconds === null
          ? t("进行中")
          : t("完成")
        : t("无数据"),
      variant: latestTrace
        ? latestTrace.duration_seconds === null
          ? "warning"
          : "success"
        : "secondary",
      icon: MessageSquareText,
    },
    {
      label: "Native QK",
      detail: latestTrace?.candidates.length
        ? t("{count} 个 Tensor Bank 候选", {
            count: formatNumber(latestTrace.candidates.length),
          })
        : hasEvent(latestTrace, "tensor.candidates_proposed")
          ? t("检索完成，无候选")
          : t("本轮未触发"),
      status: latestTrace?.candidates.length ? t("已召回") : t("未命中"),
      variant: latestTrace?.candidates.length ? "success" : "secondary",
      icon: BrainCircuit,
    },
    {
      label: "Semantic Gate",
      detail: hasEvent(latestTrace, "semantic_judge.completed")
        ? t("Reference Judge 已完成")
        : hasEvent(latestTrace, "qk.prefilter")
          ? t("QK 预过滤已完成")
          : t("本轮未触发"),
      status: hasEvent(latestTrace, "semantic_judge.completed")
        ? t("已审计")
        : t("未触发"),
      variant: hasEvent(latestTrace, "semantic_judge.completed")
        ? "success"
        : "secondary",
      icon: ShieldCheck,
    },
    {
      label: "Full KV + GDN",
      detail: latestTrace?.native_restore
        ? `${t(memoryLaneSource(latestTrace.native_restore.lane))} · ${formatNumber(latestTrace.native_restore.tokens)} tokens`
        : hasEvent(latestTrace, "memory.prepared")
          ? t("原生状态已准备")
          : t("本轮未恢复"),
      status: latestTrace?.native_restore ? t("已恢复") : t("未恢复"),
      variant: latestTrace?.native_restore ? "success" : "secondary",
      icon: Layers3,
    },
    {
      label: "Observer / Score Bias",
      detail: scoreBiasApplied
        ? t("Observer 活跃 · 偏置已施加")
        : scoreBiasAbstained
          ? t("Observer 活跃 · 本轮偏置未命中")
          : hasEventPrefix(latestTrace, "score_bias.")
            ? t("轨迹候选已准备")
            : t("本轮未触发"),
      status: observerActive ? t("已观测") : t("未观测"),
      variant: observerActive ? "success" : "secondary",
      icon: Activity,
    },
  ];

  const refresh = async () => {
    setRefreshing(true);
    try {
      await Promise.all([onRefresh(), loadOverviewData()]);
    } finally {
      setRefreshing(false);
    }
  };

  return (
    <div className="page-frame">
      <PageHeader
        title={t("总览")}
        description={t("当前运行状态、增强功能与最近一次请求的数据路径。")}
        actions={
          <Button
            variant="outline"
            size="sm"
            onClick={() => void refresh()}
            disabled={refreshing}
          >
            <RefreshCw className={refreshing ? "animate-spin" : ""} />
            {t("刷新")}
          </Button>
        }
      />

      <section
        className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5"
        aria-label={t("运行与记忆数量")}
      >
        {status
          ? metrics.map((metric) => {
              const Icon = metric.icon;
              return (
                <Card key={metric.label} className="shadow-sm">
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-medium text-muted-foreground">
                        {metric.label}
                      </span>
                      <Icon className="h-4 w-4 text-muted-foreground" />
                    </div>
                    <div className="metric-value">{metric.value}</div>
                    <p className="mt-2 text-xs text-muted-foreground">
                      {metric.detail}
                    </p>
                  </CardContent>
                </Card>
              );
            })
          : Array.from({ length: 5 }, (_, index) => (
              <Skeleton key={index} className="h-28" />
            ))}
      </section>

      <Card className="mt-5">
        <CardHeader className="border-b">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <CardTitle>{t("增强功能")}</CardTitle>
              <CardDescription>{t("后端当前实际生效状态。")}</CardDescription>
            </div>
            <Badge variant={ready ? "success" : "warning"}>
              {t("{enabled}/{total} 开启", {
                enabled: formatNumber(enabledCapabilities),
                total: formatNumber(capabilities.length),
              })}
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="grid gap-3 p-4 sm:grid-cols-2 xl:grid-cols-5">
          {capabilities.map((capability) => {
            const badge =
              capability.state === "active"
                ? { label: t("开启"), variant: "success" as const }
                : capability.state === "partial"
                  ? { label: t("部分开启"), variant: "warning" as const }
                  : capability.state === "off"
                    ? { label: t("关闭"), variant: "secondary" as const }
                    : { label: t("读取中"), variant: "outline" as const };
            return (
              <div key={capability.label} className="rounded-md border p-3">
                <div className="flex items-start justify-between gap-2">
                  <div className="text-xs font-semibold">
                    {capability.label}
                  </div>
                  <Badge variant={badge.variant}>{badge.label}</Badge>
                </div>
                <div className="mt-2 text-[11px] leading-5 text-muted-foreground">
                  {capability.detail}
                </div>
              </div>
            );
          })}
        </CardContent>
      </Card>

      <Card className="mt-5">
        <CardHeader className="border-b">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <CardTitle>{t("最近请求数据路径")}</CardTitle>
              <CardDescription>
                {latestTrace
                  ? `${formatTime(latestTrace.started_at)} · ${shortHash(latestTrace.request_id, 22)} · ${latestTrace.duration_seconds === null ? t("进行中") : formatDuration(latestTrace.duration_seconds)}`
                  : overviewDataLoaded
                    ? t("暂无请求遥测")
                    : t("正在读取最近请求")}
              </CardDescription>
            </div>
            {overviewDataError ? (
              <Badge variant="warning">{t("部分数据不可用")}</Badge>
            ) : latestTrace ? (
              <Badge
                variant={
                  latestTrace.duration_seconds === null ? "warning" : "success"
                }
              >
                {latestTrace.duration_seconds === null
                  ? t("运行中")
                  : t("已完成")}
              </Badge>
            ) : null}
          </div>
        </CardHeader>
        <CardContent className="p-0">
          <div className="grid divide-y lg:grid-cols-5 lg:divide-x lg:divide-y-0">
            {pathStages.map((stage, index) => {
              const Icon = stage.icon;
              return (
                <div key={stage.label} className="p-4">
                  <div className="flex items-center justify-between gap-3">
                    <div className="grid h-8 w-8 place-items-center rounded-md border bg-muted">
                      <Icon className="h-4 w-4" />
                    </div>
                    <span className="font-mono text-[10px] text-muted-foreground">
                      0{index + 1}
                    </span>
                  </div>
                  <div className="mt-4 flex items-center justify-between gap-2">
                    <div className="text-xs font-semibold">{stage.label}</div>
                    <Badge variant={stage.variant}>{stage.status}</Badge>
                  </div>
                  <div className="mt-2 text-[11px] leading-5 text-muted-foreground">
                    {stage.detail}
                  </div>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
        <span>
          {t("模型 {fingerprint}", {
            fingerprint: shortHash(
              String(status?.model?.fingerprint || ""),
              14,
            ),
          })}
        </span>
        <span>
          {t("遥测 {count} 条", {
            count: formatNumber(status?.telemetry?.event_count),
          })}
        </span>
        <span>
          {t("页大小 {size}", {
            size: formatNumber(status?.hybrid_state?.page_size),
          })}
        </span>
        <span>
          {t("内部任务 {state}", {
            state: status?.scheduler_native_internal_jobs
              ? t("可用")
              : t("不可用"),
          })}
        </span>
      </div>
    </div>
  );
}
