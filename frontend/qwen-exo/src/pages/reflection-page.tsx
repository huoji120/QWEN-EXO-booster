import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  BrainCircuit,
  CheckCircle2,
  CircleAlert,
  Clock3,
  Link2,
  LoaderCircle,
  Play,
  RefreshCw,
  RotateCcw,
  Search,
  X,
} from "lucide-react";
import { toast } from "sonner";
import { EmptyState } from "@/components/empty-state";
import { PageHeader } from "@/components/page-header";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import {
  ApiError,
  cancelPendingReflections,
  getReflectionEvidence,
  getReflectionMemory,
  getReflectionRegenerationStatus,
  getReflectionSource,
  listPendingReflectionMemories,
  listReflectionMemories,
  regenerateReflectionMemory,
  startPendingReflections,
} from "@/lib/api";
import { translate as t } from "@/lib/i18n";
import type {
  PendingReflectionMemory,
  ReflectionMemoryRecord,
  ReflectionMemorySummary,
  ReflectionRegenerationJobStatus,
  ReflectionSourceDetail,
} from "@/lib/types";
import { cn, formatNumber, formatTime } from "@/lib/utils";

const MEMORY_PAGE_SIZE = 25;
const REFRESH_DELAY_MS = 5000;

const REGENERATION_STEPS = [
  "读取轨迹",
  "Q×K 检索",
  "模型反思",
  "准入与发布",
  "完成",
];

const INITIAL_REGENERATION: ReflectionRegenerationJobStatus = {
  job_id: null,
  status: "idle",
  stage: "idle",
  progress: 0,
  message: "尚未开始重新反思",
  details: {},
  result: null,
  error: null,
};

const OUTCOME_LABELS: Record<ReflectionMemoryRecord["outcome"], string> = {
  success: "成功",
  failure: "失败",
  mixed: "部分完成",
  uncertain: "未确定",
};

const REGENERATION_STATUS_LABELS: Record<
  ReflectionRegenerationJobStatus["status"],
  string
> = {
  idle: "未运行",
  queued: "已排队",
  running: "后台运行中",
  succeeded: "已完成",
  failed: "失败",
};

function remainingLabel(item: PendingReflectionMemory, now: number) {
  if (item.status === "running") return t("整理中");
  if (item.status === "failed") return t("等待手动重试");
  const seconds = Math.max(0, Math.ceil(item.due_at - now));
  if (!seconds) return t("即将开始");
  const minutes = Math.floor(seconds / 60);
  const rest = seconds % 60;
  if (!minutes) return t("{count} 秒", { count: formatNumber(rest) });
  return rest
    ? t("{minutes} 分 {seconds} 秒", {
        minutes: formatNumber(minutes),
        seconds: formatNumber(rest),
      })
    : t("{count} 分", { count: formatNumber(minutes) });
}

function outcomeVariant(outcome: ReflectionMemoryRecord["outcome"]) {
  if (outcome === "success") return "success" as const;
  if (outcome === "failure") return "destructive" as const;
  if (outcome === "mixed") return "warning" as const;
  return "outline" as const;
}
function regenerationStepIndex(status: ReflectionRegenerationJobStatus) {
  if (status.status === "succeeded" || status.stage === "completed") return 4;
  if (status.stage === "publishing") return 3;
  if (["causal_review", "model_review"].includes(status.stage)) return 2;
  if (status.stage === "qk_retrieval") return 1;
  if (
    ["loading_source", "evidence_extraction", "queued"].includes(status.stage)
  )
    return 0;
  if (status.status === "failed")
    return Math.min(3, Math.max(0, Math.floor(status.progress / 20)));
  return -1;
}

function memoryTextSections(memory: ReflectionMemoryRecord) {
  return [
    ["反思", memory.reflection],
    ["证据", memory.evidence],
    ["因果分析", memory.causal_analysis],
    ["冲突与边界", memory.conflict_resolution],
    ["可复用经验", memory.reusable_experience],
    ["应避免", memory.avoid],
    ["下一次", memory.next_time],
  ] as const;
}

const CAUSAL_LABELS: Record<string, string> = {
  verified: "因果已验证",
  supported: "有证据支持（未验证）",
  unresolved: "尚未解决",
  active: "可进入 Knowledge",
  candidate: "候选",
  retired: "已退役",
  complete: "分析完整",
  partial: "分析部分完成",
  failed: "失败",
  failed_closed: "失败关闭",
  no_lesson: "未形成经验",
  pending: "待处理",
  completed: "分析完整",
  analyzed: "已分析",
  provided_events: "提供事件",
  analyzed_events: "已分析事件",
  pending_events: "待分析事件",
  failed_segments: "失败分段",
  scope: "适用范围",
  problem: "问题",
  action: "行动",
  observation: "观察事实",
  next_check: "下一步验证",
  alternatives: "竞争解释",
  counterevidence: "反证",
  missing_evidence: "缺失证据",
};

function causalLabel(value: string) {
  return t(CAUSAL_LABELS[value] || value);
}

function causalFieldLabel(key: string, status: string) {
  if (key === "mechanism") {
    return t(
      status === "verified"
        ? "已验证机制"
        : status === "supported"
          ? "机制假设（有证据支持）"
          : "待验证机制假设",
    );
  }
  if (key === "rule") {
    return t(
      status === "verified"
        ? "适用范围内的规则"
        : status === "supported"
          ? "有条件的建议"
          : "排查建议（待验证）",
    );
  }
  return causalLabel(key);
}

function RawSourceInspection({ memory }: { memory: ReflectionMemoryRecord }) {
  const [open, setOpen] = useState(false);
  const [detail, setDetail] = useState<ReflectionSourceDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (!open || !memory.source_available) return;
    const controller = new AbortController();
    setDetail(null);
    setError(null);
    getReflectionSource(memory.source_digest, controller.signal).then(
      (result) => {
        if (!controller.signal.aborted) setDetail(result);
      },
      (failure: unknown) => {
        if (!controller.signal.aborted)
          setError(failure instanceof Error ? failure.message : t("未知错误"));
      },
    );
    return () => {
      controller.abort();
    };
  }, [
    open,
    memory.source_digest,
    memory.source_snapshot_digest,
    memory.source_available,
  ]);
  return (
    <details
      className="border-t pt-4"
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary className="cursor-pointer text-sm font-medium">
        {t("原始来源检查（非经验正文）")}
      </summary>
      {!memory.source_available ? (
        <p className="mt-2 text-sm text-muted-foreground">
          {t("历史轨迹未保留")}
        </p>
      ) : open ? (
        <div className="mt-3">
          <p className="mb-2 text-xs text-muted-foreground">
            {t("仅展示服务保存的来源；完整性以来源审计和覆盖状态为准。")}
          </p>
          {error ? (
            <p className="text-sm text-destructive">{error}</p>
          ) : detail ? (
            <pre className="overflow-auto whitespace-pre-wrap break-words bg-muted/20 p-3 text-xs leading-5">
              {JSON.stringify(detail.source, null, 2)}
            </pre>
          ) : (
            <LoaderCircle className="h-4 w-4 animate-spin" />
          )}
        </div>
      ) : null}
    </details>
  );
}

export function ReflectionPage() {
  const [memories, setMemories] = useState<ReflectionMemorySummary[]>([]);
  const [memoryTotal, setMemoryTotal] = useState(0);
  const [memoryOffset, setMemoryOffset] = useState(0);
  const [debouncedMemoryQuery, setDebouncedMemoryQuery] = useState("");
  const [pending, setPending] = useState<PendingReflectionMemory[]>([]);
  const [regeneration, setRegeneration] =
    useState<ReflectionRegenerationJobStatus>(INITIAL_REGENERATION);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [memoryQuery, setMemoryQuery] = useState("");
  const [pendingQuery, setPendingQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [available, setAvailable] = useState(true);
  const [action, setAction] = useState<"start" | "cancel" | null>(null);
  const [now, setNow] = useState(() => Date.now() / 1000);
  const [summaryConversationKey, setSummaryConversationKey] = useState<
    string | null
  >(null);
  const [memoryTarget, setMemoryTarget] =
    useState<ReflectionMemorySummary | null>(null);
  const [memoryDetailLoading, setMemoryDetailLoading] = useState(false);
  const [memoryDetailError, setMemoryDetailError] = useState<string | null>(
    null,
  );
  const [memoryDetail, setMemoryDetail] =
    useState<ReflectionMemoryRecord | null>(null);
  const [regenerateTarget, setRegenerateTarget] =
    useState<ReflectionMemorySummary | null>(null);
  const [sourceDetail, setSourceDetail] =
    useState<ReflectionSourceDetail | null>(null);
  const [evidenceEvent, setEvidenceEvent] = useState<Record<
    string,
    unknown
  > | null>(null);
  const [evidenceLoading, setEvidenceLoading] = useState(false);
  const [sourceLoading, setSourceLoading] = useState(false);
  const [verifierFeedback, setVerifierFeedback] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const notifiedJob = useRef<string | null>(null);
  const mounted = useRef(false);
  const listController = useRef<AbortController | null>(null);
  const listInFlight = useRef<Promise<void> | null>(null);
  const reload = useRef<(silent?: boolean) => Promise<void>>(async () => {});
  const detailController = useRef<AbortController | null>(null);
  const sourceController = useRef<AbortController | null>(null);
  const evidenceController = useRef<AbortController | null>(null);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      detailController.current?.abort();
      sourceController.current?.abort();
      evidenceController.current?.abort();
    };
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setDebouncedMemoryQuery(memoryQuery.trim());
    }, 300);
    return () => window.clearTimeout(timer);
  }, [memoryQuery]);

  const load = useCallback((silent = false) => reload.current(silent), []);

  useEffect(() => {
    let disposed = false;
    let revision = 0;
    let timer: number | undefined;
    const queryReady = memoryQuery.trim() === debouncedMemoryQuery;
    const invalidate = () => {
      revision += 1;
      window.clearTimeout(timer);
      listController.current?.abort();
    };
    const run = async (silent = false) => {
      invalidate();
      const currentRevision = revision;
      const current = () =>
        !disposed &&
        currentRevision === revision &&
        !document.hidden &&
        queryReady;
      if (!current()) return;
      if (!silent) setLoading(true);
      // Wait for all cancelled reads to settle before issuing a fresh snapshot.
      await listInFlight.current;
      if (!current()) return;
      const controller = new AbortController();
      listController.current = controller;
      const requests = [
        listReflectionMemories(
          {
            limit: MEMORY_PAGE_SIZE,
            offset: memoryOffset,
            q: debouncedMemoryQuery,
          },
          controller.signal,
        ),
        listPendingReflectionMemories(controller.signal),
        getReflectionRegenerationStatus(controller.signal),
      ] as const;
      const work = (async () => {
        try {
          const [memoryResult, pendingResult, regenerationResult] =
            await Promise.all(requests);
          if (!current() || controller.signal.aborted) return;
          setAvailable(true);
          setMemoryTotal(memoryResult.total);
          if (memoryOffset > 0 && memoryOffset >= memoryResult.total) {
            setMemories([]);
            setMemoryOffset(
              Math.max(
                0,
                Math.ceil(memoryResult.total / MEMORY_PAGE_SIZE) - 1,
              ) * MEMORY_PAGE_SIZE,
            );
          } else {
            setMemories(memoryResult.reflections);
          }
          setPending(pendingResult.pending);
          setRegeneration(regenerationResult);
          const availableKeys = new Set(
            pendingResult.pending.map((item) => item.conversation_key),
          );
          setSelected(
            (selection) =>
              new Set([...selection].filter((key) => availableKeys.has(key))),
          );
        } catch (error) {
          if (!current() || controller.signal.aborted) return;
          if (error instanceof ApiError && error.status === 404) {
            setAvailable(false);
            setMemories([]);
            setMemoryTotal(0);
            setPending([]);
            setSelected(new Set());
          } else if (!silent) {
            toast.error(t("反思记忆加载失败"), {
              description:
                error instanceof Error ? error.message : t("未知错误"),
            });
          }
        } finally {
          await Promise.allSettled(requests);
          if (current() && !controller.signal.aborted) {
            setLoading(false);
            timer = window.setTimeout(() => void run(true), REFRESH_DELAY_MS);
          }
        }
      })();
      listInFlight.current = work;
      await work;
      if (listInFlight.current === work) listInFlight.current = null;
    };
    reload.current = run;
    setMemories([]);
    setLoading(true);
    void run();
    const onVisibilityChange = () => {
      if (document.hidden) invalidate();
      else void run();
    };
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => {
      disposed = true;
      invalidate();
      document.removeEventListener("visibilitychange", onVisibilityChange);
    };
  }, [debouncedMemoryQuery, memoryOffset, memoryQuery]);

  useEffect(() => {
    const clockTimer = window.setInterval(() => {
      if (!document.hidden) setNow(Date.now() / 1000);
    }, 1000);
    return () => window.clearInterval(clockTimer);
  }, []);

  useEffect(() => {
    if (!regeneration.job_id || notifiedJob.current === regeneration.job_id)
      return;
    if (regeneration.status === "succeeded") {
      notifiedJob.current = regeneration.job_id;
      toast.success(t("重新反思完成"), {
        description: t("分析已完成；是否发布以记录的准入与发布状态为准。"),
      });
      void load(true);
    } else if (regeneration.status === "failed") {
      notifiedJob.current = regeneration.job_id;
      toast.error(t("重新反思失败"), {
        description: regeneration.error || regeneration.message,
      });
    }
  }, [load, regeneration]);

  const visiblePending = useMemo(() => {
    const needle = pendingQuery.trim().toLowerCase();
    if (!needle) return pending;
    return pending.filter((item) =>
      [
        item.original_task,
        item.trajectory_id,
        item.conversation_key,
        item.source_digest,
      ]
        .join(" ")
        .toLowerCase()
        .includes(needle),
    );
  }, [pending, pendingQuery]);

  const visibleKeys = visiblePending.map((item) => item.conversation_key);
  const allVisibleSelected =
    visibleKeys.length > 0 && visibleKeys.every((key) => selected.has(key));
  const someVisibleSelected = visibleKeys.some((key) => selected.has(key));
  const selectedItems = pending.filter((item) =>
    selected.has(item.conversation_key),
  );
  const selectedHasRunning = selectedItems.some(
    (item) => item.status === "running",
  );
  const summaryItem = pending.find(
    (item) => item.conversation_key === summaryConversationKey,
  );
  const regenerating =
    regeneration.status === "queued" || regeneration.status === "running";
  const activeRegenerationStep = regenerationStepIndex(regeneration);

  const toggleAll = () => {
    setSelected((current) => {
      const next = new Set(current);
      if (allVisibleSelected) visibleKeys.forEach((key) => next.delete(key));
      else visibleKeys.forEach((key) => next.add(key));
      return next;
    });
  };

  const toggleOne = (key: string) => {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  const closeEvidence = () => {
    evidenceController.current?.abort();
    setEvidenceEvent(null);
    setEvidenceLoading(false);
  };

  const closeMemory = () => {
    detailController.current?.abort();
    closeEvidence();
    setMemoryTarget(null);
    setMemoryDetail(null);
    setMemoryDetailLoading(false);
    setMemoryDetailError(null);
  };

  const openMemory = async (memory: ReflectionMemorySummary) => {
    detailController.current?.abort();
    closeEvidence();
    const controller = new AbortController();
    detailController.current = controller;
    setMemoryTarget(memory);
    setMemoryDetail(null);
    setMemoryDetailError(null);
    setMemoryDetailLoading(true);
    try {
      const result = await getReflectionMemory(
        memory.source_digest,
        controller.signal,
      );
      if (!controller.signal.aborted) setMemoryDetail(result.reflection);
    } catch (error) {
      if (!controller.signal.aborted) {
        setMemoryDetailError(
          error instanceof Error ? error.message : t("未知错误"),
        );
      }
    } finally {
      if (!controller.signal.aborted) setMemoryDetailLoading(false);
    }
  };

  const closeRegeneration = () => {
    sourceController.current?.abort();
    setRegenerateTarget(null);
    setSourceDetail(null);
    setSourceLoading(false);
    setVerifierFeedback("");
  };

  const startNow = async (keys: string[]) => {
    if (!keys.length) return;
    setAction("start");
    try {
      const result = await startPendingReflections(keys);
      if (!mounted.current) return;
      setSelected(new Set());
      toast.success(
        t("已开始 {count} 条反思", {
          count: formatNumber(result.started_count),
        }),
      );
      await load(true);
    } catch (error) {
      if (!mounted.current) return;
      toast.error(t("立即反思失败"), {
        description: error instanceof Error ? error.message : t("未知错误"),
      });
    } finally {
      if (mounted.current) setAction(null);
    }
  };

  const cancel = async (keys: string[]) => {
    if (!keys.length) return;
    setAction("cancel");
    try {
      const result = await cancelPendingReflections(keys);
      if (!mounted.current) return;
      setSelected(new Set());
      toast.success(
        t("已取消 {count} 条反思", {
          count: formatNumber(result.cancelled_count),
        }),
      );
      await load(true);
    } catch (error) {
      if (!mounted.current) return;
      toast.error(t("取消反思失败"), {
        description: error instanceof Error ? error.message : t("未知错误"),
      });
    } finally {
      if (mounted.current) setAction(null);
    }
  };

  const openRegeneration = async (memory: ReflectionMemorySummary) => {
    if (
      !memory.source_available ||
      !(memory.causal_schema || memory.document_sha256)
    )
      return;
    sourceController.current?.abort();
    const controller = new AbortController();
    sourceController.current = controller;
    setRegenerateTarget(memory);
    setSourceDetail(null);
    setVerifierFeedback("");
    setSourceLoading(true);
    try {
      const detail = await getReflectionSource(
        memory.source_digest,
        controller.signal,
      );
      if (controller.signal.aborted) return;
      setRegenerateTarget(detail.reflection);
      setSourceDetail(detail);
      setVerifierFeedback(detail.source.verifier_feedback || "");
    } catch (error) {
      if (controller.signal.aborted) return;
      toast.error(t("关联轨迹加载失败"), {
        description: error instanceof Error ? error.message : t("未知错误"),
      });
    } finally {
      if (!controller.signal.aborted) setSourceLoading(false);
    }
  };
  const inspectEvidence = async (eventId: string) => {
    evidenceController.current?.abort();
    const controller = new AbortController();
    evidenceController.current = controller;
    setEvidenceEvent(null);
    setEvidenceLoading(true);
    try {
      const result = await getReflectionEvidence(eventId, controller.signal);
      if (!controller.signal.aborted) setEvidenceEvent(result);
    } catch (error) {
      if (controller.signal.aborted) return;
      toast.error(t("证据事件加载失败"), {
        description: error instanceof Error ? error.message : t("未知错误"),
      });
    } finally {
      if (!controller.signal.aborted) setEvidenceLoading(false);
    }
  };

  const submitRegeneration = async () => {
    if (
      !regenerateTarget?.source_available ||
      !(regenerateTarget.causal_schema || regenerateTarget.document_sha256)
    )
      return;
    const feedback = verifierFeedback.trim();
    if (!feedback) {
      toast.error(t("请填写 verifier 反馈"));
      return;
    }
    setSubmitting(true);
    try {
      const status = await regenerateReflectionMemory(
        regenerateTarget.source_digest,
        feedback,
        regenerateTarget.document_sha256 || "",
      );
      if (!mounted.current) return;
      setRegeneration(status);
      closeRegeneration();
      toast.success(t("重新反思已进入后台队列"));
      await load(true);
    } catch (error) {
      if (!mounted.current) return;
      toast.error(t("重新反思启动失败"), {
        description: error instanceof Error ? error.message : t("未知错误"),
      });
    } finally {
      if (mounted.current) setSubmitting(false);
    }
  };

  return (
    <div className="page-frame">
      <PageHeader
        title={t("Reflection Memory")}
        description={t(
          "已发布经验参与普通 Knowledge 召回。不同问题可因同一适用机制通过语义审查；仅主题相似不足以通过，也不保证被选中。",
        )}
        actions={
          <Button variant="outline" size="sm" onClick={() => void load()}>
            <RefreshCw />
            {t("刷新")}
          </Button>
        }
      />

      {regeneration.status !== "idle" ? (
        <div className="mb-4 space-y-4 border bg-muted/20 p-4">
          <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-start">
            <div className="flex min-w-0 items-start gap-3">
              {regenerating ? (
                <LoaderCircle className="mt-0.5 h-5 w-5 shrink-0 animate-spin text-primary" />
              ) : regeneration.status === "succeeded" ? (
                <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-emerald-600" />
              ) : (
                <CircleAlert className="mt-0.5 h-5 w-5 shrink-0 text-destructive" />
              )}
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-sm font-medium">{t("重新反思")}</span>
                  <Badge
                    variant={
                      regeneration.status === "failed"
                        ? "destructive"
                        : "outline"
                    }
                  >
                    {t(REGENERATION_STATUS_LABELS[regeneration.status])}
                  </Badge>
                </div>
                <div className="mt-1 text-xs text-muted-foreground">
                  {t(regeneration.message)}
                </div>
                {regeneration.details?.trajectory_id ? (
                  <div className="mt-1 break-all font-mono text-[10px] text-muted-foreground">
                    {String(regeneration.details.trajectory_id)}
                  </div>
                ) : null}
                {regeneration.error ? (
                  <div className="mt-1 text-xs text-destructive">
                    {regeneration.error}
                  </div>
                ) : null}
              </div>
            </div>
            <div className="shrink-0 text-right">
              <div className="font-mono text-sm font-semibold">
                {formatNumber(
                  Math.max(0, Math.min(100, regeneration.progress)),
                )}
                %
              </div>
              <div className="mt-1 text-[10px] text-muted-foreground">
                {t("服务端后台任务")}
              </div>
            </div>
          </div>
          <div className="grid grid-cols-5 gap-2">
            {REGENERATION_STEPS.map((step, index) => (
              <div key={step}>
                <div
                  className={cn(
                    "h-1 bg-muted",
                    index <= activeRegenerationStep &&
                      (regeneration.status === "failed"
                        ? "bg-destructive"
                        : "bg-primary"),
                  )}
                />
                <div className="mt-2 hidden text-[9px] text-muted-foreground sm:block">
                  {t(step)}
                </div>
              </div>
            ))}
          </div>
          {regenerating ? (
            <div className="text-[11px] text-muted-foreground">
              {t(
                "任务由服务端继续执行；可以切换页面，返回后会自动恢复当前进度。",
              )}
            </div>
          ) : null}
        </div>
      ) : null}

      <Tabs defaultValue="memories">
        <TabsList>
          <TabsTrigger value="memories">
            {t("记忆")} · {formatNumber(memoryTotal)}
          </TabsTrigger>
          <TabsTrigger value="pending">
            {t("队列")} · {formatNumber(pending.length)}
          </TabsTrigger>
        </TabsList>

        <TabsContent value="memories">
          <div className="mb-3 flex items-center justify-between gap-3">
            <div className="relative w-full sm:w-80">
              <Search className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                value={memoryQuery}
                onChange={(event) => {
                  setMemoryQuery(event.target.value);
                  setMemoryOffset(0);
                }}
                maxLength={256}
                aria-label={t("搜索标题或轨迹 ID")}
                placeholder={t("搜索标题或轨迹 ID")}
                className="pl-9"
              />
            </div>
          </div>
          <Card>
            <CardContent className="p-0">
              {memories.length ? (
                <Table className="min-w-[1040px] table-fixed">
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-72">{t("记忆")}</TableHead>
                      <TableHead className="w-28">{t("任务结果")}</TableHead>
                      <TableHead className="w-64">{t("关联轨迹")}</TableHead>
                      <TableHead className="w-36">{t("来源规模")}</TableHead>
                      <TableHead className="w-36">{t("生成时间")}</TableHead>
                      <TableHead className="w-36 text-right">
                        {t("操作")}
                      </TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {memories.map((memory) => {
                      const runningThis =
                        regenerating &&
                        regeneration.details?.source_digest ===
                          memory.source_digest;
                      return (
                        <TableRow key={memory.source_digest}>
                          <TableCell>
                            <button
                              type="button"
                              className="line-clamp-2 w-full rounded-sm text-left text-sm font-medium leading-5 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                              onClick={() => void openMemory(memory)}
                            >
                              {memory.title}
                            </button>
                            <div className="mt-1 truncate font-mono text-[10px] text-muted-foreground">
                              {memory.document_path || memory.source_digest}
                            </div>
                            <div className="mt-2 flex flex-wrap gap-1">
                              <Badge variant="outline">
                                {memory.causal_schema === 1
                                  ? t("因果记录 v1")
                                  : t("Legacy：未提供因果 schema")}
                              </Badge>
                              <Badge variant="outline">
                                {t("发布状态")}:{" "}
                                {causalLabel(memory.publication_status)}
                              </Badge>
                              {!memory.document_path ? (
                                <Badge variant="outline">{t("未索引")}</Badge>
                              ) : null}
                              {memory.analysis_status ? (
                                <Badge variant="outline">
                                  {causalLabel(memory.analysis_status)}
                                </Badge>
                              ) : null}
                            </div>
                          </TableCell>
                          <TableCell>
                            <Badge variant={outcomeVariant(memory.outcome)}>
                              {t(OUTCOME_LABELS[memory.outcome])}
                            </Badge>
                          </TableCell>
                          <TableCell>
                            <div className="flex items-start gap-2">
                              <Link2 className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                              <div className="min-w-0">
                                <div className="break-all font-mono text-[11px] leading-4">
                                  {memory.trajectory_id}
                                </div>
                                <div className="mt-1 text-[10px] text-muted-foreground">
                                  {memory.source_available
                                    ? t("轨迹快照已保留")
                                    : t("历史轨迹未保留")}
                                </div>
                              </div>
                            </div>
                          </TableCell>
                          <TableCell className="text-xs text-muted-foreground">
                            <div>
                              {formatNumber(
                                memory.trajectory_source?.source_token_count ||
                                  memory.source_token_count,
                              )}{" "}
                              tokens
                            </div>
                            <div className="mt-1">
                              {t("{count} 条事件", {
                                count: formatNumber(
                                  memory.trajectory_source
                                    ?.trajectory_row_count ||
                                    memory.source_event_count,
                                ),
                              })}
                            </div>
                          </TableCell>
                          <TableCell className="font-mono text-xs text-muted-foreground">
                            {formatTime(memory.created_at)}
                          </TableCell>
                          <TableCell className="text-right">
                            <Button
                              variant="outline"
                              size="sm"
                              disabled={
                                !memory.source_available ||
                                !(
                                  memory.causal_schema || memory.document_sha256
                                ) ||
                                regenerating
                              }
                              onClick={() => void openRegeneration(memory)}
                            >
                              {runningThis ? (
                                <LoaderCircle className="animate-spin" />
                              ) : (
                                <RotateCcw />
                              )}
                              {t("重新反思")}
                            </Button>
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              ) : (
                <EmptyState
                  icon={loading ? LoaderCircle : BrainCircuit}
                  title={
                    !available
                      ? t("服务重启后启用")
                      : loading
                        ? t("正在读取反思记忆")
                        : memoryQuery
                          ? t("没有匹配记忆")
                          : t("暂无 Reflection Memory")
                  }
                  description={
                    !available
                      ? t("后端接口尚未进入当前运行进程。")
                      : memoryQuery
                        ? t("清除搜索条件后重试。")
                        : t("轨迹完成反思后会出现在这里。")
                  }
                />
              )}
            </CardContent>
          </Card>
          <nav
            aria-label={t("反思记忆分页")}
            className="mt-3 flex flex-wrap items-center justify-between gap-3"
          >
            <span role="status" className="text-xs text-muted-foreground">
              {t("第 {start}–{end} 条，共 {total} 条", {
                start: formatNumber(memories.length ? memoryOffset + 1 : 0),
                end: formatNumber(
                  memories.length ? memoryOffset + memories.length : 0,
                ),
                total: formatNumber(memoryTotal),
              })}
            </span>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                disabled={loading || memoryOffset === 0}
                onClick={() =>
                  setMemoryOffset((offset) =>
                    Math.max(0, offset - MEMORY_PAGE_SIZE),
                  )
                }
              >
                {t("上一页")}
              </Button>
              <Button
                variant="outline"
                size="sm"
                disabled={
                  loading || memoryOffset + MEMORY_PAGE_SIZE >= memoryTotal
                }
                onClick={() =>
                  setMemoryOffset((offset) => offset + MEMORY_PAGE_SIZE)
                }
              >
                {t("下一页")}
              </Button>
            </div>
          </nav>
        </TabsContent>

        <TabsContent value="pending">
          <div className="mb-3 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <div className="relative w-full sm:w-80">
              <Search className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                value={pendingQuery}
                onChange={(event) => setPendingQuery(event.target.value)}
                placeholder={t("搜索响应 ID 或摘要")}
                className="pl-9"
              />
            </div>
            <div className="flex items-center gap-2">
              <span className="mr-1 text-xs text-muted-foreground">
                {t("已选 {count}", { count: formatNumber(selected.size) })}
              </span>
              <Button
                size="sm"
                disabled={
                  !selected.size || selectedHasRunning || action !== null
                }
                onClick={() => void startNow([...selected])}
              >
                {action === "start" ? (
                  <LoaderCircle className="animate-spin" />
                ) : (
                  <Play />
                )}
                {t("立即反思")}
              </Button>
              <Button
                size="sm"
                variant="outline"
                disabled={!selected.size || action !== null}
                onClick={() => void cancel([...selected])}
              >
                {action === "cancel" ? (
                  <LoaderCircle className="animate-spin" />
                ) : (
                  <X />
                )}
                {t("取消反思")}
              </Button>
            </div>
          </div>
          <Card>
            <CardContent className="p-0">
              {visiblePending.length ? (
                <Table className="min-w-[1080px] table-fixed">
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-11">
                        <input
                          type="checkbox"
                          aria-label={t("全选待反思轨迹")}
                          checked={allVisibleSelected}
                          ref={(node) => {
                            if (node)
                              node.indeterminate =
                                someVisibleSelected && !allVisibleSelected;
                          }}
                          onChange={toggleAll}
                          className="h-4 w-4 accent-primary"
                        />
                      </TableHead>
                      <TableHead className="w-44">{t("响应 ID")}</TableHead>
                      <TableHead className="w-64">{t("摘要")}</TableHead>
                      <TableHead className="w-24">{t("状态")}</TableHead>
                      <TableHead className="w-32">{t("上次活动")}</TableHead>
                      <TableHead className="w-28">{t("开始整理")}</TableHead>
                      <TableHead className="w-28">{t("规模")}</TableHead>
                      <TableHead className="w-40 text-right">
                        {t("操作")}
                      </TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {visiblePending.map((item) => (
                      <TableRow key={item.conversation_key}>
                        <TableCell>
                          <input
                            type="checkbox"
                            aria-label={t("选择 {id}", {
                              id: item.trajectory_id,
                            })}
                            checked={selected.has(item.conversation_key)}
                            onChange={() => toggleOne(item.conversation_key)}
                            className="h-4 w-4 accent-primary"
                          />
                        </TableCell>
                        <TableCell>
                          <span className="block select-all break-all font-mono text-[11px] leading-4 text-muted-foreground">
                            {item.trajectory_id}
                          </span>
                        </TableCell>
                        <TableCell>
                          <button
                            type="button"
                            aria-expanded={
                              summaryConversationKey === item.conversation_key
                            }
                            aria-label={t("查看任务全文")}
                            className="line-clamp-2 w-full max-w-64 rounded-sm text-left text-sm font-medium leading-5 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                            onClick={() =>
                              setSummaryConversationKey(item.conversation_key)
                            }
                          >
                            {item.original_task || t("未命名任务")}
                          </button>
                          {item.error ? (
                            <p className="mt-2 whitespace-pre-wrap break-words text-xs text-destructive">
                              {item.error}
                            </p>
                          ) : null}
                        </TableCell>
                        <TableCell>
                          <Badge
                            variant={
                              item.status === "failed"
                                ? "destructive"
                                : item.status === "running"
                                  ? "default"
                                  : "outline"
                            }
                          >
                            {item.status === "failed"
                              ? t("失败")
                              : item.status === "running"
                                ? t("整理中")
                                : t("等待")}
                          </Badge>
                        </TableCell>
                        <TableCell className="font-mono text-xs text-muted-foreground">
                          {formatTime(item.last_activity_at)}
                        </TableCell>
                        <TableCell className="font-mono text-xs">
                          {remainingLabel(item, now)}
                        </TableCell>
                        <TableCell className="text-xs text-muted-foreground">
                          <div>
                            {formatNumber(item.source_token_count)} tokens
                          </div>
                          <div className="mt-1">
                            {t("{count} 工具事件", {
                              count: formatNumber(item.event_count),
                            })}
                          </div>
                        </TableCell>
                        <TableCell className="text-right">
                          <div className="flex justify-end gap-1">
                            <Button
                              variant="ghost"
                              size="sm"
                              disabled={
                                item.status === "running" || action !== null
                              }
                              onClick={() =>
                                void startNow([item.conversation_key])
                              }
                            >
                              <Play />
                              {item.status === "failed" ? t("重试") : t("立即")}
                            </Button>
                            <Button
                              variant="ghost"
                              size="sm"
                              disabled={action !== null}
                              onClick={() =>
                                void cancel([item.conversation_key])
                              }
                            >
                              <X />
                              {t("取消")}
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              ) : (
                <EmptyState
                  icon={loading ? LoaderCircle : Clock3}
                  title={
                    loading
                      ? t("正在读取反思队列")
                      : pendingQuery
                        ? t("没有匹配轨迹")
                        : t("暂无待反思轨迹")
                  }
                  description={
                    pendingQuery
                      ? t("清除搜索条件后重试。")
                      : t("新轨迹满足反思条件后会出现在这里。")
                  }
                />
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      <Dialog
        open={Boolean(memoryTarget)}
        onOpenChange={(open) => {
          if (!open) closeMemory();
        }}
      >
        <DialogContent className="max-w-3xl">
          <DialogHeader>
            <DialogTitle>
              {memoryDetail?.title || memoryTarget?.title}
            </DialogTitle>
            <DialogDescription className="break-all font-mono text-xs leading-5">
              {memoryDetail?.trajectory_id || memoryTarget?.trajectory_id}
            </DialogDescription>
            {evidenceLoading ? (
              <p
                role="status"
                className="flex items-center gap-2 text-xs text-muted-foreground"
              >
                <LoaderCircle className="h-4 w-4 animate-spin" />
                {t("正在读取证据事件")}
              </p>
            ) : null}
          </DialogHeader>
          <div className="max-h-[72vh] space-y-5 overflow-y-auto pr-2">
            {memoryDetailLoading ? (
              <p
                role="status"
                className="flex items-center gap-2 text-sm text-muted-foreground"
              >
                <LoaderCircle className="h-4 w-4 animate-spin" />
                {t("正在读取反思详情")}
              </p>
            ) : memoryDetailError ? (
              <div role="alert" className="space-y-3 text-sm text-destructive">
                <p>
                  {t("反思详情加载失败")}: {memoryDetailError}
                </p>
                <Button
                  variant="outline"
                  onClick={() => memoryTarget && void openMemory(memoryTarget)}
                >
                  {t("重试")}
                </Button>
              </div>
            ) : null}
            {memoryDetail ? (
              <>
                <div className="flex flex-wrap gap-2">
                  <Badge variant="outline">
                    {memoryDetail.causal_schema === 1
                      ? t("因果记录 v1")
                      : t("Legacy：未提供因果 schema")}
                  </Badge>
                  {memoryDetail.analysis_status ? (
                    <Badge variant="outline">
                      {causalLabel(memoryDetail.analysis_status)}
                    </Badge>
                  ) : null}
                  <Badge variant={outcomeVariant(memoryDetail.outcome)}>
                    {t(OUTCOME_LABELS[memoryDetail.outcome])}
                  </Badge>
                  <Badge variant="outline">
                    {t("发布状态")}:{" "}
                    {causalLabel(memoryDetail.publication_status)}
                  </Badge>
                  {!memoryDetail.document_path ? (
                    <Badge variant="outline">{t("未索引")}</Badge>
                  ) : null}
                </div>
                <p className="text-xs text-muted-foreground">
                  {t(
                    "任务结果、因果强度与 Knowledge 准入相互独立。active 表示可发布为召回候选，不代表已验证或已被选中；supported / unresolved 保留历史证据与假设，不是已确认根因或必执行规则。",
                  )}
                </p>
                {!memoryDetail.document_path ? (
                  <p className="text-sm text-muted-foreground">
                    {t(
                      "尚无已发布的 Knowledge 文档，未进入知识索引；条目准入资格不等于已发布。",
                    )}
                  </p>
                ) : null}
                {memoryDetail.coverage ? (
                  <section className="border bg-muted/20 p-3">
                    <h3 className="mb-2 text-sm font-semibold">
                      {t("覆盖状态")}
                    </h3>
                    <div className="flex flex-wrap gap-3 text-xs text-muted-foreground">
                      {(
                        [
                          "provided_events",
                          "analyzed_events",
                          "pending_events",
                          "failed_segments",
                        ] as const
                      ).map((key) =>
                        memoryDetail.coverage?.[key] !== undefined ? (
                          <span key={key}>
                            {causalLabel(key)}:{" "}
                            {String(memoryDetail.coverage[key])}
                          </span>
                        ) : null,
                      )}
                    </div>
                    {Array.isArray(memoryDetail.coverage.segments) ? (
                      <div className="mt-3 space-y-1 text-xs">
                        {memoryDetail.coverage.segments.map(
                          (segment, index) => (
                            <div
                              key={String(segment.segment_id || index)}
                              className="flex flex-wrap gap-2 border-t pt-1"
                            >
                              <span className="break-all font-mono">
                                {segment.segment_id}
                              </span>
                              <Badge
                                variant={
                                  segment.status === "failed" ||
                                  segment.status === "failed_closed"
                                    ? "destructive"
                                    : "outline"
                                }
                              >
                                {causalLabel(segment.status)}
                              </Badge>
                              {segment.reason ? (
                                <span className="text-muted-foreground">
                                  {String(segment.reason)}
                                </span>
                              ) : null}
                              {segment.event_ids?.length ? (
                                <details className="w-full">
                                  <summary className="cursor-pointer">
                                    {t("分段事件")}
                                  </summary>
                                  <div className="mt-2 flex flex-wrap gap-2">
                                    {segment.event_ids.map((eventId) => (
                                      <Button
                                        key={eventId}
                                        variant="outline"
                                        size="sm"
                                        disabled={evidenceLoading}
                                        className="h-auto whitespace-normal break-all text-left"
                                        onClick={() =>
                                          void inspectEvidence(eventId)
                                        }
                                      >
                                        {eventId}
                                      </Button>
                                    ))}
                                  </div>
                                </details>
                              ) : null}
                            </div>
                          ),
                        )}
                      </div>
                    ) : null}
                  </section>
                ) : null}
                {(memoryDetail.causal_entries || []).map((entry) => (
                  <section
                    key={`${entry.entry_id}:${entry.version}`}
                    className="space-y-3 border p-4"
                  >
                    <div className="flex flex-wrap items-center gap-2">
                      <h3 className="font-semibold">
                        {entry.title || entry.entry_id}
                      </h3>
                      <Badge variant="outline">
                        {t("证据强度")}: {causalLabel(entry.causal_status)}
                      </Badge>
                      <Badge
                        variant={
                          entry.admission_status === "active"
                            ? "success"
                            : "outline"
                        }
                      >
                        {t("准入")}: {causalLabel(entry.admission_status)}
                      </Badge>
                      <span className="text-xs text-muted-foreground">
                        v{entry.version}
                      </span>
                    </div>
                    <p className="break-all font-mono text-xs text-muted-foreground">
                      {entry.entry_id}
                    </p>
                    {(
                      [
                        "scope",
                        "problem",
                        "action",
                        "observation",
                        "mechanism",
                        "rule",
                        "next_check",
                      ] as const
                    ).map((key) => (
                      <div key={key}>
                        <div className="text-xs font-semibold text-muted-foreground">
                          {causalFieldLabel(key, entry.causal_status)}
                        </div>
                        <p className="whitespace-pre-wrap text-sm leading-6">
                          {entry[key] ||
                            t(
                              key === "scope"
                                ? "适用范围未提供；不可默认泛化。"
                                : "未提供",
                            )}
                        </p>
                      </div>
                    ))}
                    {(
                      [
                        "alternatives",
                        "counterevidence",
                        "missing_evidence",
                      ] as const
                    ).map((key) => (
                      <div key={key}>
                        <div className="text-xs font-semibold text-muted-foreground">
                          {causalLabel(key)}
                        </div>
                        {entry[key]?.length ? (
                          <ul className="list-disc pl-5 text-sm">
                            {entry[key].map((item, index) => (
                              <li key={index}>{item}</li>
                            ))}
                          </ul>
                        ) : (
                          <p className="text-sm text-muted-foreground">
                            {t(
                              key === "missing_evidence"
                                ? "未记录缺失项；不代表证据完备。"
                                : "未提供",
                            )}
                          </p>
                        )}
                      </div>
                    ))}
                    <section className="space-y-2 border-t pt-3">
                      <h4 className="text-xs font-semibold text-muted-foreground">
                        {t("证据与验证")}
                      </h4>
                      {entry.verification?.method ? (
                        <p className="whitespace-pre-wrap text-sm">
                          {entry.verification.method}
                        </p>
                      ) : null}
                      {[
                        ...(entry.evidence_refs || []),
                        ...(entry.verification?.evidence_refs || []),
                      ].map((ref, index) => (
                        <div
                          key={`${ref.event_id}:${index}`}
                          className="space-y-1"
                        >
                          <blockquote className="whitespace-pre-wrap break-words border-l-2 pl-3 text-sm">
                            {ref.quote}
                          </blockquote>
                          <Button
                            variant="outline"
                            size="sm"
                            disabled={evidenceLoading}
                            className="h-auto whitespace-normal break-all text-left"
                            onClick={() => void inspectEvidence(ref.event_id)}
                          >
                            {t("查看证据")} · {ref.event_id}
                          </Button>
                        </div>
                      ))}
                    </section>
                    {entry.reason ? (
                      <p className="whitespace-pre-wrap text-sm">
                        {t("修订原因")}: {entry.reason}
                      </p>
                    ) : null}
                    <details className="border-t pt-3">
                      <summary className="cursor-pointer text-sm">
                        {t("版本历史")} · {entry.entry_id}
                      </summary>
                      {entry.versions?.length ? (
                        entry.versions.map((version) => (
                          <details
                            key={version.version}
                            className="mt-3 border-l pl-3"
                          >
                            <summary className="cursor-pointer text-xs">
                              v{version.version} · {version.title} ·{" "}
                              {causalLabel(version.admission_status)}
                            </summary>
                            <pre className="mt-2 overflow-auto whitespace-pre-wrap break-words text-xs leading-5">
                              {JSON.stringify(version, null, 2)}
                            </pre>
                            {(version.evidence_refs || []).map((ref, index) => (
                              <Button
                                key={`${ref.event_id}:${index}`}
                                variant="outline"
                                size="sm"
                                disabled={evidenceLoading}
                                className="mt-2 h-auto whitespace-normal break-all text-left"
                                onClick={() =>
                                  void inspectEvidence(ref.event_id)
                                }
                              >
                                {t("查看证据")} · {ref.event_id}
                              </Button>
                            ))}
                          </details>
                        ))
                      ) : (
                        <p className="mt-2 text-xs text-muted-foreground">
                          {t("没有更早的条目版本。")}
                        </p>
                      )}
                    </details>
                  </section>
                ))}
                <details
                  open={!memoryDetail.causal_schema}
                  className="space-y-3 border-t pt-3"
                >
                  <summary className="cursor-pointer text-sm">
                    {memoryDetail.causal_schema
                      ? t("派生摘要")
                      : t("Legacy 历史正文（因果强度未分级）")}
                  </summary>
                  {memoryTextSections(memoryDetail).map(([label, content]) => (
                    <section key={label}>
                      <h3 className="mb-1 text-xs font-semibold text-muted-foreground">
                        {t(label)}
                      </h3>
                      <p className="whitespace-pre-wrap text-sm leading-6">
                        {content || t("未提供")}
                      </p>
                    </section>
                  ))}
                </details>
                <RawSourceInspection
                  key={memoryDetail.source_digest}
                  memory={memoryDetail}
                />
              </>
            ) : null}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={closeMemory}>
              {t("关闭")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      <Dialog
        open={evidenceLoading || Boolean(evidenceEvent)}
        onOpenChange={(open) => {
          if (!open) closeEvidence();
        }}
      >
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>{t("精确事件证据")}</DialogTitle>
            <DialogDescription>
              {evidenceLoading
                ? t("正在读取")
                : String(evidenceEvent?.event_id || "")}
            </DialogDescription>
          </DialogHeader>
          <pre className="max-h-[60vh] overflow-auto whitespace-pre-wrap break-words border bg-muted/30 p-4 text-xs leading-5">
            {evidenceEvent ? JSON.stringify(evidenceEvent, null, 2) : ""}
          </pre>
          <DialogFooter>
            <Button variant="outline" onClick={closeEvidence}>
              {t("关闭")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      <Dialog
        open={Boolean(regenerateTarget)}
        onOpenChange={(open) => {
          if (!open && !submitting) closeRegeneration();
        }}
      >
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>{t("重新反思")}</DialogTitle>
            <DialogDescription>
              {t(
                "使用保留的轨迹与补充证据重新分析。未明确修订的条目保持不变，候选结论不等于已发布经验。",
              )}
            </DialogDescription>
          </DialogHeader>
          {sourceLoading ? (
            <div className="flex min-h-40 items-center justify-center text-muted-foreground">
              <LoaderCircle className="mr-2 h-4 w-4 animate-spin" />
              {t("正在读取关联轨迹")}
            </div>
          ) : sourceDetail ? (
            <div className="space-y-4">
              <div className="border bg-muted/20 p-3">
                <div className="break-all font-mono text-[11px] leading-5">
                  {sourceDetail.source.trajectory_id}
                </div>
                <div className="mt-2 line-clamp-3 whitespace-pre-wrap text-sm leading-5">
                  {sourceDetail.source.original_task || t("未命名任务")}
                </div>
                <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
                  <span>
                    {formatNumber(sourceDetail.source.source_token_count)}{" "}
                    tokens
                  </span>
                  <span>
                    {t("{count} 条事件", {
                      count: formatNumber(
                        sourceDetail.source.trajectory_row_count,
                      ),
                    })}
                  </span>
                  <span>
                    {t("{count} 个 capsule", {
                      count: formatNumber(sourceDetail.source.capsule_count),
                    })}
                  </span>
                </div>
              </div>
              <div className="space-y-2">
                <Label htmlFor="reflection-verifier-feedback">
                  {t("Verifier 反馈")}
                </Label>
                <Textarea
                  id="reflection-verifier-feedback"
                  value={verifierFeedback}
                  onChange={(event) => setVerifierFeedback(event.target.value)}
                  placeholder={t(
                    "粘贴 verifier 的通过项、失败项、错误原文与验收边界。",
                  )}
                  className="min-h-44 resize-y font-mono text-xs leading-5"
                  maxLength={131072}
                />
                <div className="text-[11px] text-muted-foreground">
                  {t(
                    "分析失败保留原记忆；只有通过证据准入的条目才可发布并更新原生记忆。",
                  )}
                </div>
              </div>
            </div>
          ) : (
            <div className="min-h-32 border border-destructive/30 bg-destructive/5 p-4 text-sm text-destructive">
              {t("关联轨迹不可用，无法重新反思。")}
            </div>
          )}
          <DialogFooter>
            <Button
              variant="outline"
              disabled={submitting}
              onClick={closeRegeneration}
            >
              {t("取消")}
            </Button>
            <Button
              disabled={
                !sourceDetail ||
                !verifierFeedback.trim() ||
                submitting ||
                regenerating
              }
              onClick={() => void submitRegeneration()}
            >
              {submitting ? (
                <LoaderCircle className="animate-spin" />
              ) : (
                <RotateCcw />
              )}
              {t("开始重新反思")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={Boolean(summaryItem)}
        onOpenChange={(open) => {
          if (!open) setSummaryConversationKey(null);
        }}
      >
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>{t("任务全文")}</DialogTitle>
            <DialogDescription className="break-all font-mono text-xs leading-5">
              {summaryItem?.trajectory_id}
            </DialogDescription>
          </DialogHeader>
          <div className="max-h-[60vh] overflow-y-auto overflow-x-hidden border bg-muted/30 p-4">
            <p className="whitespace-pre-wrap break-words text-sm leading-6">
              {summaryItem?.original_task || t("未命名任务")}
            </p>
            {summaryItem?.error ? (
              <p className="mt-4 whitespace-pre-wrap break-words text-sm text-destructive">
                {summaryItem.error}
              </p>
            ) : null}
            {summaryItem?.coverage ? (
              <section className="mt-4 border-t pt-4">
                <h3 className="mb-2 text-sm font-semibold">{t("覆盖状态")}</h3>
                <pre className="whitespace-pre-wrap break-words text-xs leading-5">
                  {JSON.stringify(summaryItem.coverage, null, 2)}
                </pre>
              </section>
            ) : null}
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setSummaryConversationKey(null)}
            >
              {t("关闭")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
