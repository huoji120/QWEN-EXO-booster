import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  BookOpenCheck,
  CheckCircle2,
  Eraser,
  FileSearch,
  Hash,
  LoaderCircle,
  RefreshCw,
  Search,
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
import {
  ApiError,
  clearTelemetry,
  getMemoryToolCalls,
  getRequestTraces,
  getServiceConfig,
  getTelemetry,
} from "@/lib/api";
import { currentLanguage, translate as t } from "@/lib/i18n";
import type {
  MemoryToolCall,
  RequestTrace,
  RequestTraceListing,
  TelemetryEvent,
} from "@/lib/types";
import { isTraceDefaultScope } from "@/lib/trace-preferences";
import { cn, formatDuration, formatNumber, shortHash } from "@/lib/utils";

type TraceFilter = "activity" | "all" | "running";

const FILTERS: { value: TraceFilter; label: string }[] = [
  { value: "activity", label: "有召回" },
  { value: "all", label: "全部" },
  { value: "running", label: "推理中" },
];

const TOOL_LABELS: Record<string, string> = {
  qwen_exo_memory_search: "搜索记忆",
  qwen_exo_memory_read: "读取记忆",
};

const EVENT_TYPE_LABELS: Record<string, string> = {
  "request.started": "请求开始",
  "request.completed": "请求完成",
  "request.cancelled": "请求取消",
  "request.stage_summary": "阶段汇总",
  "memory_tool.call": "记忆工具调用",
  "reasoning.budget_forced": "推理预算截断",
  "observer.decode_summary": "解码观测汇总",
};

function timeOnly(value?: string | number | null) {
  if (value === null || value === undefined || value === "") return "—";
  const date = new Date(typeof value === "number" ? value * 1000 : value);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleTimeString(currentLanguage() === "en-US" ? "en-US" : "zh-CN", {
    hour12: false,
  });
}

function toolArguments(call: MemoryToolCall): Record<string, unknown> {
  try {
    const parsed = JSON.parse(call.arguments || "{}");
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

/** The one-line gist of a call: the search pattern or the read range. */
function toolGist(call: MemoryToolCall) {
  const args = toolArguments(call);
  if (call.name === "qwen_exo_memory_search") {
    const scope = args.path ? ` · ${String(args.path)}` : "";
    return `${String(args.pattern ?? "")}${scope}`;
  }
  if (call.name === "qwen_exo_memory_read") {
    const offset = Number(args.offset ?? 1) || 1;
    const limit = Number(args.limit ?? 200) || 200;
    return `${String(args.path ?? "")} · L${offset}–${offset + limit - 1}`;
  }
  return call.arguments;
}

function eventType(event: TelemetryEvent): string {
  return String(event.event_type ?? event.type ?? event.event ?? "");
}

function eventSummary(event: TelemetryEvent) {
  const payload = (event.payload ?? {}) as Record<string, unknown>;
  if (eventType(event) === "memory_tool.call") {
    return `${t(TOOL_LABELS[String(payload.tool)] ?? String(payload.tool))} · ${
      payload.ok ? t("成功") : t("失败")
    } · ${formatNumber(Number(payload.output_chars ?? 0))} ${t("字")}`;
  }
  if (eventType(event) === "request.completed") {
    return t("输出 {count} token", { count: formatNumber(Number(payload.output_tokens ?? 0)) });
  }
  const keys = Object.keys(payload).slice(0, 4);
  return keys.join(" · ");
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
  return (
    <button
      type="button"
      onClick={onSelect}
      className={cn(
        "relative w-full border-b px-4 py-3 text-left transition-colors last:border-b-0 hover:bg-muted/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring",
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
        {trace.memory_tool_calls ? (
          <Badge
            variant={trace.memory_tool_failures ? "warning" : "success"}
            className="px-1.5 py-0 text-[10px]"
          >
            {t("已召回 ×{count}", { count: formatNumber(trace.memory_tool_calls) })}
          </Badge>
        ) : (
          <Badge variant="secondary" className="px-1.5 py-0 text-[10px]">
            {t("未召回")}
          </Badge>
        )}
        <span className="ml-auto font-mono text-[9px] text-muted-foreground/70">
          {shortHash(trace.request_id, 9)}
        </span>
      </div>
    </button>
  );
}

function ToolCallRow({ call, index }: { call: MemoryToolCall; index: number }) {
  return (
    <details className="group border-b last:border-b-0">
      <summary className="flex cursor-pointer list-none items-center gap-3 px-4 py-3 hover:bg-muted/50 [&::-webkit-details-marker]:hidden">
        <span className="w-6 font-mono text-[10px] text-muted-foreground">#{index + 1}</span>
        {call.ok ? (
          <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-600" />
        ) : (
          <XCircle className="h-4 w-4 shrink-0 text-destructive" />
        )}
        <Badge variant="outline" className="shrink-0 px-1.5 py-0 text-[10px]">
          {t(TOOL_LABELS[call.name] ?? call.name)}
        </Badge>
        <span className="min-w-0 flex-1 truncate font-mono text-xs" title={toolGist(call)}>
          {toolGist(call)}
        </span>
        <span className="shrink-0 font-mono text-[10px] text-muted-foreground">
          {t("第 {round} 轮", { round: call.round + 1 })} ·{" "}
          {formatNumber(Math.round(call.duration_ms))} ms ·{" "}
          {formatNumber(call.output_chars)} {t("字")}
        </span>
      </summary>
      <div className="space-y-2 border-t bg-muted/30 p-4">
        <div>
          <div className="mb-1 text-[11px] font-semibold text-muted-foreground">
            {t("参数")}
          </div>
          <pre className="overflow-auto whitespace-pre-wrap break-words rounded-md border bg-background p-2 font-mono text-[11px] leading-5">
            {JSON.stringify(toolArguments(call), null, 2)}
          </pre>
        </div>
        <div>
          <div className="mb-1 text-[11px] font-semibold text-muted-foreground">
            {call.ok ? t("召回内容") : t("错误")}
          </div>
          <pre className="max-h-[480px] overflow-auto whitespace-pre-wrap break-words rounded-md border bg-background p-2 font-mono text-[11px] leading-5">
            {call.output ?? ""}
          </pre>
        </div>
      </div>
    </details>
  );
}

function EventRow({ event }: { event: TelemetryEvent }) {
  const [open, setOpen] = useState(false);
  const type = eventType(event);
  return (
    <details
      className="group border-b last:border-b-0"
      onToggle={(toggle) => setOpen(toggle.currentTarget.open)}
    >
      <summary className="grid cursor-pointer list-none grid-cols-[54px_minmax(150px,220px)_minmax(0,1fr)] items-center gap-3 px-3 py-2.5 hover:bg-muted/50 [&::-webkit-details-marker]:hidden">
        <span className="font-mono text-[10px] text-muted-foreground">
          #{event.event_id ?? "—"}
        </span>
        <span className="truncate text-xs font-semibold" title={type}>
          {t(EVENT_TYPE_LABELS[type] ?? type)}
        </span>
        <span className="truncate text-[11px] text-muted-foreground">{eventSummary(event)}</span>
      </summary>
      {open ? (
        <div className="border-t bg-slate-950 p-3 text-slate-200">
          <div className="mb-2 flex items-center justify-between font-mono text-[10px] text-slate-400">
            <span>{type}</span>
            <span>{timeOnly(event.timestamp)}</span>
          </div>
          <pre className="max-h-80 overflow-auto whitespace-pre-wrap break-words font-mono text-[11px] leading-5">
            {JSON.stringify(event.payload || {}, null, 2)}
          </pre>
        </div>
      ) : null}
    </details>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border bg-card px-3 py-2">
      <div className="text-[10px] text-muted-foreground">{label}</div>
      <div className="mt-0.5 font-mono text-sm font-semibold">{value}</div>
    </div>
  );
}

function TextBlock({ title, text }: { title: string; text?: string }) {
  if (!text) return null;
  return (
    <details className="rounded-xl border bg-card shadow-sm">
      <summary className="cursor-pointer list-none px-4 py-3 text-sm font-semibold [&::-webkit-details-marker]:hidden">
        {title}
        <span className="ml-2 text-[10px] font-normal text-muted-foreground">
          {formatNumber(text.length)} {t("字")}
        </span>
      </summary>
      <pre className="max-h-[420px] overflow-auto whitespace-pre-wrap break-words border-t p-4 font-mono text-xs leading-5">
        {text}
      </pre>
    </details>
  );
}

type Detail = {
  requestId: string;
  calls: MemoryToolCall[];
  events: TelemetryEvent[];
};

function DetailPanel({
  trace,
  detail,
  loading,
  error,
  onRetry,
}: {
  trace: RequestTrace;
  detail: Detail | null;
  loading: boolean;
  error: string | null;
  onRetry: () => void;
}) {
  return (
    <div className="min-w-0 space-y-4">
      <div className="rounded-xl border bg-card p-4 shadow-sm">
        <div className="flex items-start justify-between gap-3">
          <p className="min-w-0 text-sm font-medium leading-6">
            {trace.input_text || t("该事件片段缺少请求输入")}
          </p>
          <span className="shrink-0 font-mono text-[10px] text-muted-foreground">
            {trace.request_id}
          </span>
        </div>
        <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Metric
            label={t("耗时")}
            value={
              trace.duration_seconds === null
                ? t("推理中")
                : formatDuration(trace.duration_seconds)
            }
          />
          <Metric label={t("输出 token")} value={formatNumber(trace.output_tokens)} />
          <Metric label={t("记忆工具调用")} value={formatNumber(trace.memory_tool_calls)} />
          <Metric label={t("失败")} value={formatNumber(trace.memory_tool_failures)} />
        </div>
      </div>

      <section className="overflow-hidden rounded-xl border bg-card shadow-sm">
        <div className="flex items-center gap-2 border-b px-4 py-3">
          <BookOpenCheck className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-semibold">{t("已召回")}</span>
          <span className="text-[11px] text-muted-foreground">
            {t("模型通过服务端记忆工具搜索、读取的内容，点击展开查看完整召回结果")}
          </span>
        </div>
        {loading ? (
          <div className="flex items-center gap-2 px-4 py-6 text-sm text-muted-foreground">
            <LoaderCircle className="h-4 w-4 animate-spin" />
            {t("加载中")}
          </div>
        ) : error ? (
          <div className="px-4 py-6 text-sm">
            <p className="text-destructive">{error}</p>
            <Button className="mt-3" size="sm" variant="outline" onClick={onRetry}>
              <RefreshCw />
              {t("重试")}
            </Button>
          </div>
        ) : detail?.calls.length ? (
          detail.calls.map((call, index) => (
            <ToolCallRow key={`${call.call_id}-${index}`} call={call} index={index} />
          ))
        ) : (
          <p className="px-4 py-6 text-sm text-muted-foreground">
            {trace.memory_tool_calls
              ? t("召回明细已超出保留范围。")
              : t("这次请求没有调用记忆工具。")}
          </p>
        )}
      </section>

      <TextBlock title={t("模型输出")} text={trace.output_text} />
      <TextBlock title={t("推理过程")} text={trace.reasoning_text} />

      <details className="overflow-hidden rounded-xl border bg-card shadow-sm">
        <summary className="cursor-pointer list-none px-4 py-3 text-sm font-semibold [&::-webkit-details-marker]:hidden">
          {t("原始事件")}
          <span className="ml-2 text-[10px] font-normal text-muted-foreground">
            {formatNumber(detail?.events.length ?? 0)}
          </span>
        </summary>
        <div className="border-t">
          {(detail?.events ?? []).map((event, index) => (
            <EventRow key={`${event.event_id ?? index}`} event={event} />
          ))}
        </div>
      </details>
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
  const [detail, setDetail] = useState<Detail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [clearAllOpen, setClearAllOpen] = useState(false);
  const [clearingAll, setClearingAll] = useState(false);
  const listController = useRef<AbortController | null>(null);
  const detailController = useRef<AbortController | null>(null);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const config = await getServiceConfig();
        const value = String(config.values.qwen_exo_console_trace_default_scope || "activity");
        if (!cancelled && isTraceDefaultScope(value)) {
          setFilter(value === "all" ? "all" : "activity");
        }
      } catch {
        // 保留页面默认筛选；配置不可读不应阻断轨迹查看。
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const loadListing = useCallback(async () => {
    listController.current?.abort();
    const controller = new AbortController();
    listController.current = controller;
    setLoading(true);
    try {
      const next = await getRequestTraces(100, "", controller.signal);
      if (controller.signal.aborted) return;
      setListing(next);
      setUnavailable(false);
    } catch (error) {
      if (controller.signal.aborted) return;
      if (error instanceof ApiError && error.status === 404) setUnavailable(true);
      else
        toast.error(t("请求轨迹加载失败"), {
          description: error instanceof Error ? error.message : t("未知错误"),
        });
    } finally {
      if (!controller.signal.aborted) setLoading(false);
    }
  }, []);

  const loadDetail = useCallback(async (requestId: string) => {
    detailController.current?.abort();
    const controller = new AbortController();
    detailController.current = controller;
    setDetailError(null);
    setDetailLoading(true);
    try {
      const [calls, telemetry] = await Promise.all([
        getMemoryToolCalls(requestId, controller.signal),
        getTelemetry(1000, requestId, controller.signal),
      ]);
      if (controller.signal.aborted) return;
      setDetail({ requestId, calls: calls.calls, events: telemetry.events || [] });
    } catch (error) {
      if (controller.signal.aborted) return;
      setDetailError(error instanceof Error ? error.message : t("未知错误"));
    } finally {
      if (!controller.signal.aborted) setDetailLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadListing();
    return () => {
      listController.current?.abort();
      detailController.current?.abort();
    };
  }, [loadListing]);

  const requests = useMemo(() => listing?.requests ?? [], [listing]);
  const filteredRequests = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return requests.filter((trace) => {
      if (
        needle &&
        !trace.request_id.toLowerCase().includes(needle) &&
        !trace.input_text.toLowerCase().includes(needle) &&
        !trace.output_text.toLowerCase().includes(needle)
      ) {
        return false;
      }
      if (filter === "activity") return trace.memory_tool_calls > 0;
      if (filter === "running") return trace.duration_seconds === null;
      return true;
    });
  }, [filter, query, requests]);

  useEffect(() => {
    if (!filteredRequests.length) {
      setSelectedId(null);
      return;
    }
    if (!selectedId || !filteredRequests.some((trace) => trace.request_id === selectedId)) {
      setSelectedId(filteredRequests[0].request_id);
    }
  }, [filteredRequests, selectedId]);

  const selectedTrace = useMemo(
    () => requests.find((trace) => trace.request_id === selectedId) || null,
    [requests, selectedId],
  );

  useEffect(() => {
    setDetail(null);
    if (selectedId) void loadDetail(selectedId);
  }, [loadDetail, selectedId]);

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
          "每个请求里模型调用服务端记忆工具（qwen_exo_memory_search / qwen_exo_memory_read）的完整记录：搜了什么、读了哪里、召回了什么内容。",
        )}
        actions={
          <>
            <Button
              variant="outline"
              size="sm"
              onClick={() => void loadListing()}
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
        <EmptyState icon={LoaderCircle} title={t("正在加载请求轨迹")} description="" />
      ) : !requests.length ? (
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
                placeholder={t("搜索请求或响应…")}
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
                    filter === item.value && "bg-background text-foreground shadow-sm",
                  )}
                >
                  {t(item.label)}
                </button>
              ))}
            </div>
            <span className="shrink-0 px-1 text-[10px] text-muted-foreground">
              {t("{count} 请求", { count: formatNumber(listing?.total_requests) })}
            </span>
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
                    <p className="mt-3 text-sm font-medium">{t("没有匹配请求")}</p>
                    <p className="mt-1 text-xs text-muted-foreground">
                      {t("清除搜索或切换筛选条件。")}
                    </p>
                  </div>
                )}
              </ScrollArea>
            </aside>

            {selectedTrace ? (
              <DetailPanel
                key={selectedTrace.request_id}
                trace={selectedTrace}
                detail={detail?.requestId === selectedTrace.request_id ? detail : null}
                loading={detailLoading || detail?.requestId !== selectedTrace.request_id}
                error={detailError}
                onRetry={() => void loadDetail(selectedTrace.request_id)}
              />
            ) : (
              <div className="grid min-h-[560px] place-items-center rounded-xl border border-dashed">
                <p className="text-sm text-muted-foreground">{t("选择一个请求查看召回记录。")}</p>
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
              {t("将删除全部请求事件。召回明细日志保留，不会重启服务，此操作无法撤销。")}
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
