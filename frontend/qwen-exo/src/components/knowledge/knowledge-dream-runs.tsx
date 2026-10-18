import { useEffect, useState } from "react";
import { ArrowRight, ChevronDown, ChevronRight, LoaderCircle } from "lucide-react";
import { Badge, type BadgeProps } from "@/components/ui/badge";
import { getDreamRun } from "@/lib/api";
import { translate as t } from "@/lib/i18n";
import type { DreamAction, DreamRunDetail, DreamRunSummary } from "@/lib/types";
import { cn, formatDuration, formatNumber, formatTime } from "@/lib/utils";
import { errorText } from "./knowledge-utils";

/** Chinese source strings; pass through ``t()`` when rendering. */
const ACTION_LABELS: Record<string, string> = {
  write: "写入",
  move: "移动",
  delete: "删除",
  mkdir: "新建文件夹",
};

const TRIGGER_LABELS: Record<string, string> = {
  nightly: "夜间",
  manual: "手动",
};

const STATUS_VIEWS: Record<string, { label: string; variant: BadgeProps["variant"] }> = {
  running: { label: "运行中", variant: "default" },
  completed: { label: "已完成", variant: "success" },
  stopped: { label: "已停止", variant: "warning" },
  failed: { label: "失败", variant: "destructive" },
  cancelled: { label: "已取消", variant: "outline" },
};

export function dreamActionLabel(action: string) {
  return ACTION_LABELS[action] ? t(ACTION_LABELS[action]) : action;
}

export function nsToTime(ns: number | null | undefined) {
  return typeof ns === "number" ? formatTime(ns / 1_000_000_000) : "—";
}

export function DreamStatusBadge({ status }: { status: string }) {
  const view = STATUS_VIEWS[status];
  return (
    <Badge variant={view?.variant ?? "outline"} className="whitespace-nowrap">
      {status === "running" ? <LoaderCircle className="h-3 w-3 animate-spin" /> : null}
      {view ? t(view.label) : status}
    </Badge>
  );
}

/** One applied action: label, ``target -> to``; failures show their message. */
export function DreamActionLine({ action }: { action: DreamAction }) {
  return (
    <div
      className={cn(
        "rounded-md px-2 py-1",
        !action.ok && "border border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-950",
      )}
    >
      <div className="flex min-w-0 flex-wrap items-center gap-1.5">
        <span className="shrink-0 font-mono text-[11px] text-muted-foreground">
          #{action.step}
        </span>
        <Badge variant={action.ok ? "secondary" : "destructive"} className="shrink-0">
          {dreamActionLabel(action.action)}
        </Badge>
        <span className="min-w-0 break-all font-mono">{action.target}</span>
        {action.to ? (
          <>
            <ArrowRight className="h-3 w-3 shrink-0 text-muted-foreground" />
            <span className="min-w-0 break-all font-mono">{action.to}</span>
          </>
        ) : null}
        {!action.ok ? (
          <Badge variant="destructive" className="shrink-0">
            {t("失败")}
          </Badge>
        ) : null}
      </div>
      {!action.ok && action.message ? (
        <p className="mt-1 break-words text-red-700 dark:text-red-300">{action.message}</p>
      ) : null}
    </div>
  );
}

function RunActions({ run }: { run: DreamRunSummary }) {
  const [detail, setDetail] = useState<DreamRunDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Refetch while the run is still progressing.
  const version = `${run.id}:${run.status}:${run.steps}:${run.action_count}`;

  useEffect(() => {
    const controller = new AbortController();
    setError(null);
    getDreamRun(run.id, controller.signal)
      .then(setDetail)
      .catch((cause) => {
        if (!controller.signal.aborted) setError(errorText(cause, t("未知错误")));
      });
    return () => controller.abort();
  }, [run.id, version]);

  if (error) return <p className="text-destructive">{error}</p>;
  if (!detail) {
    return (
      <p className="flex items-center gap-1.5 text-muted-foreground">
        <LoaderCircle className="h-3.5 w-3.5 animate-spin" />
        {t("加载中")}
      </p>
    );
  }
  return (
    <div className="space-y-2">
      {detail.summary ? (
        <p className="whitespace-pre-wrap break-words">{detail.summary}</p>
      ) : null}
      {detail.actions.length ? (
        <div className="space-y-1">
          {detail.actions.map((action, index) => (
            <DreamActionLine key={`${action.step}-${index}`} action={action} />
          ))}
        </div>
      ) : (
        <p className="text-muted-foreground">{t("这次整理没有改动任何文件")}</p>
      )}
    </div>
  );
}

/** Run history, latest first; a row expands into its action list. */
export function DreamRunList({ runs }: { runs: DreamRunSummary[] }) {
  const [expanded, setExpanded] = useState<number | null>(null);
  const ordered = [...runs].sort((a, b) => b.started_ns - a.started_ns);

  if (!ordered.length) {
    return <p className="py-4 text-center text-muted-foreground">{t("还没有整理记录")}</p>;
  }
  return (
    <ul className="divide-y rounded-md border">
      {ordered.map((run) => {
        const open = expanded === run.id;
        const seconds =
          run.finished_ns !== null && run.finished_ns >= run.started_ns
            ? (run.finished_ns - run.started_ns) / 1_000_000_000
            : null;
        return (
          <li key={run.id}>
            <button
              type="button"
              className="flex w-full flex-wrap items-center gap-x-3 gap-y-1 px-2 py-2 text-left hover:bg-muted/50"
              aria-expanded={open}
              onClick={() => setExpanded(open ? null : run.id)}
            >
              {open ? (
                <ChevronDown className="h-3.5 w-3.5 shrink-0" />
              ) : (
                <ChevronRight className="h-3.5 w-3.5 shrink-0" />
              )}
              <span className="font-mono" title={`#${run.id}`}>
                {nsToTime(run.started_ns)}
              </span>
              <Badge variant="outline">
                {TRIGGER_LABELS[run.trigger] ? t(TRIGGER_LABELS[run.trigger]) : run.trigger}
              </Badge>
              <DreamStatusBadge status={run.status} />
              <span className="text-muted-foreground">
                {t("{count} 步", { count: formatNumber(run.steps) })}
              </span>
              <span className="text-muted-foreground">
                {t("{count} 处改动", { count: formatNumber(run.action_count) })}
              </span>
              {seconds !== null ? (
                <span className="text-muted-foreground">{formatDuration(seconds)}</span>
              ) : null}
              {run.summary && !open ? (
                <span className="min-w-0 basis-full truncate pl-5 text-muted-foreground sm:basis-0 sm:flex-1 sm:pl-0">
                  {run.summary}
                </span>
              ) : null}
            </button>
            {open ? (
              <div className="border-t bg-muted/20 px-3 py-2 pl-7">
                <RunActions run={run} />
              </div>
            ) : null}
          </li>
        );
      })}
    </ul>
  );
}
