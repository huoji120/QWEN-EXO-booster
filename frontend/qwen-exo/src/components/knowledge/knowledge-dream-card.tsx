import { useEffect, useState } from "react";
import { LoaderCircle, Moon, RefreshCw, Sparkles } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { translate as t } from "@/lib/i18n";
import { formatNumber, formatTime } from "@/lib/utils";
import { DreamActionLine, DreamRunList } from "./knowledge-dream-runs";
import type { DreamState } from "./use-dream-status";

function countdown(seconds: number) {
  const total = Math.max(0, Math.round(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const rest = total % 60;
  const pad = (value: number) => String(value).padStart(2, "0");
  return `${pad(hours)}:${pad(minutes)}:${pad(rest)}`;
}

/** Ticking "next automatic run" line anchored at the last poll. */
function NextRun({ seconds, fetchedAt }: { seconds: number; fetchedAt: number }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1_000);
    return () => window.clearInterval(timer);
  }, []);
  const at = fetchedAt + seconds * 1_000;
  const left = (at - now) / 1_000;
  return (
    <span>
      {t("下次自动整理")}：
      <span className="font-mono text-foreground">{formatTime(at / 1_000)}</span>
      <span className="ml-1 font-mono">
        {left > 0 ? t("（{time} 后）", { time: countdown(left) }) : t("（即将开始）")}
      </span>
    </span>
  );
}

/** 记忆整理（Dream）: nightly work-memory tidy-up status, manual run, history. */
export function KnowledgeDreamCard({ state }: { state: DreamState }) {
  const { status, fetchedAt, error, starting } = state;
  const available = status?.available ?? false;
  const running = Boolean(status?.running);
  const current = status?.current ?? null;

  return (
    <Card>
      <CardContent className="space-y-3 pt-6 text-xs">
        <div className="flex flex-wrap items-center gap-2">
          <Moon className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium">{t("记忆整理（Dream）")}</span>
          {status === null ? null : !available ? (
            <Badge variant="outline">{t("不可用")}</Badge>
          ) : running ? (
            <Badge variant="default">
              <LoaderCircle className="h-3 w-3 animate-spin" />
              {t("整理中")}
            </Badge>
          ) : (
            <Badge variant="secondary">{t("空闲")}</Badge>
          )}
          <div className="ml-auto flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              aria-label={t("刷新")}
              onClick={() => void state.refresh()}
            >
              <RefreshCw />
            </Button>
            <Button
              size="sm"
              disabled={!available || running || starting}
              onClick={() => void state.start()}
            >
              {starting ? <LoaderCircle className="animate-spin" /> : <Sparkles />}
              {t("立即整理")}
            </Button>
          </div>
        </div>
        <p className="text-muted-foreground">
          {t(
            "每天服务器时间 00:00 自动运行一个整理代理，合并、归档或清理模型自己写下的工作笔记；删除的笔记会进入下方回收站，可随时恢复。",
          )}
        </p>
        {error ? <p className="break-words text-destructive">{error}</p> : null}
        {status !== null && !available ? (
          <p className="text-muted-foreground">
            {t("记忆整理不可用（服务端未启用或模型未就绪）。")}
          </p>
        ) : null}
        {available ? (
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-muted-foreground">
            {running && current ? (
              <span className="text-foreground">
                {t("运行 #{id}，第 {step} 步", {
                  id: current.run_id,
                  step: formatNumber(current.step),
                })}
              </span>
            ) : null}
            {typeof status?.next_run_in_seconds === "number" ? (
              <NextRun seconds={status.next_run_in_seconds} fetchedAt={fetchedAt} />
            ) : null}
          </div>
        ) : null}
        {running && current?.last_action ? (
          <div className="space-y-1">
            <span className="text-muted-foreground">{t("最近一步")}</span>
            <DreamActionLine action={current.last_action} />
          </div>
        ) : null}
        {status?.runs ? (
          <div className="space-y-1">
            <span className="text-muted-foreground">{t("整理记录")}</span>
            <DreamRunList runs={status.runs} />
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
