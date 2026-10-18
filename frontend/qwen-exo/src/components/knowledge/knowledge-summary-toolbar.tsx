import { LoaderCircle, Sparkles, TriangleAlert } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { translate as t } from "@/lib/i18n";
import type { KnowledgeSummaryStatus } from "@/lib/types";
import { formatNumber } from "@/lib/utils";

/** Summary worker progress plus bulk generation actions. */
export function KnowledgeSummaryToolbar({
  status,
  selectedCount,
  busy,
  onGenerateMissing,
  onGenerateSelected,
  onClearSelection,
}: {
  status: KnowledgeSummaryStatus | null;
  selectedCount: number;
  busy: boolean;
  onGenerateMissing: () => void;
  onGenerateSelected: () => void;
  onClearSelection: () => void;
}) {
  const available = status?.available ?? false;
  const current = status?.current ?? null;
  const queued = status?.queued ?? [];
  const failed = status?.failed ?? [];
  const missing = status?.missing ?? 0;

  return (
    <div className="flex flex-col gap-2 rounded-md border bg-muted/30 px-3 py-2 text-xs">
      <div className="flex flex-wrap items-center gap-2">
        <Button
          size="sm"
          variant="outline"
          disabled={!available || busy || missing === 0}
          onClick={onGenerateMissing}
        >
          {busy ? <LoaderCircle className="animate-spin" /> : <Sparkles />}
          {t("生成全部缺失的总结")}
        </Button>
        {selectedCount ? (
          <>
            <Button size="sm" disabled={!available || busy} onClick={onGenerateSelected}>
              <Sparkles />
              {t("为选中文件生成总结（{count}）", { count: formatNumber(selectedCount) })}
            </Button>
            <Button size="sm" variant="ghost" onClick={onClearSelection}>
              {t("取消选择")}
            </Button>
          </>
        ) : null}
      </div>
      {status === null ? null : !available ? (
        <p className="text-muted-foreground">
          {t("总结服务不可用（需要已加载模型的分词器）。")}
        </p>
      ) : (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-muted-foreground">
          {current ? (
            <span className="flex min-w-0 items-center gap-1.5 text-foreground">
              <LoaderCircle className="h-3.5 w-3.5 shrink-0 animate-spin" />
              {t("正在总结")}
              <span className="max-w-[24rem] truncate font-mono" title={current.path}>
                {current.path}
              </span>
              <span className="font-mono">
                {current.done}/{current.chunks}
              </span>
              {current.sampled ? <Badge variant="outline">{t("抽样")}</Badge> : null}
            </span>
          ) : (
            <span>{t("总结空闲")}</span>
          )}
          <span>{t("排队 {count}", { count: formatNumber(queued.length) })}</span>
          <span>{t("缺失 {count}", { count: formatNumber(missing) })}</span>
          <span>
            {t("本次运行已完成 {count}", { count: formatNumber(status.completed ?? 0) })}
          </span>
          {failed.length ? (
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="flex cursor-help items-center gap-1 text-destructive">
                  <TriangleAlert className="h-3.5 w-3.5" />
                  {t("失败 {count}", { count: formatNumber(failed.length) })}
                </span>
              </TooltipTrigger>
              <TooltipContent className="max-w-md">
                <ul className="space-y-1">
                  {failed.slice(0, 10).map((item) => (
                    <li key={item.path} className="break-words">
                      <span className="font-mono">{item.path}</span>：{item.error}
                    </li>
                  ))}
                  {failed.length > 10 ? (
                    <li>{t("另有 {count} 个", { count: failed.length - 10 })}</li>
                  ) : null}
                </ul>
              </TooltipContent>
            </Tooltip>
          ) : null}
        </div>
      )}
    </div>
  );
}
