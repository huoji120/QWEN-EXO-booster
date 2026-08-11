import { useEffect, useMemo, useState } from "react";
import { Clock3, LoaderCircle, Play, RefreshCw, Search, X } from "lucide-react";
import { toast } from "sonner";
import { EmptyState } from "@/components/empty-state";
import { PageHeader } from "@/components/page-header";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  ApiError,
  cancelPendingReflections,
  listPendingReflectionMemories,
  startPendingReflections,
} from "@/lib/api";
import { translate as t } from "@/lib/i18n";
import type { PendingReflectionMemory } from "@/lib/types";
import { formatNumber, formatTime } from "@/lib/utils";

function remainingLabel(item: PendingReflectionMemory, now: number) {
  if (item.status === "running") return t("整理中");
  const seconds = Math.max(0, Math.ceil(item.due_at - now));
  if (seconds < 60) return t("{count} 秒", { count: formatNumber(seconds) });
  const minutes = Math.floor(seconds / 60);
  const rest = seconds % 60;
  return rest
    ? t("{minutes} 分 {seconds} 秒", {
        minutes: formatNumber(minutes),
        seconds: formatNumber(rest),
      })
    : t("{count} 分", { count: formatNumber(minutes) });
}

export function ReflectionPage() {
  const [items, setItems] = useState<PendingReflectionMemory[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [available, setAvailable] = useState(true);
  const [action, setAction] = useState<"start" | "cancel" | null>(null);
  const [now, setNow] = useState(() => Date.now() / 1000);
  const [summaryConversationKey, setSummaryConversationKey] = useState<
    string | null
  >(null);

  const load = async (silent = false) => {
    if (!silent) setLoading(true);
    try {
      const result = await listPendingReflectionMemories();
      setAvailable(true);
      setItems(result.pending || []);
      const available = new Set(
        (result.pending || []).map((item) => item.conversation_key),
      );
      setSelected(
        (current) => new Set([...current].filter((key) => available.has(key))),
      );
    } catch (error) {
      if (error instanceof ApiError && error.status === 404) {
        setAvailable(false);
        setItems([]);
        setSelected(new Set());
      } else if (!silent) {
        toast.error(t("待反思轨迹加载失败"), {
          description: error instanceof Error ? error.message : t("未知错误"),
        });
      }
    } finally {
      if (!silent) setLoading(false);
    }
  };

  useEffect(() => {
    void load();
    const refreshTimer = window.setInterval(() => void load(true), 2000);
    const clockTimer = window.setInterval(
      () => setNow(Date.now() / 1000),
      1000,
    );
    return () => {
      window.clearInterval(refreshTimer);
      window.clearInterval(clockTimer);
    };
  }, []);

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return items;
    return items.filter((item) =>
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
  }, [items, query]);

  const visibleKeys = visible.map((item) => item.conversation_key);
  const allVisibleSelected =
    visibleKeys.length > 0 && visibleKeys.every((key) => selected.has(key));
  const someVisibleSelected = visibleKeys.some((key) => selected.has(key));
  const selectedItems = items.filter((item) =>
    selected.has(item.conversation_key),
  );
  const selectedHasRunning = selectedItems.some(
    (item) => item.status === "running",
  );
  const summaryItem = items.find(
    (item) => item.conversation_key === summaryConversationKey,
  );

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

  const startNow = async (keys: string[]) => {
    if (!keys.length) return;
    setAction("start");
    try {
      const result = await startPendingReflections(keys);
      setSelected(new Set());
      toast.success(
        t("已开始 {count} 条反思", {
          count: formatNumber(result.started_count),
        }),
      );
      await load(true);
    } catch (error) {
      toast.error(t("立即反思失败"), {
        description: error instanceof Error ? error.message : t("未知错误"),
      });
    } finally {
      setAction(null);
    }
  };

  const cancel = async (keys: string[]) => {
    if (!keys.length) return;
    setAction("cancel");
    try {
      const result = await cancelPendingReflections(keys);
      setSelected(new Set());
      toast.success(
        t("已取消 {count} 条反思", {
          count: formatNumber(result.cancelled_count),
        }),
      );
      await load(true);
    } catch (error) {
      toast.error(t("取消反思失败"), {
        description: error instanceof Error ? error.message : t("未知错误"),
      });
    } finally {
      setAction(null);
    }
  };

  return (
    <div className="page-frame">
      <PageHeader
        title={t("反思队列")}
        actions={
          <Button variant="outline" size="sm" onClick={() => void load()}>
            <RefreshCw />
            {t("刷新")}
          </Button>
        }
      />

      <div className="mb-4 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div className="relative w-full sm:w-80">
          <Search className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
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
            disabled={!selected.size || selectedHasRunning || action !== null}
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
          {visible.length ? (
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
                  <TableHead className="w-40 text-right">{t("操作")}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {visible.map((item) => (
                  <TableRow key={item.conversation_key}>
                    <TableCell>
                      <input
                        type="checkbox"
                        aria-label={t("选择 {id}", { id: item.trajectory_id })}
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
                        className="line-clamp-2 w-full max-w-64 rounded-sm text-left text-sm font-medium leading-5 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                        onClick={() =>
                          setSummaryConversationKey(item.conversation_key)
                        }
                      >
                        {item.original_task || t("未命名任务")}
                      </button>
                    </TableCell>
                    <TableCell>
                      <Badge
                        variant={
                          item.status === "running" ? "default" : "outline"
                        }
                      >
                        {item.status === "running" ? t("整理中") : t("等待")}
                      </Badge>
                    </TableCell>
                    <TableCell className="font-mono text-xs text-muted-foreground">
                      {formatTime(item.last_activity_at)}
                    </TableCell>
                    <TableCell className="font-mono text-xs">
                      {remainingLabel(item, now)}
                    </TableCell>
                    <TableCell className="text-xs text-muted-foreground">
                      <div>{formatNumber(item.source_token_count)} tokens</div>
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
                          onClick={() => void startNow([item.conversation_key])}
                        >
                          <Play />
                          {t("立即")}
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          disabled={action !== null}
                          onClick={() => void cancel([item.conversation_key])}
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
                !available
                  ? t("服务重启后启用")
                  : loading
                    ? t("正在读取反思队列")
                    : query
                      ? t("没有匹配轨迹")
                      : t("暂无待反思轨迹")
              }
              description={
                !available
                  ? t("后端源码已部署；当前进程仍保留旧接口。")
                  : loading
                    ? t("正在同步服务端状态。")
                    : query
                      ? t("清除搜索条件后重试。")
                      : t("新轨迹满足反思条件后会出现在这里。")
              }
            />
          )}
        </CardContent>
      </Card>

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
          <div className="max-h-[60vh] overflow-y-auto overflow-x-hidden rounded-md border bg-muted/30 p-4">
            <p className="whitespace-pre-wrap break-words text-sm leading-6">
              {summaryItem?.original_task || t("未命名任务")}
            </p>
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
