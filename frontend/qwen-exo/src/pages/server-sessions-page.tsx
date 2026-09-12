import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
} from "react";
import {
  Check,
  ChevronDown,
  Inbox,
  MoreHorizontal,
  RefreshCw,
  Search,
  Trash2,
  X,
} from "lucide-react";
import { PageHeader } from "@/components/page-header";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/empty-state";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { deleteServerSessions, listServerSessions } from "@/lib/api";
import { translate, useI18n } from "@/lib/i18n";
import type { ServerSessionDeletion, ServerSessionListing } from "@/lib/types";
import { formatBytes, formatNumber, formatTime } from "@/lib/utils";

const PAGE_SIZE = 25;
const MAX_SELECTION = 1000;
type DeleteIntent = { all: true } | { conversation_keys: string[] };

export function ServerSessionsPage() {
  const { t } = useI18n();
  const [listing, setListing] = useState<ServerSessionListing | null>(null);
  const [query, setQuery] = useState("");
  const [input, setInput] = useState("");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [result, setResult] = useState<ServerSessionDeletion | null>(null);
  const [intent, setIntent] = useState<DeleteIntent | null>(null);
  const [deleting, setDeleting] = useState(false);
  const request = useRef<AbortController | null>(null);
  const mutation = useRef(false);
  const mounted = useRef(false);
  const headerCheckbox = useRef<HTMLInputElement | null>(null);
  const sessions = listing?.sessions ?? [];
  const offset = listing?.offset ?? 0;

  const load = useCallback(async (nextOffset: number, nextQuery: string) => {
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setLoading(true);
    setError(null);
    try {
      let page = await listServerSessions(
        PAGE_SIZE,
        nextOffset,
        nextQuery,
        controller.signal,
      );
      if (controller.signal.aborted) return;
      // Deletions in another tab can empty the last page, too.
      while (page.offset > 0 && !page.sessions.length) {
        const lastOffset =
          Math.max(0, Math.ceil(page.total / PAGE_SIZE) - 1) * PAGE_SIZE;
        page = await listServerSessions(
          PAGE_SIZE,
          Math.min(lastOffset, page.offset - PAGE_SIZE),
          nextQuery,
          controller.signal,
        );
        if (controller.signal.aborted) return;
      }
      setListing(page);
      setQuery(nextQuery);
      setSelected((previous) => {
        const next = new Set(previous);
        for (const row of page.sessions) {
          if (row.active) next.delete(row.conversation_key);
        }
        return next;
      });
    } catch (failure) {
      if (!controller.signal.aborted)
        setError(
          failure instanceof Error ? failure.message : translate("未知错误"),
        );
    } finally {
      if (!controller.signal.aborted) setLoading(false);
    }
  }, []);

  useEffect(() => {
    mounted.current = true;
    void load(0, "");
    return () => {
      mounted.current = false;
      request.current?.abort();
    };
  }, [load]);

  const pageKeys = useMemo(
    () =>
      (listing?.sessions ?? [])
        .filter((row) => !row.active)
        .map((row) => row.conversation_key),
    [listing],
  );
  const selectedOnPage = pageKeys.filter((key) => selected.has(key)).length;
  const allPageSelected =
    pageKeys.length > 0 && selectedOnPage === pageKeys.length;
  const locked = loading || deleting || intent !== null;
  const canSelectPage =
    allPageSelected ||
    selected.size + pageKeys.length - selectedOnPage <= MAX_SELECTION;
  useEffect(() => {
    if (headerCheckbox.current)
      headerCheckbox.current.indeterminate =
        selectedOnPage > 0 && !allPageSelected;
  }, [selectedOnPage, allPageSelected]);

  function submitSearch(event: FormEvent) {
    event.preventDefault();
    if (!locked) void load(0, input.trim());
  }

  function toggle(key: string) {
    setSelected((previous) => {
      const next = new Set(previous);
      if (next.has(key)) next.delete(key);
      else if (next.size < MAX_SELECTION) next.add(key);
      return next;
    });
  }

  function togglePage() {
    setSelected((previous) => {
      const next = new Set(previous);
      if (allPageSelected) pageKeys.forEach((key) => next.delete(key));
      else
        pageKeys.forEach((key) => {
          if (next.size < MAX_SELECTION) next.add(key);
        });
      return next;
    });
  }

  function openDelete(next: DeleteIntent) {
    setDeleteError(null);
    setIntent(next);
  }

  async function confirmDelete() {
    if (!intent || mutation.current) return;
    mutation.current = true;
    setDeleting(true);
    setDeleteError(null);
    request.current?.abort();
    try {
      const outcome = await deleteServerSessions(intent);
      if (!mounted.current) return;
      setResult(outcome);
      setIntent(null);
      const removed = new Set([...outcome.deleted, ...outcome.missing]);
      setSelected(
        (previous) => new Set([...previous].filter((key) => !removed.has(key))),
      );
      setListing(
        (previous) =>
          previous && {
            ...previous,
            sessions: previous.sessions.filter(
              (row) => !removed.has(row.conversation_key),
            ),
          },
      );
      await load(offset, query);
    } catch (failure) {
      if (mounted.current)
        setDeleteError(
          failure instanceof Error ? failure.message : t("删除失败"),
        );
    } finally {
      mutation.current = false;
      if (mounted.current) setDeleting(false);
    }
  }

  const scope = t(
    "删除会移除诊断与重新反思的保存来源证据；不会删除已发布 Knowledge/Reflection、人格、Tensor Bank 或浏览器聊天记录。忙碌会话会被跳过。",
  );

  return (
    <div className="page-frame space-y-5">
      <PageHeader
        title={t("服务器会话")}
        description={t("查看和整理服务器保留的对话记录。")}
        actions={
          <Button
            variant="outline"
            onClick={() => void load(offset, query)}
            disabled={locked}
          >
            <RefreshCw className="mr-2 h-4 w-4" />
            {t("刷新")}
          </Button>
        }
      />
      <div className="overflow-hidden rounded-lg border bg-card">
        <div className="flex flex-col gap-3 border-b p-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-2 text-sm font-medium">
            {t("会话记录")}
            {listing && (
              <span className="rounded-md bg-muted px-2 py-0.5 font-mono text-xs tabular-nums text-muted-foreground">
                {formatNumber(listing.total)}
              </span>
            )}
          </div>
          <div className="flex min-w-0 items-center gap-2">
            <form
              onSubmit={submitSearch}
              className="relative flex min-w-0 flex-1 sm:w-72"
            >
              <Search className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                value={input}
                onChange={(event) => setInput(event.target.value)}
                aria-label={t("搜索会话 ID")}
                placeholder={t("搜索会话 ID")}
                className="pl-9 pr-12"
                disabled={deleting || intent !== null}
              />
              <button
                type="submit"
                aria-label={t("搜索")}
                disabled={locked}
                className="absolute right-1 top-1 rounded px-2 py-1 text-xs text-muted-foreground hover:bg-muted disabled:opacity-40"
              >
                {t("搜索")}
              </button>
            </form>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label={t("会话操作")}
                  disabled={locked || !listing}
                >
                  <MoreHorizontal className="h-4 w-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem
                  disabled={!listing || (listing.total === 0 && !query)}
                  className="text-destructive focus:text-destructive"
                  onSelect={() => openDelete({ all: true })}
                >
                  <Trash2 className="mr-2 h-4 w-4" />
                  {t("清空全部服务器会话")}
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
        {selected.size > 0 && (
          <div className="flex flex-wrap items-center justify-between gap-2 border-b bg-muted/40 px-4 py-2">
            <span className="text-sm">
              {t("已选择 {count} 个会话", { count: selected.size })}
            </span>
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setSelected(new Set())}
                disabled={locked}
              >
                {t("清除选择")}
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="text-destructive hover:text-destructive"
                onClick={() => openDelete({ conversation_keys: [...selected] })}
                disabled={locked}
              >
                <Trash2 className="mr-2 h-3.5 w-3.5" />
                {t("删除选中 ({count})", { count: selected.size })}
              </Button>
            </div>
          </div>
        )}
        {result && (
          <div
            role="status"
            className="flex items-center gap-2 border-b px-4 py-3 text-xs text-muted-foreground"
          >
            <Check className="h-4 w-4 shrink-0" />
            <span className="flex-1">
              {t(
                "已删除 {deleted} 个；跳过忙碌会话 {skipped} 个；已不存在 {missing} 个。",
                {
                  deleted: result.deleted_count,
                  skipped: result.skipped_active.length,
                  missing: result.missing.length,
                },
              )}
            </span>
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6"
              aria-label={t("关闭")}
              onClick={() => setResult(null)}
            >
              <X className="h-3.5 w-3.5" />
            </Button>
          </div>
        )}
        {error && (
          <div
            role="alert"
            className="rounded-md border border-destructive p-3 text-sm text-destructive"
          >
            {t("会话列表加载失败；保留上次列表与选择，请刷新重试。")} {error}
          </div>
        )}
        <div aria-busy={loading}>
          {sessions.length > 0 ? (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-10">
                    <input
                      ref={headerCheckbox}
                      className="h-4 w-4 align-middle accent-current"
                      type="checkbox"
                      aria-label={t("选择本页可删除会话")}
                      checked={allPageSelected}
                      onChange={togglePage}
                      disabled={locked || !pageKeys.length || !canSelectPage}
                    />
                  </TableHead>
                  <TableHead>{t("会话 ID")}</TableHead>
                  <TableHead>{t("更新时间")}</TableHead>
                  <TableHead>{t("事件数")}</TableHead>
                  <TableHead>{t("保存大小估计")}</TableHead>
                  <TableHead>{t("快照数")}</TableHead>
                  <TableHead>{t("状态")}</TableHead>
                  <TableHead>
                    <span className="sr-only">{t("删除")}</span>
                  </TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {sessions.map((row) => {
                  const busy = row.active;
                  return (
                    <TableRow key={row.conversation_key}>
                      <TableCell>
                        <input
                          className="h-4 w-4 align-middle accent-current"
                          type="checkbox"
                          aria-label={t("选择会话 {id}", {
                            id: row.conversation_key,
                          })}
                          checked={selected.has(row.conversation_key)}
                          onChange={() => toggle(row.conversation_key)}
                          disabled={
                            locked ||
                            busy ||
                            (!selected.has(row.conversation_key) &&
                              selected.size >= MAX_SELECTION)
                          }
                        />
                      </TableCell>
                      <TableCell className="min-w-48 max-w-80 py-4 font-mono text-xs">
                        <span
                          className="block truncate"
                          title={row.conversation_key}
                        >
                          {row.conversation_key}
                        </span>
                      </TableCell>
                      <TableCell className="whitespace-nowrap">
                        {row.updated_at > 0 ? formatTime(row.updated_at) : "—"}
                      </TableCell>
                      <TableCell>{formatNumber(row.event_count)}</TableCell>
                      <TableCell>{formatBytes(row.raw_bytes)}</TableCell>
                      <TableCell>{formatNumber(row.source_count)}</TableCell>
                      <TableCell>
                        <div className="flex flex-wrap gap-1">
                          {row.active && (
                            <Badge variant="warning">{t("忙碌")}</Badge>
                          )}
                          {row.pending_reflection && (
                            <Badge variant="outline">{t("等待反思")}</Badge>
                          )}
                        </div>
                      </TableCell>
                      <TableCell>
                        <Button
                          variant="ghost"
                          size="sm"
                          aria-label={t("删除会话 {id}", {
                            id: row.conversation_key,
                          })}
                          disabled={locked || busy}
                          onClick={() =>
                            openDelete({
                              conversation_keys: [row.conversation_key],
                            })
                          }
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          ) : (
            <div className="[&>div]:min-h-80 [&>div]:border-0 [&>div]:bg-transparent">
              <EmptyState
                icon={query ? Search : Inbox}
                title={
                  loading
                    ? t("正在读取")
                    : error
                      ? t("会话列表不可用")
                      : query
                        ? t("没有匹配的服务器会话")
                        : t("暂无会话记录")
                }
                description={
                  query
                    ? t("换一个会话 ID 试试，或清除搜索查看全部记录。")
                    : t("服务器保存的对话会显示在这里，方便查找和清理。")
                }
                actionLabel={query && !loading ? t("清除搜索") : undefined}
                onAction={() => {
                  setInput("");
                  void load(0, "");
                }}
              />
            </div>
          )}
        </div>
        {listing && listing.total > 0 && (
          <nav
            aria-label={t("服务器会话分页")}
            className="flex flex-wrap items-center justify-between gap-3 border-t px-4 py-3"
          >
            <span className="text-xs text-muted-foreground">
              {listing
                ? t("第 {start}–{end} 条，共 {total} 条", {
                    start: sessions.length ? offset + 1 : 0,
                    end: offset + sessions.length,
                    total: listing.total,
                  })
                : "—"}
              {loading && ` · ${t("正在读取")}`}
            </span>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                disabled={locked || offset === 0}
                onClick={() =>
                  void load(Math.max(0, offset - PAGE_SIZE), query)
                }
              >
                {t("上一页")}
              </Button>
              <Button
                variant="outline"
                size="sm"
                disabled={
                  locked || !listing || offset + listing.limit >= listing.total
                }
                onClick={() => void load(offset + PAGE_SIZE, query)}
              >
                {t("下一页")}
              </Button>
            </div>
          </nav>
        )}
      </div>
      <details className="group text-xs text-muted-foreground">
        <summary className="flex w-fit cursor-pointer list-none items-center gap-1.5 hover:text-foreground">
          <ChevronDown className="h-3.5 w-3.5 transition-transform group-open:rotate-180" />
          {t("保存与删除说明")}
        </summary>
        <div className="mt-3 max-w-3xl space-y-2 pl-5 leading-6">
          <p>{scope}</p>
          <p>
            {t(
              "这不是遥测或备份的完整隐私清除，也不影响正在服务的 Responses 状态。后续客户端请求可能重新保存历史。",
            )}
          </p>
          <p>
            {t(
              "保存大小是保留载荷的合计估计，不是数据库磁盘占用；事件数以事件日志为准，无日志时使用保留记录数。",
            )}
          </p>
          <p>
            {t(
              "选择跨页保留，最多 {count} 个；清空全部不受当前分页或搜索条件限制。",
              { count: MAX_SELECTION },
            )}
          </p>
        </div>
      </details>
      <Dialog
        open={intent !== null}
        onOpenChange={(open) => {
          if (!open && !mutation.current) setIntent(null);
        }}
      >
        <DialogContent className="max-h-[85vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>{t("确认不可逆删除")}</DialogTitle>
            <DialogDescription>
              {intent && "all" in intent
                ? t(
                    "这将删除服务器保留的全部会话来源（忙碌会话会跳过），不受当前页面或搜索条件限制。",
                  )
                : t(
                    "这将不可逆地删除选中的 {count} 个会话来源，包括其他页面上的选择。",
                    {
                      count:
                        intent && "conversation_keys" in intent
                          ? intent.conversation_keys.length
                          : 0,
                    },
                  )}
            </DialogDescription>
          </DialogHeader>
          {intent && "conversation_keys" in intent && (
            <ul className="max-h-40 space-y-1 overflow-y-auto rounded-md border p-3 font-mono text-xs">
              {intent.conversation_keys.map((key) => (
                <li key={key} className="break-all">
                  {key}
                </li>
              ))}
            </ul>
          )}
          <p className="text-sm text-muted-foreground">{scope}</p>
          <p className="text-sm text-muted-foreground">
            {t(
              "这不是遥测或备份的完整隐私清除，也不影响正在服务的 Responses 状态。后续客户端请求可能重新保存历史。",
            )}
          </p>
          {deleteError && (
            <p role="alert" className="text-sm text-destructive">
              {t(
                "删除未确认；保留列表与选择。请求可能已到达服务器，请关闭对话框并刷新以核对状态。",
              )}{" "}
              {deleteError}
            </p>
          )}
          <DialogFooter>
            <Button
              variant="outline"
              disabled={deleting}
              onClick={() => setIntent(null)}
            >
              {t("取消")}
            </Button>
            <Button
              variant="destructive"
              disabled={deleting || deleteError !== null}
              onClick={() => void confirmDelete()}
            >
              {deleting ? t("正在删除") : t("确认删除")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
