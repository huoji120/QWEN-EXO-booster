import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { FileText, FolderPlus, RefreshCw } from "lucide-react";
import { toast } from "sonner";
import { EmptyState } from "@/components/empty-state";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import {
  createKnowledgeFolder,
  deleteKnowledgeFile,
  deleteKnowledgeFolder,
  generateKnowledgeSummaries,
  getKnowledgeTree,
  getSummaryStatus,
  moveKnowledgeEntry,
} from "@/lib/api";
import { translate as t } from "@/lib/i18n";
import type { KnowledgeFile, KnowledgeSummaryStatus, KnowledgeTree } from "@/lib/types";
import { formatBytes, formatNumber } from "@/lib/utils";
import {
  KnowledgeDeleteDialog,
  KnowledgeEntryDialog,
  type DeleteRequest,
  type EntryRequest,
} from "@/components/knowledge/knowledge-entry-dialogs";
import { KnowledgeFileTable, type FileAction } from "@/components/knowledge/knowledge-file-table";
import {
  KnowledgeFolderTree,
  type FolderAction,
} from "@/components/knowledge/knowledge-folder-tree";
import { KnowledgeSearchCard } from "@/components/knowledge/knowledge-search-card";
import { KnowledgeSummaryDialog } from "@/components/knowledge/knowledge-summary-dialog";
import { KnowledgeSummaryToolbar } from "@/components/knowledge/knowledge-summary-toolbar";
import { KnowledgeUploadCard } from "@/components/knowledge/knowledge-upload-card";
import { useKnowledgeViewer } from "@/components/knowledge/knowledge-viewer-dialog";
import {
  buildFolderTree,
  errorText,
  isWithin,
  parentOf,
  rebasePath,
  summaryView,
  type SummaryView,
} from "@/components/knowledge/knowledge-utils";

const BUSY_POLL_MS = 5_000;
const IDLE_POLL_MS = 30_000;

export function FileKnowledgePanel() {
  const [tree, setTree] = useState<KnowledgeTree>({ directories: [], files: [] });
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [folder, setFolder] = useState("");
  const [nested, setNested] = useState(true);
  const [nameFilter, setNameFilter] = useState("");
  const [selected, setSelected] = useState<Set<string>>(() => new Set());

  const [status, setStatus] = useState<KnowledgeSummaryStatus | null>(null);
  const [generating, setGenerating] = useState(false);

  const [entryRequest, setEntryRequest] = useState<EntryRequest | null>(null);
  const [deleteRequest, setDeleteRequest] = useState<DeleteRequest | null>(null);
  const [summaryPath, setSummaryPath] = useState<string | null>(null);
  const [searchResetKey, setSearchResetKey] = useState(0);

  const viewer = useKnowledgeViewer();

  const load = useCallback(async (silent = false) => {
    if (!silent) setLoading(true);
    try {
      setTree(await getKnowledgeTree());
      setLoadError(null);
    } catch (error) {
      if (!silent) setLoadError(errorText(error, t("未知错误")));
    } finally {
      if (!silent) setLoading(false);
    }
  }, []);

  // Summary worker status; reload the tree whenever a file finishes.
  const lastRun = useRef<{ current: string | null; completed: number; failed: number } | null>(
    null,
  );
  const refreshStatus = useCallback(async () => {
    try {
      const next = await getSummaryStatus();
      setStatus(next);
      const run = {
        current: next.current?.path ?? null,
        completed: next.completed ?? 0,
        failed: next.failed?.length ?? 0,
      };
      const previous = lastRun.current;
      lastRun.current = run;
      if (
        previous &&
        ((previous.current && previous.current !== run.current) ||
          previous.completed !== run.completed ||
          previous.failed !== run.failed)
      ) {
        void load(true);
      }
    } catch {
      // Keep the last known status; the next poll retries.
    }
  }, [load]);

  useEffect(() => {
    void load();
    void refreshStatus();
  }, [load, refreshStatus]);

  const summaryBusy = Boolean(status?.current) || Boolean(status?.queued?.length);
  useEffect(() => {
    const timer = window.setInterval(
      () => void refreshStatus(),
      summaryBusy ? BUSY_POLL_MS : IDLE_POLL_MS,
    );
    const onFocus = () => void refreshStatus();
    window.addEventListener("focus", onFocus);
    return () => {
      window.clearInterval(timer);
      window.removeEventListener("focus", onFocus);
    };
  }, [refreshStatus, summaryBusy]);

  const { root, folders } = useMemo(
    () => buildFolderTree(tree.directories, tree.files),
    [tree],
  );

  // Drop a selected folder / checked files that no longer exist.
  // Runs on tree changes only, so a just-renamed selection survives until reload.
  useEffect(() => {
    const known = new Set(folders);
    setFolder((current) => {
      let next = current;
      while (next && !known.has(next)) next = parentOf(next);
      return next;
    });
  }, [folders]);
  useEffect(() => {
    setSelected((previous) => {
      if (!previous.size) return previous;
      const existing = new Set(tree.files.map((file) => file.relative_path));
      const next = new Set(Array.from(previous).filter((path) => existing.has(path)));
      return next.size === previous.size ? previous : next;
    });
  }, [tree.files]);

  const visibleFiles = useMemo(() => {
    const needle = nameFilter.trim().toLowerCase();
    return tree.files.filter((file) => {
      const path = file.relative_path;
      if (nested ? !isWithin(path, folder) : parentOf(path) !== folder) return false;
      if (!needle) return true;
      return (
        path.toLowerCase().includes(needle) || file.title.toLowerCase().includes(needle)
      );
    });
  }, [tree.files, folder, nested, nameFilter]);

  const views = useMemo(() => {
    const failures = new Map((status?.failed ?? []).map((item) => [item.path, item.error]));
    const queued = new Set(status?.queued ?? []);
    const map = new Map<string, SummaryView>();
    for (const file of tree.files) {
      map.set(file.relative_path, summaryView(file, status, failures, queued));
    }
    return map;
  }, [tree.files, status]);

  const totalBytes = tree.files.reduce((sum, file) => sum + file.byte_count, 0);
  const folderBytes = visibleFiles.reduce((sum, file) => sum + file.byte_count, 0);

  const reportError = (title: string, error: unknown): never => {
    toast.error(title, { description: errorText(error, t("未知错误")) });
    throw error;
  };

  const generate = async (paths: string[], force = false) => {
    if (generating) return;
    setGenerating(true);
    try {
      const next = await generateKnowledgeSummaries(paths, force);
      setStatus(next);
      toast.success(
        paths.length
          ? t("已将 {count} 个文件加入总结队列", { count: paths.length })
          : t("已开始生成缺失的总结"),
      );
    } catch (error) {
      toast.error(t("生成总结失败"), { description: errorText(error, t("未知错误")) });
    } finally {
      setGenerating(false);
    }
  };

  const submitEntry = async (request: EntryRequest, destination: string) => {
    if (request.kind === "create") {
      try {
        const result = await createKnowledgeFolder(destination);
        toast.success(t("文件夹已创建"));
        await load(true);
        setFolder(result.path || destination);
      } catch (error) {
        reportError(t("创建文件夹失败"), error);
      }
      return;
    }
    try {
      const result = await moveKnowledgeEntry(request.path, destination);
      const target = result.to || destination;
      toast.success(
        request.kind === "rename" ? t("已重命名") : t("已移动"),
        {
          description: request.isFolder
            ? t("{to}（{count} 个文件）", { to: target, count: result.files })
            : target,
        },
      );
      setFolder((current) => rebasePath(current, request.path, target));
      setSelected(
        (previous) =>
          new Set(Array.from(previous).map((path) => rebasePath(path, request.path, target))),
      );
      setSummaryPath((current) => (current ? rebasePath(current, request.path, target) : current));
      await load(true);
    } catch (error) {
      reportError(request.kind === "rename" ? t("重命名失败") : t("移动失败"), error);
    }
  };

  const confirmDelete = async (request: DeleteRequest) => {
    try {
      if (request.kind === "folder") {
        const result = await deleteKnowledgeFolder(request.path);
        toast.success(t("文件夹已删除"), {
          description: t("共删除 {count} 个文件", { count: result.files }),
        });
        if (isWithin(folder, request.path)) setFolder(parentOf(request.path));
      } else {
        await deleteKnowledgeFile(request.path);
        toast.success(t("知识库文件已删除"));
      }
      await load(true);
      void refreshStatus();
    } catch (error) {
      reportError(t("删除失败"), error);
    }
  };

  const onFolderAction = (action: FolderAction, path: string) => {
    if (action === "create") {
      setEntryRequest({ kind: "create", path, isFolder: true });
    } else if (action === "delete") {
      setDeleteRequest({
        kind: "folder",
        path,
        fileCount: tree.files.filter((file) => isWithin(file.relative_path, path)).length,
        folderCount: folders.filter((item) => item.startsWith(`${path}/`)).length,
      });
    } else {
      setEntryRequest({ kind: action, path, isFolder: true });
    }
  };

  const onFileAction = (action: FileAction, file: KnowledgeFile) => {
    const path = file.relative_path;
    if (action === "view") void viewer.open(path);
    else if (action === "summary") setSummaryPath(path);
    else if (action === "generate") void generate([path], true);
    else if (action === "delete") setDeleteRequest({ kind: "file", path });
    else setEntryRequest({ kind: action, path, isFolder: false });
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant="secondary">
            {t("{count} 个文件", { count: formatNumber(tree.files.length) })}
          </Badge>
          <Badge variant="secondary">{formatBytes(totalBytes)}</Badge>
          <Badge variant="secondary">
            {t("{count} 个文件夹", { count: formatNumber(folders.length) })}
          </Badge>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            void load();
            void refreshStatus();
          }}
        >
          <RefreshCw />
          {t("刷新")}
        </Button>
      </div>

      <KnowledgeUploadCard
        targetDirectory={folder}
        onImported={async () => {
          setSearchResetKey((key) => key + 1);
          await load(true);
          void refreshStatus();
        }}
      />

      <KnowledgeSearchCard resetKey={searchResetKey} onOpen={(path) => void viewer.open(path)} />

      <Card>
        <CardContent className="p-0">
          {loading ? (
            <div className="space-y-2 p-5">
              {Array.from({ length: 4 }).map((_, index) => (
                <Skeleton key={index} className="h-10 w-full" />
              ))}
            </div>
          ) : loadError ? (
            <div className="p-5">
              <p className="text-sm font-medium">{t("知识库加载失败")}</p>
              <p className="mt-2 break-words text-xs text-muted-foreground">{loadError}</p>
              <Button className="mt-4" variant="outline" onClick={() => void load()}>
                <RefreshCw />
                {t("重试")}
              </Button>
            </div>
          ) : (
            <div className="grid lg:grid-cols-[17rem_minmax(0,1fr)]">
              <aside className="border-b p-3 lg:border-b-0 lg:border-r">
                <div className="mb-2 flex items-center justify-between gap-2">
                  <span className="text-xs font-medium text-muted-foreground">
                    {t("文件夹")}
                  </span>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setEntryRequest({ kind: "create", path: folder, isFolder: true })}
                  >
                    <FolderPlus />
                    {t("新建文件夹")}
                  </Button>
                </div>
                <div className="max-h-[60vh] overflow-auto">
                  <KnowledgeFolderTree
                    root={root}
                    selected={folder}
                    onSelect={setFolder}
                    onAction={onFolderAction}
                  />
                </div>
              </aside>
              <section className="min-w-0 space-y-3 p-3">
                <div className="flex flex-wrap items-center gap-2">
                  <span
                    className="min-w-0 truncate font-mono text-sm"
                    title={folder || t("全部记忆")}
                  >
                    {folder ? `${folder}/` : t("全部记忆")}
                  </span>
                  <Badge variant="outline">
                    {t("{count} 个文件", { count: formatNumber(visibleFiles.length) })}
                    {" · "}
                    {formatBytes(folderBytes)}
                  </Badge>
                  <label className="flex items-center gap-2 text-xs text-muted-foreground">
                    <input
                      type="checkbox"
                      checked={nested}
                      onChange={(event) => setNested(event.target.checked)}
                      className="h-4 w-4 accent-primary"
                    />
                    {t("包含子文件夹")}
                  </label>
                  <Input
                    className="ml-auto h-8 w-full sm:w-56"
                    value={nameFilter}
                    placeholder={t("按名称或标题筛选")}
                    onChange={(event) => setNameFilter(event.target.value)}
                  />
                </div>
                <KnowledgeSummaryToolbar
                  status={status}
                  selectedCount={selected.size}
                  busy={generating}
                  onGenerateMissing={() => void generate([])}
                  onGenerateSelected={() => void generate(Array.from(selected))}
                  onClearSelection={() => setSelected(new Set())}
                />
                {visibleFiles.length ? (
                  <div className="rounded-md border">
                    <KnowledgeFileTable
                      files={visibleFiles}
                      folder={folder}
                      views={views}
                      selected={selected}
                      viewerLoading={viewer.loading}
                      onToggle={(path, checked) =>
                        setSelected((previous) => {
                          const next = new Set(previous);
                          if (checked) next.add(path);
                          else next.delete(path);
                          return next;
                        })
                      }
                      onToggleAll={(checked) =>
                        setSelected((previous) => {
                          const next = new Set(previous);
                          for (const file of visibleFiles) {
                            if (checked) next.add(file.relative_path);
                            else next.delete(file.relative_path);
                          }
                          return next;
                        })
                      }
                      onAction={onFileAction}
                    />
                  </div>
                ) : tree.files.length ? (
                  <p className="py-8 text-center text-sm text-muted-foreground">
                    {nameFilter.trim() ? t("没有匹配的文件") : t("这个文件夹里还没有文件")}
                  </p>
                ) : (
                  <EmptyState
                    icon={FileText}
                    title={t("知识库还是空的")}
                    description={t(
                      "导入文本文件后，模型会通过服务端工具 qwen_exo_memory_search / qwen_exo_memory_read 按需搜索和读取原文。",
                    )}
                  />
                )}
              </section>
            </div>
          )}
        </CardContent>
      </Card>

      {viewer.dialog}

      <KnowledgeSummaryDialog
        path={summaryPath}
        view={summaryPath ? views.get(summaryPath) ?? null : null}
        onClose={() => setSummaryPath(null)}
        onChanged={() => {
          void load(true);
          void refreshStatus();
        }}
      />

      <KnowledgeEntryDialog
        key={entryRequest ? `${entryRequest.kind}:${entryRequest.path}` : "closed"}
        request={entryRequest}
        folders={folders}
        onClose={() => setEntryRequest(null)}
        onSubmit={submitEntry}
      />

      <KnowledgeDeleteDialog
        request={deleteRequest}
        onClose={() => setDeleteRequest(null)}
        onConfirm={confirmDelete}
      />
    </div>
  );
}
