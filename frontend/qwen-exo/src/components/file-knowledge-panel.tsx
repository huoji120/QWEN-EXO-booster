import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { BookOpen, FileText, FolderPlus, NotebookPen, RefreshCw } from "lucide-react";
import { toast } from "sonner";
import { EmptyState } from "@/components/empty-state";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  createKnowledgeFolder,
  deleteKnowledgeFile,
  deleteKnowledgeFolder,
  generateKnowledgeSummaries,
  getKnowledgeTree,
  getSummaryStatus,
  listKnowledgeFiles,
  moveKnowledgeEntry,
} from "@/lib/api";
import { translate as t } from "@/lib/i18n";
import type {
  KnowledgeDirectoryCount,
  KnowledgeFile,
  KnowledgeFilePage,
  KnowledgeSummaryStatus,
  KnowledgeTree,
} from "@/lib/types";
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
import { KnowledgeDreamCard } from "@/components/knowledge/knowledge-dream-card";
import { KnowledgeWorkTrash } from "@/components/knowledge/knowledge-work-trash";
import { useDreamStatus } from "@/components/knowledge/use-dream-status";
import { useKnowledgeViewer } from "@/components/knowledge/knowledge-viewer-dialog";
import {
  AREA_LABELS,
  areaOf,
  areaRoot,
  buildFolderTree,
  errorText,
  isWithin,
  parentOf,
  rebasePath,
  summaryView,
  type MemoryArea,
  type SummaryView,
} from "@/components/knowledge/knowledge-utils";

const BUSY_POLL_MS = 5_000;
const IDLE_POLL_MS = 30_000;
const PAGE_SIZE = 100;
const EMPTY_COUNT: KnowledgeDirectoryCount = { files: 0, bytes: 0 };
const EMPTY_TREE: KnowledgeTree = {
  directories: [],
  counts: {},
  areas: { imported: EMPTY_COUNT, work: EMPTY_COUNT },
};
const EMPTY_PAGE: KnowledgeFilePage = {
  files: [],
  total: 0,
  total_bytes: 0,
  offset: 0,
  limit: PAGE_SIZE,
};

export function FileKnowledgePanel() {
  const [tree, setTree] = useState<KnowledgeTree>(EMPTY_TREE);
  const [page, setPage] = useState<KnowledgeFilePage>(EMPTY_PAGE);
  const [pageIndex, setPageIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [area, setArea] = useState<MemoryArea>("imported");
  const [folder, setFolder] = useState("");
  // Last selected folder of each area, restored when switching back.
  const areaFolders = useRef<Record<MemoryArea, string>>({
    imported: areaRoot("imported"),
    work: areaRoot("work"),
  });
  const [nested, setNested] = useState(true);
  const [nameFilter, setNameFilter] = useState("");
  const [debouncedFilter, setDebouncedFilter] = useState("");
  const [selected, setSelected] = useState<Set<string>>(() => new Set());

  const [status, setStatus] = useState<KnowledgeSummaryStatus | null>(null);
  const [generating, setGenerating] = useState(false);

  const [entryRequest, setEntryRequest] = useState<EntryRequest | null>(null);
  const [deleteRequest, setDeleteRequest] = useState<DeleteRequest | null>(null);
  const [summaryPath, setSummaryPath] = useState<string | null>(null);
  const [searchResetKey, setSearchResetKey] = useState(0);

  const [trashKey, setTrashKey] = useState(0);

  const viewer = useKnowledgeViewer();

  useEffect(() => {
    const timer = window.setTimeout(() => setDebouncedFilter(nameFilter), 250);
    return () => window.clearTimeout(timer);
  }, [nameFilter]);

  const loadTree = useCallback(async () => {
    const next = await getKnowledgeTree();
    setTree({
      directories: next.directories ?? [],
      counts: next.counts ?? {},
      areas: {
        imported: next.areas?.imported ?? EMPTY_COUNT,
        work: next.areas?.work ?? EMPTY_COUNT,
      },
    });
  }, []);

  const pageEpoch = useRef(0);
  const loadPage = useCallback(
    async (index: number) => {
      const epoch = ++pageEpoch.current;
      try {
        const next = await listKnowledgeFiles({
          folder,
          nested,
          q: debouncedFilter,
          area,
          limit: PAGE_SIZE,
          offset: index * PAGE_SIZE,
        });
        if (epoch !== pageEpoch.current) return;
        const files = Array.isArray(next.files) ? next.files.slice(0, PAGE_SIZE) : [];
        setPage({
          files,
          total: typeof next.total === "number" ? next.total : files.length,
          total_bytes: typeof next.total_bytes === "number" ? next.total_bytes : 0,
          offset: typeof next.offset === "number" ? next.offset : index * PAGE_SIZE,
          limit: PAGE_SIZE,
        });
        setLoadError(null);
      } catch (error) {
        if (epoch === pageEpoch.current) setLoadError(errorText(error, t("未知错误")));
      } finally {
        if (epoch === pageEpoch.current) setLoading(false);
      }
    },
    [area, debouncedFilter, folder, nested],
  );

  const loadPageRef = useRef(loadPage);
  loadPageRef.current = loadPage;
  const pageIndexRef = useRef(pageIndex);
  pageIndexRef.current = pageIndex;

  const reload = useCallback(async () => {
    try {
      await loadTree();
      await loadPage(pageIndex);
    } catch (error) {
      setLoadError(errorText(error, t("未知错误")));
      setLoading(false);
    }
  }, [loadPage, loadTree, pageIndex]);

  const filterKey = `${area}|${folder}|${nested}|${debouncedFilter}`;
  const [seenFilter, setSeenFilter] = useState(filterKey);
  if (seenFilter !== filterKey) {
    setSeenFilter(filterKey);
    if (pageIndex !== 0) setPageIndex(0);
  }

  // Summary worker status; reload the visible page whenever a file finishes.
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
        void loadPageRef.current(pageIndexRef.current);
      }
    } catch {
      // Keep the last known status; the next poll retries.
    }
  }, []);

  const [treeReady, setTreeReady] = useState(false);
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        await loadTree();
        if (!cancelled) {
          setTreeReady(true);
          setLoadError(null);
        }
      } catch (error) {
        if (!cancelled) {
          setLoadError(errorText(error, t("未知错误")));
          setLoading(false);
        }
      }
    })();
    void refreshStatus();
    return () => {
      cancelled = true;
    };
  }, [loadTree, refreshStatus]);

  useEffect(() => {
    if (!treeReady) return;
    void loadPage(pageIndex);
  }, [treeReady, loadPage, pageIndex]);

  // A finished dream run may have written, moved or deleted work notes.
  const dream = useDreamStatus(() => {
    void reload();
    void refreshStatus();
    setTrashKey((key) => key + 1);
  });

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

  const rootPath = areaRoot(area);
  const areaCounts = {
    imported: tree.areas.imported.files,
    work: tree.areas.work.files,
  };
  const { root, folders } = useMemo(() => {
    const built = buildFolderTree(
      tree.directories.filter((path) => areaOf(path) === area),
      [],
      rootPath,
      tree.counts,
    );
    built.root.fileCount = area === "work" ? tree.areas.work.files : tree.areas.imported.files;
    return built;
  }, [area, rootPath, tree.areas.imported.files, tree.areas.work.files, tree.counts, tree.directories]);
  // Move targets may lie in either area.
  const allFolders = useMemo(
    () => buildFolderTree(tree.directories, [], "", tree.counts).folders,
    [tree.counts, tree.directories],
  );

  const switchArea = (next: MemoryArea, nextFolder?: string) => {
    if (next !== area) {
      areaFolders.current[area] = folder;
      setArea(next);
      setSelected(new Set());
    }
    setFolder(nextFolder ?? areaFolders.current[next]);
  };

  /** Select ``path``, switching to the area that owns it. */
  const focusFolder = (path: string) => switchArea(areaOf(path), path);

  // Drop a selected folder / checked files that no longer exist.
  // Runs on tree changes only, so a just-renamed selection survives until reload.
  useEffect(() => {
    const known = new Set(folders);
    setFolder((current) => {
      if (areaOf(current) !== area) return rootPath;
      let next = current;
      while (next && next !== rootPath && !known.has(next)) next = parentOf(next);
      return next;
    });
  }, [folders, area, rootPath]);
  const visibleFiles = page.files;
  const pageCount = Math.max(1, Math.ceil(page.total / PAGE_SIZE));
  const currentPage = Math.min(pageIndex, Math.max(0, pageCount - 1));
  if (pageIndex !== currentPage) setPageIndex(currentPage);
  const pagePaths = useMemo(
    () => new Set(visibleFiles.map((file) => file.relative_path)),
    [visibleFiles],
  );
  useEffect(() => {
    setSelected((previous) => {
      if (!previous.size) return previous;
      const next = new Set(Array.from(previous).filter((path) => pagePaths.has(path)));
      return next.size === previous.size ? previous : next;
    });
  }, [pagePaths]);

  const views = useMemo(() => {
    const failures = new Map((status?.failed ?? []).map((item) => [item.path, item.error]));
    const queued = new Set(status?.queued ?? []);
    const map = new Map<string, SummaryView>();
    for (const file of visibleFiles) {
      map.set(file.relative_path, summaryView(file, status, failures, queued));
    }
    return map;
  }, [status, visibleFiles]);

  const totalBytes = area === "work" ? tree.areas.work.bytes : tree.areas.imported.bytes;
  const folderBytes = page.total_bytes;

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
        await reload();
        focusFolder(result.path || destination);
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
      const rebased = rebasePath(folder, request.path, target);
      if (rebased !== folder) focusFolder(rebased);
      setSelected(
        (previous) =>
          new Set(Array.from(previous).map((path) => rebasePath(path, request.path, target))),
      );
      setSummaryPath((current) => (current ? rebasePath(current, request.path, target) : current));
      await reload();
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
        if (isWithin(folder, request.path)) focusFolder(parentOf(request.path));
      } else {
        await deleteKnowledgeFile(request.path);
        toast.success(t("知识库文件已删除"));
      }
      await reload();
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
        fileCount: tree.counts[path]?.files ?? 0,
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
          <Tabs value={area} onValueChange={(value) => switchArea(value as MemoryArea)}>
            <TabsList>
              <TabsTrigger value="imported" className="gap-1.5">
                <BookOpen className="h-3.5 w-3.5" />
                {t(AREA_LABELS.imported)}
                <span className="font-mono text-[11px] text-muted-foreground">
                  {formatNumber(areaCounts.imported)}
                </span>
              </TabsTrigger>
              <TabsTrigger value="work" className="gap-1.5">
                <NotebookPen className="h-3.5 w-3.5" />
                {t(AREA_LABELS.work)}
                <span className="font-mono text-[11px] text-muted-foreground">
                  {formatNumber(areaCounts.work)}
                </span>
              </TabsTrigger>
            </TabsList>
          </Tabs>
          <Badge variant="secondary">
            {t("{count} 个文件", { count: formatNumber(areaCounts[area]) })}
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
            setLoading(true);
            void reload();
            void refreshStatus();
            void dream.refresh();
            setTrashKey((key) => key + 1);
          }}
        >
          <RefreshCw />
          {t("刷新")}
        </Button>
      </div>

      {area === "imported" ? (
        <KnowledgeUploadCard
          targetDirectory={areaOf(folder) === "imported" ? folder : ""}
          onImported={async () => {
            setSearchResetKey((key) => key + 1);
            await reload();
            void refreshStatus();
          }}
        />
      ) : (
        <KnowledgeDreamCard state={dream} />
      )}

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
              <Button
                className="mt-4"
                variant="outline"
                onClick={() => {
                  setLoading(true);
                  setLoadError(null);
                  void reload();
                }}
              >
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
                    key={area}
                    root={root}
                    rootLabel={AREA_LABELS[area]}
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
                    title={folder || t(AREA_LABELS[area])}
                  >
                    {folder !== rootPath ? `${folder}/` : t(AREA_LABELS[area])}
                  </span>
                  <Badge variant="outline">
                    {t("{count} 个文件", { count: formatNumber(page.total) })}
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
                ) : areaCounts[area] ? (
                  <p className="py-8 text-center text-sm text-muted-foreground">
                    {nameFilter.trim() ? t("没有匹配的文件") : t("这个文件夹里还没有文件")}
                  </p>
                ) : area === "work" ? (
                  <EmptyState
                    icon={NotebookPen}
                    title={t("还没有日常工作记忆")}
                    description={t(
                      "模型在对话中通过服务端工具写下的工作笔记会保存在 work-memory/ 下并显示在这里；每晚的记忆整理会归并和清理它们。",
                    )}
                  />
                ) : (
                  <EmptyState
                    icon={FileText}
                    title={t("知识库还是空的")}
                    description={t(
                      "导入文本文件后，模型会通过服务端工具 qwen_exo_memory_search / qwen_exo_memory_read 按需搜索和读取原文。",
                    )}
                  />
                )}
                {page.total > PAGE_SIZE ? (
                  <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
                    <span>
                      {t("第 {start}–{end} 条，共 {total} 条", {
                        start: currentPage * PAGE_SIZE + 1,
                        end: Math.min(page.total, (currentPage + 1) * PAGE_SIZE),
                        total: page.total,
                      })}
                    </span>
                    <div className="flex items-center gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        disabled={currentPage <= 0}
                        onClick={() => setPageIndex(currentPage - 1)}
                      >
                        {t("上一页")}
                      </Button>
                      <span>
                        {t("第 {page} / {pages} 页", {
                          page: currentPage + 1,
                          pages: pageCount,
                        })}
                      </span>
                      <Button
                        variant="outline"
                        size="sm"
                        disabled={currentPage + 1 >= pageCount}
                        onClick={() => setPageIndex(currentPage + 1)}
                      >
                        {t("下一页")}
                      </Button>
                    </div>
                  </div>
                ) : null}
              </section>
            </div>
          )}
        </CardContent>
      </Card>

      {area === "work" ? (
        <KnowledgeWorkTrash
          refreshKey={trashKey}
          onRestored={async () => {
            await reload();
            void refreshStatus();
          }}
        />
      ) : null}

      {viewer.dialog}

      <KnowledgeSummaryDialog
        path={summaryPath}
        view={summaryPath ? views.get(summaryPath) ?? null : null}
        onClose={() => setSummaryPath(null)}
        onChanged={() => {
          void reload();
          void refreshStatus();
        }}
      />

      <KnowledgeEntryDialog
        key={entryRequest ? `${entryRequest.kind}:${entryRequest.path}` : "closed"}
        request={entryRequest}
        folders={allFolders}
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
