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
import { createKnowledgeApi, getKnowledgeMemorySpaces } from "@/lib/api";
import { KnowledgeScope, useKnowledgeApi } from "@/components/knowledge/knowledge-scope";
import { translate as t } from "@/lib/i18n";
import type {
  KnowledgeMemorySpace,
  KnowledgeFile,
  KnowledgeFilePage,
  KnowledgeFileQuery,
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
import { KnowledgeDragProvider, type TransferSelection } from "@/components/knowledge/knowledge-drag";
import { KnowledgeTransferDialog } from "@/components/knowledge/knowledge-transfer-dialog";
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
type FileQuery = KnowledgeFileQuery & { area: MemoryArea };

export function FileKnowledgePanel() {
  const [scope, setScope] = useState<{ keyId?: string; area: MemoryArea; folder: string }>({
    area: "imported", folder: "",
  });
  const [spaces, setSpaces] = useState<KnowledgeMemorySpace[]>([]);
  const [spacesError, setSpacesError] = useState<string | null>(null);
  const [sharedTree, setSharedTree] = useState<KnowledgeTree | null>(null);
  const spacesController = useRef<AbortController | null>(null);
  const refreshSpaces = useCallback(async () => {
    spacesController.current?.abort();
    const controller = new AbortController();
    spacesController.current = controller;
    try {
      const next = await getKnowledgeMemorySpaces(controller.signal);
      if (!controller.signal.aborted) {
        setSpaces(next.spaces);
        setSpacesError(null);
      }
    } catch (error) {
      if (!controller.signal.aborted) setSpacesError(errorText(error, t("未知错误")));
    }
  }, []);
  useEffect(() => {
    void refreshSpaces();
    return () => spacesController.current?.abort();
  }, [refreshSpaces]);
  return (
    <>
      {spacesError ? <div role="alert" className="mb-3 text-sm text-destructive">
        {t("私有记忆列表加载失败")}：{spacesError}
        <Button variant="ghost" size="sm" onClick={() => void refreshSpaces()}>{t("重试")}</Button>
      </div> : null}
      <KnowledgeScope key={scope.keyId ?? "shared"} memoryKeyId={scope.keyId}>
        <ScopedFileKnowledgePanel
          initialArea={scope.area}
          initialFolder={scope.folder}
          spaces={spaces}
          sharedTree={sharedTree}
          onSharedTree={setSharedTree}
          onScopeChange={(keyId, area, folder) => setScope({ keyId, area, folder })}
          onRefreshSpaces={refreshSpaces}
        />
      </KnowledgeScope>
    </>
  );
}

function ScopedFileKnowledgePanel({ initialArea, initialFolder, spaces, sharedTree, onSharedTree, onScopeChange, onRefreshSpaces }: {
  initialArea: MemoryArea;
  initialFolder: string;
  spaces: KnowledgeMemorySpace[];
  sharedTree: KnowledgeTree | null;
  onSharedTree: (tree: KnowledgeTree) => void;
  onScopeChange: (keyId: string | undefined, area: MemoryArea, folder: string) => void;
  onRefreshSpaces: () => Promise<void>;
}) {
  const api = useKnowledgeApi();
  const { createKnowledgeFolder, deleteKnowledgeFile, deleteKnowledgeFolder,
    generateKnowledgeSummaries, getKnowledgeTree, getSummaryStatus,
    listKnowledgeFiles, moveKnowledgeEntry, memoryKeyId, signal } = api;
  const privateMemory = Boolean(memoryKeyId);
  const [tree, setTree] = useState<KnowledgeTree>({
    directories: [],
    counts: {},
    areas: { imported: { files: 0, bytes: 0 }, work: { files: 0, bytes: 0 } },
  });
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [page, setPage] = useState<KnowledgeFilePage | null>(null);
  const [query, setQuery] = useState<FileQuery>({
    area: initialArea, folder: initialFolder, nested: true, q: "", offset: 0, limit: PAGE_SIZE,
  });
  // Updated synchronously by navigation, so delayed mutations always reload the
  // current query, not the render in which their callbacks were created.
  const queryRef = useRef(query);
  const { area, folder, nested } = query;
  const treeReady = useRef(false);
  const treePending = useRef(false);
  const treeController = useRef<AbortController | null>(null);
  const pageController = useRef<AbortController | null>(null);
  const pageGeneration = useRef(0);
  const [treeRevision, setTreeRevision] = useState(0);
  // Last selected folder of each area, restored when switching back.
  const areaFolders = useRef<Record<MemoryArea, string>>({
    imported: areaRoot("imported"),
    work: areaRoot("work"),
  });
  const [nameFilter, setNameFilter] = useState("");
  const [selected, setSelected] = useState<Set<string>>(() => new Set());

  const [status, setStatus] = useState<KnowledgeSummaryStatus | null>(null);
  const [generating, setGenerating] = useState(false);

  const [entryRequest, setEntryRequest] = useState<EntryRequest | null>(null);
  const [deleteRequest, setDeleteRequest] = useState<DeleteRequest | null>(null);
  const [transfer, setTransfer] = useState<TransferSelection | null>(null);
  const [summaryPath, setSummaryPath] = useState<string | null>(null);
  const [searchResetKey, setSearchResetKey] = useState(0);

  const [trashKey, setTrashKey] = useState(0);

  const viewer = useKnowledgeViewer(() => {
    setSearchResetKey((key) => key + 1);
    void load();
    void refreshStatus();
  });

  const invalidatePage = useCallback(() => {
    pageGeneration.current += 1;
    pageController.current?.abort();
    setPage(null);
    setSelected(new Set());
    setLoadError(null);
    setLoading(true);
  }, []);

  const updateQuery = useCallback((patch: Partial<FileQuery>) => {
    const current = queryRef.current;
    const next = { ...current, ...patch };
    if (
      next.area === current.area && next.folder === current.folder &&
      next.nested === current.nested && next.q === current.q &&
      next.offset === current.offset
    ) return;
    queryRef.current = next;
    invalidatePage();
    setQuery(next);
  }, [invalidatePage]);

  const refreshPage = useCallback(async () => {
    // A fresh tree must reconcile deleted/moved folders before fetching rows.
    if (!treeReady.current || treePending.current) return;
    invalidatePage();
    const generation = pageGeneration.current;
    const controller = new AbortController();
    pageController.current = controller;
    const current = queryRef.current;
    try {
      const next = await listKnowledgeFiles({ ...current, area: privateMemory ? "all" : current.area }, controller.signal);
      if (controller.signal.aborted || generation !== pageGeneration.current) return;
      const lastOffset = Math.max(0, Math.ceil(next.total / PAGE_SIZE) - 1) * PAGE_SIZE;
      if (current.offset > lastOffset) {
        updateQuery({ offset: lastOffset });
        return;
      }
      setPage(next);
      setLoadError(null);
    } catch (error) {
      if (!controller.signal.aborted && generation === pageGeneration.current) {
        setLoadError(errorText(error, t("未知错误")));
      }
    } finally {
      if (!controller.signal.aborted && generation === pageGeneration.current) {
        setLoading(false);
      }
    }
  }, [invalidatePage, updateQuery, listKnowledgeFiles, privateMemory]);

  const load = useCallback(async () => {
    treeController.current?.abort();
    const controller = new AbortController();
    treeController.current = controller;
    treePending.current = true;
    treeReady.current = false;
    invalidatePage();
    try {
      const next = await getKnowledgeTree(controller.signal);
      if (controller.signal.aborted) return;
      const current = queryRef.current;
      const root = privateMemory ? "" : areaRoot(current.area);
      const known = new Set(next.directories);
      let nextFolder = current.folder;
      if (!privateMemory && areaOf(nextFolder) !== current.area) nextFolder = root;
      while (nextFolder && nextFolder !== root && !known.has(nextFolder)) {
        nextFolder = parentOf(nextFolder);
      }
      if (nextFolder !== current.folder) updateQuery({ folder: nextFolder, offset: 0 });
      areaFolders.current[current.area] = nextFolder;
      setTree(next);
      if (!privateMemory) onSharedTree(next);
      setLoadError(null);
      treeReady.current = true;
      setTreeRevision((revision) => revision + 1);
    } catch (error) {
      if (!controller.signal.aborted) {
        setLoadError(errorText(error, t("未知错误")));
        setLoading(false);
      }
    } finally {
      if (!controller.signal.aborted) treePending.current = false;
    }
  }, [invalidatePage, updateQuery, getKnowledgeTree, privateMemory, onSharedTree]);

  useEffect(() => {
    void load();
    return () => {
      treeController.current?.abort();
      pageController.current?.abort();
      pageGeneration.current += 1;
    };
  }, [load]);

  useEffect(() => {
    if (treePending.current) return;
    void (treeReady.current ? refreshPage() : load());
  }, [query, treeRevision, refreshPage, load]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      const q = nameFilter.trim().slice(0, 256);
      if (q !== queryRef.current.q) updateQuery({ q, offset: 0 });
    }, 250);
    return () => window.clearTimeout(timer);
  }, [nameFilter, updateQuery]);

  // Summary updates refresh only the current metadata page, without a tree scan.
  const lastRun = useRef<{ current: string | null; completed: number; failed: number } | null>(
    null,
  );
  const refreshStatus = useCallback(async () => {
    if (privateMemory || signal.aborted) return;
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
        void refreshPage();
      }
    } catch {
      // Keep the last known status; the next poll retries.
    }
  }, [refreshPage, getSummaryStatus, privateMemory, signal]);

  useEffect(() => {
    void refreshStatus();
  }, [refreshStatus]);

  // A finished dream run may have written, moved or deleted work notes.
  const dream = useDreamStatus(() => {
    void load();
    void refreshStatus();
    setTrashKey((key) => key + 1);
  }, !privateMemory, signal);

  const summaryBusy = Boolean(status?.current) || Boolean(status?.queued?.length);
  useEffect(() => {
    if (privateMemory) return;
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
  }, [refreshStatus, summaryBusy, privateMemory]);

  const rootPath = privateMemory ? "" : areaRoot(area);
  const areaCount = privateMemory ? {
    files: tree.areas.imported.files + tree.areas.work.files,
    bytes: tree.areas.imported.bytes + tree.areas.work.bytes,
  } : tree.areas[area];
  const { root, folders } = useMemo(
    () =>
      buildFolderTree(
        tree.directories.filter((path) => privateMemory || areaOf(path) === area),
        tree.counts,
        rootPath,
        areaCount.files,
      ),
    [tree.directories, tree.counts, areaCount.files, area, rootPath, privateMemory],
  );
  const space = spaces.find((item) => item.key_id === memoryKeyId);
  const currentLabel = privateMemory
    ? space?.label || memoryKeyId || t("私有记忆")
    : t(AREA_LABELS[area]);
  const switchArea = (next: MemoryArea, nextFolder?: string) => {
    if (privateMemory) {
      onScopeChange(undefined, next, nextFolder ?? areaRoot(next));
      return;
    }
    const current = queryRef.current;
    areaFolders.current[current.area] = current.folder;
    let destination = nextFolder ?? areaFolders.current[next];
    // Explicit destinations can be newly created/moved and await the next tree.
    if (nextFolder === undefined && treeReady.current) {
      const root = areaRoot(next);
      while (destination && destination !== root && !tree.directories.includes(destination)) {
        destination = parentOf(destination);
      }
    }
    areaFolders.current[next] = destination;
    updateQuery({ area: next, folder: destination, offset: 0 });
  };

  /** Select ``path``, switching to the area that owns it. */
  const focusFolder = (path: string) => privateMemory
    ? updateQuery({ folder: path, offset: 0 }) : switchArea(areaOf(path), path);

  const visibleFiles = page?.files ?? [];

  const views = useMemo(() => {
    const failures = new Map((status?.failed ?? []).map((item) => [item.path, item.error]));
    const queued = new Set(status?.queued ?? []);
    const map = new Map<string, SummaryView>();
    for (const file of page?.files ?? []) {
      map.set(file.relative_path, summaryView(file, status, failures, queued));
    }
    return map;
  }, [page, status]);

  const totalBytes = areaCount.bytes;
  const folderBytes = page?.total_bytes ?? 0;

  const reportError = (title: string, error: unknown): never => {
    toast.error(title, { description: errorText(error, t("未知错误")) });
    throw error;
  };

  const generate = async (paths: string[], force = false) => {
    if (generating || privateMemory || signal.aborted) return;
    setGenerating(true);
    try {
      const next = await generateKnowledgeSummaries(paths, force);
      if (signal.aborted) return;
      setStatus(next);
      toast.success(
        paths.length
          ? t("已将 {count} 个文件加入总结队列", { count: paths.length })
          : t("已开始生成缺失的总结"),
      );
    } catch (error) {
      if (signal.aborted) return;
      toast.error(t("生成总结失败"), { description: errorText(error, t("未知错误")) });
    } finally {
      setGenerating(false);
    }
  };

  const submitEntry = async (request: EntryRequest, destination: string) => {
    const submittedQuery = queryRef.current;
    if (request.kind === "create") {
      try {
        const result = await createKnowledgeFolder(destination);
        if (signal.aborted) return;
        toast.success(t("文件夹已创建"));
        if (queryRef.current === submittedQuery) focusFolder(result.path || destination);
        await load();
      } catch (error) {
        if (signal.aborted) return;
        reportError(t("创建文件夹失败"), error);
      }
      return;
    }
    try {
      const result = await moveKnowledgeEntry(request.path, destination);
      if (signal.aborted) return;
      const target = result.to || destination;
      toast.success(
        t("已重命名"),
        {
          description: request.isFolder
            ? t("{to}（{count} 个文件）", { to: target, count: result.files })
            : target,
        },
      );
      const currentFolder = queryRef.current.folder;
      const rebased = rebasePath(currentFolder, request.path, target);
      for (const memoryArea of ["imported", "work"] as const) {
        const saved = rebasePath(areaFolders.current[memoryArea], request.path, target);
        areaFolders.current[memoryArea] = areaOf(saved) === memoryArea
          ? saved : areaRoot(memoryArea);
      }
      if (rebased !== currentFolder) focusFolder(rebased);
      setSummaryPath((current) => (current ? rebasePath(current, request.path, target) : current));
      await load();
    } catch (error) {
      if (signal.aborted) return;
      reportError(t("重命名失败"), error);
    }
  };

  const confirmDelete = async (request: DeleteRequest) => {
    try {
      if (request.kind === "folder") {
        const result = await deleteKnowledgeFolder(request.path);
        if (signal.aborted) return;
        toast.success(t("文件夹已删除"), {
          description: t("共删除 {count} 个文件", { count: result.files }),
        });
        if (isWithin(queryRef.current.folder, request.path)) {
          focusFolder(parentOf(request.path));
        }
      } else {
        await deleteKnowledgeFile(request.path);
        if (signal.aborted) return;
        toast.success(t("知识库文件已删除"));
      }
      await load();
      void refreshStatus();
    } catch (error) {
      if (signal.aborted) return;
      reportError(t("删除失败"), error);
    }
  };

  const requestMove = (path: string, kind: "file" | "folder", keyId = memoryKeyId ?? null) => {
    setTransfer({ source: { keyId, path, kind }, destinationKeyId: keyId, folder: parentOf(path) });
  };

  const refreshAfterTransfer = async (selection: TransferSelection) => {
    setSearchResetKey((key) => key + 1);
    setSummaryPath(null);
    setTrashKey((key) => key + 1);
    await Promise.all([load(), onRefreshSpaces(), refreshStatus()]);
    if (signal.aborted) return;
    // The sidebar retains the shared counts across scope remounts. Refresh both
    // physical endpoints; destination picker reads are never reused across KEYs.
    const owners = new Set([selection.source.keyId, selection.destinationKeyId]);
    owners.delete(memoryKeyId ?? null);
    await Promise.all(Array.from(owners, async (keyId) => {
      const next = await createKnowledgeApi(keyId ?? undefined, signal).getKnowledgeTree();
      if (!signal.aborted && keyId === null) onSharedTree(next);
    }));
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
    } else if (action === "move") {
      requestMove(path, "folder");
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
    else if (action === "move") requestMove(path, "file");
    else setEntryRequest({ kind: action, path, isFolder: false });
  };

  return (
    <KnowledgeDragProvider onTransfer={setTransfer}>
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <Tabs value={area} onValueChange={(value) => switchArea(value as MemoryArea)}>
            <TabsList>
              <TabsTrigger value="imported" className="gap-1.5">
                <BookOpen className="h-3.5 w-3.5" />
                {t(AREA_LABELS.imported)}
                <span className="font-mono text-[11px] text-muted-foreground">
                  {formatNumber((privateMemory ? sharedTree : tree)?.areas.imported.files ?? 0)}
                </span>
              </TabsTrigger>
              <TabsTrigger value="work" className="gap-1.5">
                <NotebookPen className="h-3.5 w-3.5" />
                {t(AREA_LABELS.work)}
                <span className="font-mono text-[11px] text-muted-foreground">
                  {formatNumber((privateMemory ? sharedTree : tree)?.areas.work.files ?? 0)}
                </span>
              </TabsTrigger>
            </TabsList>
          </Tabs>
          <Badge variant="secondary">
            {t("{count} 个文件", { count: formatNumber(areaCount.files) })}
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
            void dream.refresh();
            void onRefreshSpaces();
            setTrashKey((key) => key + 1);
          }}
        >
          <RefreshCw />
          {t("刷新")}
        </Button>
      </div>

      {area === "imported" || privateMemory ? (
        <KnowledgeUploadCard
          targetDirectory={folder}
          rootLabel={currentLabel}
          onImported={async () => {
            setSearchResetKey((key) => key + 1);
            await load();
            void refreshStatus();
          }}
        />
      ) : (
        <KnowledgeDreamCard state={dream} />
      )}

      <KnowledgeSearchCard key={searchResetKey} resetKey={searchResetKey} onOpen={(path) => void viewer.open(path)} />

      <Card>
        <CardContent className="p-0">
            <div className="grid lg:grid-cols-[17rem_minmax(0,1fr)]">
              <aside className="min-w-0 border-b p-3 lg:border-b-0 lg:border-r">
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
                <div className="space-y-1 border-b pb-2">
                  {(["imported", "work"] as const).map((memoryArea) => {
                    const shared = privateMemory ? sharedTree : tree;
                    const path = areaRoot(memoryArea);
                    return <KnowledgeFolderTree key={memoryArea}
                      root={{ path, name: AREA_LABELS[memoryArea], fileCount: shared?.areas[memoryArea].files ?? 0, children: [] }}
                      rootLabel={t(AREA_LABELS[memoryArea])}
                      rootTitle={t("共享记忆") + " / " + (path || "/")}
                      selected={!privateMemory && area === memoryArea ? path : "\u0000"}
                      onSelect={() => privateMemory ? onScopeChange(undefined, memoryArea, path) : switchArea(memoryArea, path)}
                      onAction={!privateMemory ? onFolderAction : undefined}
                      onMigrate={memoryArea === "work" ? () => requestMove(path, "folder", null) : undefined}
                    />;
                  })}
                  {spaces.map((owner) => <KnowledgeFolderTree key={owner.key_id}
                    root={{ path: "", name: owner.label, fileCount: owner.files, children: [] }}
                    rootLabel={owner.label || owner.key_id}
                    rootTitle={`${owner.label || owner.key_id}\n${owner.key_id}\n${t("私有记忆")} · ${owner.memory_scope}${owner.deleted ? ` · ${t("已删除")}` : owner.revoked ? ` · ${t("已撤销")}` : ""}`}
                    memoryKeyId={owner.key_id}
                    selected={owner.key_id === memoryKeyId ? "" : "\u0000"}
                    onSelect={() => owner.key_id === memoryKeyId ? focusFolder("") : onScopeChange(owner.key_id, "work", "")}
                    onAction={owner.key_id === memoryKeyId ? onFolderAction : undefined}
                    onMigrate={() => setTransfer({ source: { keyId: owner.key_id, path: "", kind: "folder" }, destinationKeyId: null, folder: "work-memory" })}
                  />)}
                </div>
                <div className="max-h-[50vh] overflow-auto pt-2">
                  <KnowledgeFolderTree key={`${memoryKeyId ?? "shared"}:${area}`}
                    root={root} rootLabel={currentLabel} selected={folder} memoryKeyId={memoryKeyId}
                    onSelect={focusFolder} onAction={onFolderAction} hideRoot
                  />
                </div>
              </aside>
              <section className="min-w-0 space-y-3 p-3">
                <div className="flex flex-wrap items-center gap-2">
                  <span
                    className="min-w-0 truncate font-mono text-sm"
                    title={currentLabel + (folder !== rootPath ? ` / ${folder}` : "")}
                  >
                    {folder !== rootPath ? `${currentLabel} / ${folder}/` : currentLabel}
                  </span>
                  <Badge variant="outline">
                    {t("{count} 个文件", { count: formatNumber(page?.total ?? 0) })}
                    {" · "}
                    {formatBytes(folderBytes)}
                  </Badge>
                  <label className="flex items-center gap-2 text-xs text-muted-foreground">
                    <input
                      type="checkbox"
                      checked={nested}
                      onChange={(event) => updateQuery({ nested: event.target.checked, offset: 0 })}
                      className="h-4 w-4 accent-primary"
                    />
                    {t("包含子文件夹")}
                  </label>
                  <Input
                    className="ml-auto h-8 w-full sm:w-56"
                    value={nameFilter}
                    placeholder={t("按名称或标题筛选")}
                    maxLength={256}
                    onChange={(event) => {
                      setNameFilter(event.target.value.slice(0, 256));
                      setSelected(new Set());
                    }}
                  />
                </div>
                {!privateMemory ? <KnowledgeSummaryToolbar
                  status={status}
                  selectedCount={selected.size}
                  busy={generating}
                  onGenerateMissing={() => void generate([])}
                  onGenerateSelected={() => void generate(Array.from(selected))}
                  onClearSelection={() => setSelected(new Set())}
                /> : null}
                {loading ? (
                  <div className="space-y-2 py-5" aria-busy="true">
                    {Array.from({ length: 4 }).map((_, index) => (
                      <Skeleton key={index} className="h-10 w-full" />
                    ))}
                  </div>
                ) : loadError ? (
                  <div className="py-5" role="alert">
                    <p className="text-sm font-medium">{t("知识库加载失败")}</p>
                    <p className="mt-2 break-words text-xs text-muted-foreground">{loadError}</p>
                    <Button
                      className="mt-4"
                      variant="outline"
                      onClick={() => void (treeReady.current ? refreshPage() : load())}
                    >
                      <RefreshCw />
                      {t("重试")}
                    </Button>
                  </div>
                ) : visibleFiles.length ? (
                  <div className="rounded-md border">
                    <KnowledgeFileTable
                      files={visibleFiles}
                      allowGenerate={!privateMemory}
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
                ) : areaCount.files ? (
                  <p className="py-8 text-center text-sm text-muted-foreground">
                    {nameFilter.trim() ? t("没有匹配的文件") : t("这个文件夹里还没有文件")}
                  </p>
                ) : area === "work" ? (
                  <EmptyState
                    icon={NotebookPen}
                    title={t("还没有日常工作记忆")}
                    description={t("上传文件或在对话中写入笔记。")}
                  />
                ) : (
                  <EmptyState
                    icon={FileText}
                    title={t("知识库还是空的")}
                    description={t("上传文件或在对话中写入笔记。")}
                  />
                )}
                <nav className="flex flex-wrap items-center justify-between gap-2">
                  <span className="text-xs text-muted-foreground" aria-live="polite">
                    {page ? t("第 {start}–{end} 条，共 {total} 条", {
                      start: page.total ? page.offset + 1 : 0,
                      end: page.offset + page.files.length,
                      total: page.total,
                    }) : "—"}
                  </span>
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={loading || query.offset === 0}
                      onClick={() => updateQuery({ offset: Math.max(0, queryRef.current.offset - PAGE_SIZE) })}
                    >
                      {t("上一页")}
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={loading || !page || page.offset + PAGE_SIZE >= page.total}
                      onClick={() => updateQuery({ offset: queryRef.current.offset + PAGE_SIZE })}
                    >
                      {t("下一页")}
                    </Button>
                  </div>
                </nav>
              </section>
            </div>
        </CardContent>
      </Card>

      {area === "work" ? (
        <KnowledgeWorkTrash
          refreshKey={trashKey}
          onRestored={async () => {
            await load();
            void refreshStatus();
          }}
        />
      ) : null}

      {viewer.dialog}

      <KnowledgeSummaryDialog
        key={summaryPath ?? "closed"}
        path={summaryPath}
        view={summaryPath ? views.get(summaryPath) ?? null : null}
        onClose={() => setSummaryPath(null)}
        onChanged={() => {
          void refreshPage();
          void refreshStatus();
        }}
      />

      {transfer ? <KnowledgeTransferDialog
        selection={transfer} spaces={spaces} signal={signal}
        onClose={() => setTransfer(null)} onTransferred={refreshAfterTransfer}
      /> : null}

      <KnowledgeEntryDialog
        key={entryRequest ? `${entryRequest.kind}:${entryRequest.path}` : "closed"}
        request={entryRequest}
        onClose={() => setEntryRequest(null)}
        onSubmit={submitEntry}
      />

      <KnowledgeDeleteDialog
        request={deleteRequest}
        onClose={() => setDeleteRequest(null)}
        onConfirm={confirmDelete}
      />
    </div>
    </KnowledgeDragProvider>
  );
}
