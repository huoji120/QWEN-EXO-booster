import { useCallback, useEffect, useRef, useState } from "react";
import {
  FileText,
  FolderOpen,
  FolderUp,
  LoaderCircle,
  RefreshCw,
  Search,
  Trash2,
  Upload,
} from "lucide-react";
import { toast } from "sonner";
import { EmptyState } from "@/components/empty-state";
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
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  deleteKnowledgeFile,
  listKnowledgeFiles,
  readKnowledge,
  searchKnowledge,
  uploadKnowledgeArchiveRaw,
  uploadKnowledgeFileRaw,
} from "@/lib/api";
import { translate as t } from "@/lib/i18n";
import type { KnowledgeFile, KnowledgeSearchResponse } from "@/lib/types";
import { cn, formatBytes, formatNumber, formatTime, shortHash } from "@/lib/utils";

const VIEWER_WINDOW = 262_144;
const MAX_UPLOAD_BYTES = 100 * 1024 * 1024;
const MAX_ARCHIVE_BYTES = 100 * 1024 * 1024;

type PendingFile = { file: File; relativePath: string };

const isZip = (item: PendingFile) => /\.zip$/i.test(item.file.name);

type ViewerState = {
  path: string;
  text: string;
  endOffset: number;
  totalBytes: number;
  eof: boolean;
};

/** Folder-relative path for picked/dropped files; bare name otherwise. */
function pickedPath(file: File) {
  const relative = (file as File & { webkitRelativePath?: string })
    .webkitRelativePath;
  return (relative || file.name).replace(/\\/g, "/").replace(/^\/+/, "");
}

function joinPath(directory: string, path: string) {
  const prefix = directory.trim().replace(/\\/g, "/").replace(/^\/+|\/+$/g, "");
  return prefix ? `${prefix}/${path}` : path;
}

/** Expand dropped items, walking dropped folders and keeping their paths. */
async function collectDropped(items: DataTransferItemList): Promise<PendingFile[]> {
  const out: PendingFile[] = [];
  const walk = async (entry: FileSystemEntry): Promise<void> => {
    if (entry.isFile) {
      const file = await new Promise<File>((resolve, reject) =>
        (entry as FileSystemFileEntry).file(resolve, reject),
      );
      out.push({ file, relativePath: entry.fullPath.replace(/^\/+/, "") });
    } else if (entry.isDirectory) {
      const reader = (entry as FileSystemDirectoryEntry).createReader();
      // readEntries returns batches; an empty batch means done.
      for (;;) {
        const batch = await new Promise<FileSystemEntry[]>((resolve, reject) =>
          reader.readEntries(resolve, reject),
        );
        if (!batch.length) break;
        for (const child of batch) await walk(child);
      }
    }
  };
  const entries = Array.from(items)
    .map((item) => item.webkitGetAsEntry())
    .filter((entry): entry is FileSystemEntry => entry !== null);
  for (const entry of entries) await walk(entry);
  return out;
}

export function FileKnowledgePanel() {
  const [files, setFiles] = useState<KnowledgeFile[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [pending, setPending] = useState<PendingFile[]>([]);
  const [targetDirectory, setTargetDirectory] = useState("");
  const [importing, setImporting] = useState(false);
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  const [deleting, setDeleting] = useState<string | null>(null);
  const [deleteBusy, setDeleteBusy] = useState(false);

  const [viewer, setViewer] = useState<ViewerState | null>(null);
  const [viewerLoading, setViewerLoading] = useState(false);

  const [pattern, setPattern] = useState("");
  const [pathGlob, setPathGlob] = useState("**/*");
  const [caseSensitive, setCaseSensitive] = useState(false);
  const [regex, setRegex] = useState(false);
  const [searching, setSearching] = useState(false);
  const [searchResult, setSearchResult] =
    useState<KnowledgeSearchResponse | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setLoadError(null);
    try {
      setFiles((await listKnowledgeFiles()).files);
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : t("未知错误"));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const openFile = useCallback(async (path: string) => {
    setViewerLoading(true);
    try {
      const result = await readKnowledge(path, 0, VIEWER_WINDOW);
      setViewer({
        path: result.path,
        text: result.text,
        endOffset: result.end_offset,
        totalBytes: result.total_bytes,
        eof: result.eof,
      });
    } catch (error) {
      toast.error(t("文件读取失败"), {
        description: error instanceof Error ? error.message : t("未知错误"),
      });
    } finally {
      setViewerLoading(false);
    }
  }, []);

  const loadMore = async () => {
    if (!viewer || viewer.eof || viewerLoading) return;
    setViewerLoading(true);
    try {
      const result = await readKnowledge(viewer.path, viewer.endOffset, VIEWER_WINDOW);
      setViewer({
        ...viewer,
        text: viewer.text + result.text,
        endOffset: result.end_offset,
        eof: result.eof,
      });
    } catch (error) {
      toast.error(t("文件读取失败"), {
        description: error instanceof Error ? error.message : t("未知错误"),
      });
    } finally {
      setViewerLoading(false);
    }
  };

  const queue = (picked: PendingFile[]) => {
    const fits = (item: PendingFile) =>
      item.file.size <= (isZip(item) ? MAX_ARCHIVE_BYTES : MAX_UPLOAD_BYTES);
    const tooLarge = picked.filter((item) => !fits(item));
    if (tooLarge.length) {
      toast.error(
        t("已跳过 {count} 个超过 100 MiB 的文件", {
          count: tooLarge.length,
        }),
      );
    }
    setPending(picked.filter(fits));
  };

  const importFiles = async () => {
    if (!pending.length || importing) return;
    setImporting(true);
    let succeeded = 0;
    try {
      for (const item of pending) {
        const relativePath = joinPath(targetDirectory, item.relativePath);
        try {
          if (isZip(item)) {
            // The server unpacks the archive; each text member is one file.
            const directory = relativePath.replace(/\/?[^/]*$/, "");
            const result = await uploadKnowledgeArchiveRaw(item.file, directory);
            succeeded += result.files.length;
            if (result.skipped.length) {
              toast.warning(
                t("{name} 中有 {count} 个文件未导入", {
                  name: item.file.name,
                  count: result.skipped.length,
                }),
                {
                  description: result.skipped
                    .slice(0, 5)
                    .map((entry) => `${entry.path}：${entry.reason}`)
                    .join("；"),
                },
              );
            }
          } else {
            await uploadKnowledgeFileRaw(relativePath, item.file);
            succeeded += 1;
          }
        } catch (error) {
          toast.error(t("导入失败"), {
            description: `${relativePath}：${error instanceof Error ? error.message : t("未知错误")}`,
          });
        }
      }
      if (succeeded) {
        toast.success(t("已导入 {count} 个知识库文件", { count: succeeded }));
        setPending([]);
        if (fileInputRef.current) fileInputRef.current.value = "";
        if (folderInputRef.current) folderInputRef.current.value = "";
        setSearchResult(null);
        await load();
      }
    } finally {
      setImporting(false);
    }
  };

  const confirmDelete = async () => {
    if (!deleting || deleteBusy) return;
    setDeleteBusy(true);
    try {
      await deleteKnowledgeFile(deleting);
      toast.success(t("知识库文件已删除"));
      setDeleting(null);
      await load();
    } catch (error) {
      toast.error(t("删除失败"), {
        description: error instanceof Error ? error.message : t("未知错误"),
      });
    } finally {
      setDeleteBusy(false);
    }
  };

  const runSearch = async () => {
    if (!pattern.trim() || searching) return;
    setSearching(true);
    try {
      setSearchResult(
        await searchKnowledge({
          pattern: pattern.trim(),
          pathGlob: pathGlob.trim() || "**/*",
          caseSensitive,
          regex,
        }),
      );
    } catch (error) {
      toast.error(t("搜索失败"), {
        description: error instanceof Error ? error.message : t("未知错误"),
      });
    } finally {
      setSearching(false);
    }
  };

  const totalBytes = files.reduce((sum, file) => sum + file.byte_count, 0);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant="secondary">
            {t("{count} 个文件", { count: formatNumber(files.length) })}
          </Badge>
          <Badge variant="secondary">{formatBytes(totalBytes)}</Badge>
        </div>
        <Button variant="outline" size="sm" onClick={() => void load()}>
          <RefreshCw />
          {t("刷新")}
        </Button>
      </div>

      <Card
        className={cn(dragging && "border-primary ring-1 ring-primary")}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          void collectDropped(event.dataTransfer.items).then((picked) => {
            if (picked.length) queue(picked);
          });
        }}
      >
        <CardContent className="flex flex-col gap-3 pt-6">
          <input
            ref={fileInputRef}
            type="file"
            multiple
            hidden
            onChange={(event) =>
              queue(
                Array.from(event.target.files ?? []).map((file) => ({
                  file,
                  relativePath: pickedPath(file),
                })),
              )
            }
          />
          <input
            ref={folderInputRef}
            type="file"
            multiple
            hidden
            {...({ webkitdirectory: "", directory: "" } as Record<string, string>)}
            onChange={(event) =>
              queue(
                Array.from(event.target.files ?? []).map((file) => ({
                  file,
                  relativePath: pickedPath(file),
                })),
              )
            }
          />
          <div className="grid gap-3 sm:grid-cols-[auto_auto_minmax(0,1fr)_auto] sm:items-end">
            <Button variant="outline" onClick={() => fileInputRef.current?.click()}>
              <Upload />
              {t("选择文件")}
            </Button>
            <Button variant="outline" onClick={() => folderInputRef.current?.click()}>
              <FolderUp />
              {t("选择文件夹")}
            </Button>
            <div className="space-y-2">
              <Label htmlFor="knowledge-target-dir">{t("目标目录（可选）")}</Label>
              <Input
                id="knowledge-target-dir"
                value={targetDirectory}
                placeholder={t("例如 docs/ops")}
                onChange={(event) => setTargetDirectory(event.target.value)}
              />
            </div>
            <Button disabled={!pending.length || importing} onClick={() => void importFiles()}>
              {importing ? <LoaderCircle className="animate-spin" /> : <Upload />}
              {importing
                ? t("导入中")
                : t("导入 {count} 个文件", { count: pending.length })}
            </Button>
          </div>
          {pending.length ? (
            <div className="max-h-32 overflow-auto rounded-md border bg-muted/40 p-2 font-mono text-xs">
              {pending.map((item) => (
                <div key={item.relativePath} className="truncate">
                  {joinPath(targetDirectory, item.relativePath)}
                  <span className="ml-2 text-muted-foreground">
                    {formatBytes(item.file.size)}
                  </span>
                </div>
              ))}
            </div>
          ) : null}
          <p className="text-xs text-muted-foreground">
            {t(
              "可拖拽文件、文件夹或 ZIP 到这里。支持任意 UTF-8 文本（文档、代码、日志、配置），单个文件不超过 100 MiB；ZIP（不超过 100 MiB）会在服务端解压，每个文本文件各存一份。同路径文件会被覆盖。",
            )}
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="flex flex-col gap-3 pt-6">
          <div className="grid gap-3 sm:grid-cols-[minmax(0,2fr)_minmax(0,1fr)_auto]">
            <div className="space-y-2">
              <Label htmlFor="knowledge-pattern">{t("搜索原文")}</Label>
              <Input
                id="knowledge-pattern"
                value={pattern}
                placeholder={t("输入要检索的文本")}
                onChange={(event) => setPattern(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") void runSearch();
                }}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="knowledge-glob">{t("路径过滤")}</Label>
              <Input
                id="knowledge-glob"
                value={pathGlob}
                placeholder="**/*"
                onChange={(event) => setPathGlob(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") void runSearch();
                }}
              />
            </div>
            <div className="flex items-end gap-4">
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={caseSensitive}
                  onChange={(event) => setCaseSensitive(event.target.checked)}
                  className="h-4 w-4 accent-primary"
                />
                {t("区分大小写")}
              </label>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={regex}
                  onChange={(event) => setRegex(event.target.checked)}
                  className="h-4 w-4 accent-primary"
                />
                {t("正则表达式")}
              </label>
              <Button disabled={!pattern.trim() || searching} onClick={() => void runSearch()}>
                {searching ? <LoaderCircle className="animate-spin" /> : <Search />}
                {t("搜索")}
              </Button>
            </div>
          </div>
          {searchResult ? (
            searchResult.results.length ? (
              <div className="space-y-2">
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <span>
                    {t("{count} 条结果", { count: formatNumber(searchResult.count) })}
                  </span>
                  {searchResult.truncated ? (
                    <Badge variant="outline">{t("结果已截断")}</Badge>
                  ) : null}
                </div>
                <div className="max-h-[420px] overflow-auto rounded-md border">
                  <ul className="divide-y">
                    {searchResult.results.map((result, index) => (
                      <li key={`${result.path}:${result.line}:${index}`}>
                        <button
                          type="button"
                          className="w-full px-3 py-2 text-left hover:bg-muted"
                          onClick={() => void openFile(result.path)}
                        >
                          <div className="font-mono text-xs text-muted-foreground">
                            {result.path}:{result.line}
                          </div>
                          {result.before.map((line, beforeIndex) => (
                            <div
                              key={`before-${beforeIndex}`}
                              className="mt-1 whitespace-pre-wrap break-words font-mono text-xs text-muted-foreground"
                            >
                              {line}
                            </div>
                          ))}
                          <div className="mt-1 whitespace-pre-wrap break-words font-mono text-xs">
                            {result.line_text ?? result.text}
                          </div>
                          {result.after.map((line, afterIndex) => (
                            <div
                              key={`after-${afterIndex}`}
                              className="whitespace-pre-wrap break-words font-mono text-xs text-muted-foreground"
                            >
                              {line}
                            </div>
                          ))}
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">{t("没有匹配结果")}</p>
            )
          ) : null}
        </CardContent>
      </Card>

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
          ) : files.length ? (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>{t("路径")}</TableHead>
                  <TableHead className="w-24">{t("大小")}</TableHead>
                  <TableHead className="w-24">{t("行数")}</TableHead>
                  <TableHead className="w-32">{t("SHA-256")}</TableHead>
                  <TableHead className="w-40">{t("索引时间")}</TableHead>
                  <TableHead className="w-32 text-right">{t("操作")}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {files.map((file) => (
                  <TableRow key={file.relative_path}>
                    <TableCell className="break-all font-mono text-xs">
                      {file.relative_path}
                      {file.encoding !== "utf-8" ? (
                        <Badge variant="outline" className="ml-2">
                          {t("非 UTF-8，不可检索")}
                        </Badge>
                      ) : null}
                    </TableCell>
                    <TableCell className="font-mono text-xs">
                      {formatBytes(file.byte_count)}
                    </TableCell>
                    <TableCell className="font-mono text-xs">
                      {formatNumber(file.line_count)}
                    </TableCell>
                    <TableCell className="font-mono text-xs text-muted-foreground">
                      {shortHash(file.sha256)}
                    </TableCell>
                    <TableCell className="font-mono text-xs text-muted-foreground">
                      {formatTime(file.indexed_ns / 1_000_000_000)}
                    </TableCell>
                    <TableCell className="text-right">
                      <div className="flex justify-end gap-1">
                        <Button
                          variant="ghost"
                          size="sm"
                          disabled={viewerLoading || file.encoding !== "utf-8"}
                          onClick={() => void openFile(file.relative_path)}
                        >
                          <FolderOpen />
                          {t("查看")}
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="text-destructive hover:text-destructive"
                          onClick={() => setDeleting(file.relative_path)}
                        >
                          <Trash2 />
                          {t("删除")}
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <EmptyState
              icon={FileText}
              title={t("知识库还是空的")}
              description={t(
                "导入文本文件后，模型会通过服务端工具 qwen_exo_memory_search / qwen_exo_memory_read 按需搜索和读取原文。",
              )}
            />
          )}
        </CardContent>
      </Card>

      <Dialog open={Boolean(viewer)} onOpenChange={(open) => !open && setViewer(null)}>
        <DialogContent className="max-w-3xl">
          <DialogHeader>
            <DialogTitle className="break-all font-mono text-sm">{viewer?.path}</DialogTitle>
            <DialogDescription>
              {viewer
                ? viewer.eof
                  ? t("完整内容，共 {total} 字节", {
                      total: formatNumber(viewer.totalBytes),
                    })
                  : t("已显示前 {end} 字节，共 {total} 字节", {
                      end: formatNumber(viewer.endOffset),
                      total: formatNumber(viewer.totalBytes),
                    })
                : null}
            </DialogDescription>
          </DialogHeader>
          <div className="max-h-[60vh] overflow-auto rounded-md border bg-muted/40 p-3">
            <pre className="whitespace-pre-wrap break-words font-mono text-xs leading-5">
              {viewer?.text}
            </pre>
          </div>
          {viewer && !viewer.eof ? (
            <DialogFooter>
              <Button variant="outline" disabled={viewerLoading} onClick={() => void loadMore()}>
                {viewerLoading ? <LoaderCircle className="animate-spin" /> : null}
                {t("继续加载")}
              </Button>
            </DialogFooter>
          ) : null}
        </DialogContent>
      </Dialog>

      <Dialog
        open={Boolean(deleting)}
        onOpenChange={(open) => !open && !deleteBusy && setDeleting(null)}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t("删除知识库文件")}</DialogTitle>
            <DialogDescription>
              {t("删除后文件与索引记录一并移除，模型将无法再检索该文件。")}
            </DialogDescription>
          </DialogHeader>
          <div className="break-all rounded-md border bg-muted/40 p-3 font-mono text-xs">
            {deleting}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleting(null)}>
              {t("取消")}
            </Button>
            <Button
              variant="destructive"
              disabled={deleteBusy}
              onClick={() => void confirmDelete()}
            >
              {deleteBusy ? <LoaderCircle className="animate-spin" /> : <Trash2 />}
              {t("确认删除")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
