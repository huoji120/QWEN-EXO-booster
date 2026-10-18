import { useRef, useState } from "react";
import { Folder, FolderUp, LoaderCircle, Upload, X } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ApiError, uploadKnowledgeArchiveRaw, uploadKnowledgeFileRaw } from "@/lib/api";
import { translate as t } from "@/lib/i18n";
import { cn, formatBytes } from "@/lib/utils";
import {
  collectDropped,
  errorText,
  isZip,
  joinPath,
  parentOf,
  pickedPath,
  type PendingFile,
} from "./knowledge-utils";

/** Upload files, folders or ZIPs into the selected folder (no size limit). */
export function KnowledgeUploadCard({
  targetDirectory,
  onImported,
}: {
  targetDirectory: string;
  onImported: () => void | Promise<void>;
}) {
  const [pending, setPending] = useState<PendingFile[]>([]);
  const [importing, setImporting] = useState(false);
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  const fromInput = (list: FileList | null) =>
    setPending(
      Array.from(list ?? []).map((file) => ({ file, relativePath: pickedPath(file) })),
    );

  const clear = () => {
    setPending([]);
    if (fileInputRef.current) fileInputRef.current.value = "";
    if (folderInputRef.current) folderInputRef.current.value = "";
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
            const result = await uploadKnowledgeArchiveRaw(item.file, parentOf(relativePath));
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
          if (error instanceof ApiError && error.status === 507) {
            // Disk would drop below the reserve; the rest would fail too.
            toast.error(t("服务器磁盘空间不足，已停止导入"), {
              description: `${relativePath}：${error.message}`,
              duration: 15_000,
            });
            break;
          }
          toast.error(t("导入失败"), {
            description: `${relativePath}：${errorText(error, t("未知错误"))}`,
          });
        }
      }
      if (succeeded) {
        toast.success(t("已导入 {count} 个知识库文件", { count: succeeded }));
        clear();
        await onImported();
      }
    } finally {
      setImporting(false);
    }
  };

  const totalBytes = pending.reduce((sum, item) => sum + item.file.size, 0);

  return (
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
          if (picked.length) setPending(picked);
        });
      }}
    >
      <CardContent className="flex flex-col gap-3 pt-6">
        <input
          ref={fileInputRef}
          type="file"
          multiple
          hidden
          onChange={(event) => fromInput(event.target.files)}
        />
        <input
          ref={folderInputRef}
          type="file"
          multiple
          hidden
          {...({ webkitdirectory: "", directory: "" } as Record<string, string>)}
          onChange={(event) => fromInput(event.target.files)}
        />
        <div className="flex flex-wrap items-center gap-2">
          <Button variant="outline" onClick={() => fileInputRef.current?.click()}>
            <Upload />
            {t("选择文件")}
          </Button>
          <Button variant="outline" onClick={() => folderInputRef.current?.click()}>
            <FolderUp />
            {t("选择文件夹")}
          </Button>
          <span className="flex min-w-0 items-center gap-1 text-sm text-muted-foreground">
            <Folder className="h-4 w-4 shrink-0" />
            {t("导入到")}
            <span className="truncate font-mono text-xs text-foreground">
              {targetDirectory ? `${targetDirectory}/` : t("导入的记忆（根目录）")}
            </span>
          </span>
          <div className="ml-auto flex items-center gap-2">
            {pending.length ? (
              <Button variant="ghost" size="sm" disabled={importing} onClick={clear}>
                <X />
                {t("清空")}
              </Button>
            ) : null}
            <Button disabled={!pending.length || importing} onClick={() => void importFiles()}>
              {importing ? <LoaderCircle className="animate-spin" /> : <Upload />}
              {importing
                ? t("导入中")
                : t("导入 {count} 个文件", { count: pending.length })}
            </Button>
          </div>
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
            <div className="mt-1 text-muted-foreground">
              {t("合计 {size}", { size: formatBytes(totalBytes) })}
            </div>
          </div>
        ) : null}
        <p className="text-xs text-muted-foreground">
          {t(
            "可拖拽文件、文件夹或 ZIP 到这里，导入到左侧选中的文件夹。支持任意 UTF-8 文本（文档、代码、日志、配置），不限大小；ZIP 会在服务端解压，每个文本文件各存一份。同路径文件会被覆盖。导入后服务端会在后台生成总结。",
          )}
        </p>
      </CardContent>
    </Card>
  );
}
