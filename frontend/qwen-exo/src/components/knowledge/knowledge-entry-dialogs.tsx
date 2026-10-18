import { useId, useState } from "react";
import { LoaderCircle, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
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
import { translate as t } from "@/lib/i18n";
import { formatNumber } from "@/lib/utils";
import { baseName, joinPath, normalizePath, parentOf } from "./knowledge-utils";

export type EntryRequest = {
  kind: "create" | "rename" | "move";
  /** Parent folder for ``create``; the entry itself otherwise. */
  path: string;
  isFolder: boolean;
};

function initialValue(request: EntryRequest) {
  if (request.kind === "rename") return baseName(request.path);
  if (request.kind === "move") return parentOf(request.path);
  return "";
}

function destinationFor(request: EntryRequest, value: string) {
  if (request.kind === "create") return joinPath(request.path, value);
  if (request.kind === "rename") return joinPath(parentOf(request.path), value);
  return joinPath(value, baseName(request.path));
}

/** New folder / rename / move dialog; ``onSubmit`` gets the final path. */
export function KnowledgeEntryDialog({
  request,
  folders,
  onClose,
  onSubmit,
}: {
  request: EntryRequest | null;
  folders: string[];
  onClose: () => void;
  onSubmit: (request: EntryRequest, destination: string) => Promise<void>;
}) {
  const inputId = useId();
  const listId = useId();
  // Callers remount via ``key`` per request, so the prefill is computed once.
  const [value, setValue] = useState(() => (request ? initialValue(request) : ""));
  const [busy, setBusy] = useState(false);

  if (!request) return null;

  const destination = destinationFor(request, value);
  const unchanged = request.kind !== "create" && destination === request.path;
  const invalid =
    !destination ||
    unchanged ||
    (request.kind !== "move" && !normalizePath(value)) ||
    (request.isFolder &&
      request.kind === "move" &&
      (destination === request.path || destination.startsWith(`${request.path}/`)));

  const title =
    request.kind === "create"
      ? t("新建文件夹")
      : request.kind === "rename"
        ? request.isFolder
          ? t("重命名文件夹")
          : t("重命名文件")
        : request.isFolder
          ? t("移动文件夹")
          : t("移动文件");

  const submit = async () => {
    if (invalid || busy) return;
    setBusy(true);
    try {
      await onSubmit(request, destination);
      onClose();
    } catch {
      // The caller reports the error; keep the dialog open for a retry.
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog open onOpenChange={(open) => !open && !busy && onClose()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription>
            {request.kind === "create"
              ? t("在 {folder} 下创建；可用 / 一次创建多级。", {
                  folder: request.path || t("导入的记忆"),
                })
              : request.kind === "rename"
                ? t("只修改名称，位置不变；已有总结会随之保留。")
                : t("选择或输入目标文件夹（留空为根目录）；不存在的文件夹会自动创建。")}
          </DialogDescription>
        </DialogHeader>
        {request.kind !== "create" ? (
          <div className="break-all rounded-md border bg-muted/40 p-2 font-mono text-xs text-muted-foreground">
            {request.path}
          </div>
        ) : null}
        <div className="space-y-2">
          <Label htmlFor={inputId}>
            {request.kind === "move" ? t("目标文件夹") : t("名称")}
          </Label>
          <Input
            id={inputId}
            autoFocus
            value={value}
            list={request.kind === "move" ? listId : undefined}
            placeholder={request.kind === "move" ? t("留空表示根目录") : undefined}
            onChange={(event) => setValue(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") void submit();
            }}
          />
          {request.kind === "move" ? (
            <datalist id={listId}>
              {folders
                .filter(
                  (folder) =>
                    !request.isFolder ||
                    (folder !== request.path && !folder.startsWith(`${request.path}/`)),
                )
                .map((folder) => (
                  <option key={folder} value={folder} />
                ))}
            </datalist>
          ) : null}
        </div>
        <p className="break-all text-xs text-muted-foreground">
          {t("新路径")}：<span className="font-mono">{destination || "—"}</span>
        </p>
        <DialogFooter>
          <Button variant="outline" disabled={busy} onClick={onClose}>
            {t("取消")}
          </Button>
          <Button disabled={invalid || busy} onClick={() => void submit()}>
            {busy ? <LoaderCircle className="animate-spin" /> : null}
            {request.kind === "create" ? t("创建") : t("确定")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export type DeleteRequest = {
  kind: "file" | "folder";
  path: string;
  /** Files inside the folder (folders only). */
  fileCount?: number;
  folderCount?: number;
};

export function KnowledgeDeleteDialog({
  request,
  onClose,
  onConfirm,
}: {
  request: DeleteRequest | null;
  onClose: () => void;
  onConfirm: (request: DeleteRequest) => Promise<void>;
}) {
  const [busy, setBusy] = useState(false);

  const confirm = async () => {
    if (!request || busy) return;
    setBusy(true);
    try {
      await onConfirm(request);
      onClose();
    } catch {
      // Reported by the caller.
    } finally {
      setBusy(false);
    }
  };

  const folder = request?.kind === "folder";
  return (
    <Dialog open={Boolean(request)} onOpenChange={(open) => !open && !busy && onClose()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{folder ? t("删除文件夹") : t("删除知识库文件")}</DialogTitle>
          <DialogDescription>
            {folder
              ? t(
                  "将删除该文件夹及其中的 {files} 个文件、{folders} 个子文件夹，以及它们的总结和索引记录。此操作无法撤销。",
                  {
                    files: formatNumber(request?.fileCount ?? 0),
                    folders: formatNumber(request?.folderCount ?? 0),
                  },
                )
              : t("删除后文件与索引记录一并移除，模型将无法再检索该文件。")}
          </DialogDescription>
        </DialogHeader>
        <div className="break-all rounded-md border bg-muted/40 p-3 font-mono text-xs">
          {request?.path}
          {folder ? "/" : ""}
        </div>
        <DialogFooter>
          <Button variant="outline" disabled={busy} onClick={onClose}>
            {t("取消")}
          </Button>
          <Button variant="destructive" disabled={busy} onClick={() => void confirm()}>
            {busy ? <LoaderCircle className="animate-spin" /> : <Trash2 />}
            {t("确认删除")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
