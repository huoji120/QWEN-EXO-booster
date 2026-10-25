import { useEffect, useId, useRef, useState } from "react";
import { LoaderCircle } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { createKnowledgeApi, transferKnowledgeEntry } from "@/lib/api";
import { translate as t } from "@/lib/i18n";
import type { KnowledgeMemorySpace } from "@/lib/types";
import { baseName, errorText, joinPath } from "./knowledge-utils";
import type { TransferSelection } from "./knowledge-drag";

export function KnowledgeTransferDialog({ selection, spaces, signal, onClose, onTransferred }: {
  selection: TransferSelection;
  spaces: KnowledgeMemorySpace[];
  signal: AbortSignal;
  onClose: () => void;
  onTransferred: (selection: TransferSelection) => Promise<void>;
}) {
  const id = useId();
  const [keyId, setKeyId] = useState(selection.destinationKeyId);
  const [folder, setFolder] = useState(selection.folder);
  const [name, setName] = useState(baseName(selection.source.path));
  const [folders, setFolders] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [retry, setRetry] = useState(0);
  const [busy, setBusy] = useState(false);
  const [approved, setApproved] = useState(false);
  const generation = useRef(0);
  const lifetime = useRef<AbortController | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    lifetime.current = controller;
    return () => controller.abort();
  }, []);
  useEffect(() => {
    const controller = new AbortController();
    const current = ++generation.current;
    const combined = AbortSignal.any([signal, controller.signal]);
    setLoading(true);
    setLoadError(null);
    setFolders([]);
    void createKnowledgeApi(keyId ?? undefined, combined).getKnowledgeTree().then((tree) => {
      if (combined.aborted || current !== generation.current) return;
      const paths = new Set(["", ...tree.directories]);
      if (keyId === null) paths.add("work-memory");
      setFolders(Array.from(paths).sort());
      setLoading(false);
    }).catch((error: unknown) => {
      if (combined.aborted || current !== generation.current) return;
      setLoadError(errorText(error, t("未知错误")));
      setLoading(false);
    });
    return () => controller.abort();
  }, [keyId, retry, signal]);

  const source = selection.source;
  const wholeSpace = !source.path;
  const publishing = source.keyId !== null && keyId === null;
  const crossing = source.keyId !== keyId;
  const destination = joinPath(folder, name);
  const invalid = loading || Boolean(loadError) || !folders.includes(folder) ||
    !name.trim() || name !== name.trim() || name === "." || name === ".." || /[\\/\x00-\x1f]/.test(name) ||
    (source.keyId === keyId && (wholeSpace || destination === source.path ||
      (source.kind === "folder" && destination.startsWith(`${source.path}/`)))) ||
    (publishing && !approved);
  const sourceSpace = spaces.find((space) => space.key_id === source.keyId);
  const destinationSpace = spaces.find((space) => space.key_id === keyId);
  const sourceLabel = source.keyId === null ? t("共享记忆") : sourceSpace?.label || source.keyId;
  const destinationLabel = keyId === null ? t("共享记忆") : destinationSpace?.label || keyId;

  const submit = async () => {
    if (invalid || busy || signal.aborted || !lifetime.current || lifetime.current.signal.aborted) return;
    const requestSignal = AbortSignal.any([signal, lifetime.current.signal]);
    const captured: TransferSelection = { source: { ...source }, destinationKeyId: keyId, folder };
    setBusy(true);
    try {
      await transferKnowledgeEntry({
        source_key_id: source.keyId, destination_key_id: keyId,
        source: source.path, destination, confirm_publish: publishing && approved,
      }, requestSignal);
    } catch (error) {
      if (!requestSignal.aborted) {
        toast.error(t("移动失败"), { description: errorText(error, t("未知错误")) });
        setBusy(false);
      }
      return;
    }
    if (requestSignal.aborted) return;
    toast.success(t("已移动"));
    // The move has committed. A refresh failure must not offer a duplicate move.
    try { await onTransferred(captured); }
    catch (error) {
      if (!requestSignal.aborted) toast.error(t("刷新失败"), { description: errorText(error, t("未知错误")) });
    }
    if (!requestSignal.aborted) onClose();
  };

  return <Dialog open onOpenChange={(open) => !open && !busy && onClose()}>
    <DialogContent>
      <DialogHeader>
        <DialogTitle>{wholeSpace ? t("迁移记忆") : t("移动到…")}</DialogTitle>
        <DialogDescription>{t("移动后按目标空间控制读取权限，保留总结，不覆盖已有文件。KEY 读取模式不变。")}</DialogDescription>
      </DialogHeader>
      {wholeSpace ? <p className="text-xs text-muted-foreground">{t("将全部私有知识与反思文件移入新的命名文件夹；不迁移运行状态、回收站或会话记录，不合并已有目录。")}</p> : null}
      <div className="space-y-3">
        <div className="space-y-1">
          <Label htmlFor={`${id}-scope`}>{t("目标空间")}</Label>
          <select id={`${id}-scope`} className="h-9 w-full min-w-0 rounded-md border bg-background px-3 text-sm" disabled={busy} value={keyId ?? ""}
            onChange={(event) => { setKeyId(event.target.value || null); setFolder(event.target.value ? "" : "work-memory"); setApproved(false); }}>
            <option value="">{t("共享记忆")}</option>
            {spaces.map((space) => <option key={space.key_id} value={space.key_id} title={space.key_id}>
              {space.label || space.key_id}{space.deleted ? ` · ${t("已删除")}` : space.revoked ? ` · ${t("已撤销")}` : ""}
            </option>)}
          </select>
        </div>
        <div className="space-y-1">
          <Label htmlFor={`${id}-folder`}>{t("目标文件夹")}</Label>
          <select id={`${id}-folder`} className="h-9 w-full min-w-0 rounded-md border bg-background px-3 text-sm" disabled={busy || loading || Boolean(loadError)} value={folder}
            onChange={(event) => { setFolder(event.target.value); setApproved(false); }}>
            {!folders.includes(folder) ? <option value={folder}>{folder || "/"}</option> : null}
            {folders.map((path) => <option key={path} value={path}>{path || "/"}</option>)}
          </select>
          {loading ? <p className="text-xs text-muted-foreground" role="status">{t("加载中…")}</p> : null}
          {loadError ? <div className="text-xs text-destructive" role="alert">{loadError} <Button variant="ghost" size="sm" onClick={() => setRetry((value) => value + 1)}>{t("重试")}</Button></div> : null}
        </div>
        <div className="space-y-1">
          <Label htmlFor={`${id}-name`}>{wholeSpace ? t("迁移文件夹名称") : t("名称")}</Label>
          <Input id={`${id}-name`} value={name} disabled={busy} autoFocus={wholeSpace} onChange={(event) => { setName(event.target.value); setApproved(false); }} />
        </div>
        <div className="space-y-1 rounded-md border bg-muted/40 p-3 text-xs">
          <p className="truncate" title={`${source.keyId ?? t("共享记忆")} / ${source.path || "/"}`}>{sourceLabel} / {source.path || "/"}</p>
          <p className="truncate" title={`${keyId ?? t("共享记忆")} / ${destination}`}>→ {destinationLabel} / {destination}</p>
        </div>
        {crossing ? <p className="text-xs text-muted-foreground">{publishing ? t("私有 → 共享：所有具有共享读取权限或超级读取权限的 KEY 都可读取这些内容。") : t("目标 KEY 将拥有这些内容；原空间不再保留。超级 KEY 仍可读取。")}</p> : null}
        {crossing && keyId !== null ? <p className="text-xs text-muted-foreground">{t("移入私有空间不会撤回旧会话已经读取的内容。")}</p> : null}
        {publishing ? <label className="flex items-start gap-2 text-sm text-destructive">
          <input type="checkbox" className="mt-1 accent-destructive" checked={approved} disabled={busy} onChange={(event) => setApproved(event.target.checked)} />
          {t("我确认将这些私有记忆发布到共享空间")}
        </label> : null}
      </div>
      <DialogFooter>
        <Button variant="outline" disabled={busy} onClick={onClose}>{t("取消")}</Button>
        <Button disabled={invalid || busy} onClick={() => void submit()}>{busy ? <LoaderCircle className="animate-spin" /> : null}{t("确认移动")}</Button>
      </DialogFooter>
    </DialogContent>
  </Dialog>;
}
