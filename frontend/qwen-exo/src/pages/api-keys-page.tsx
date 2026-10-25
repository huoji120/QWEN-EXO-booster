import { useEffect, useMemo, useState } from "react";
import {
  Check,
  Copy,
  KeyRound,
  LoaderCircle,
  Plus,
  ShieldCheck,
  Trash2,
  X,
} from "lucide-react";
import { toast } from "sonner";
import { PageHeader } from "@/components/page-header";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
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
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  createApiKey,
  deleteApiKeys,
  getApiKeys,
  revokeApiKey,
  updateApiKeyMemoryScope,
} from "@/lib/api";
import { useI18n } from "@/lib/i18n";
import type {
  ApiKeyInfo,
  ApiKeyListing,
  ApiKeyMemoryScope,
  CreatedApiKey,
} from "@/lib/types";
import { formatTime } from "@/lib/utils";

const CHECKBOX_CLASS = "h-4 w-4 cursor-pointer accent-slate-900";

const MEMORY_SCOPE_DESCRIPTIONS: Record<ApiKeyMemoryScope, string> = {
  shared: "读写共享记忆，不读取任何私有记忆。",
  private: "只写本密钥的私有记忆，读取本密钥的私有记忆和共享记忆。",
  private_only: "仅读写本密钥的私有记忆，不读取共享记忆、共享提示或共享原生记忆。",
  super: "读取共享记忆和所有密钥的私有记忆（包括已撤销或已删除密钥的保留记忆），只写本密钥的私有记忆；不能修改其他密钥的记忆，也不授予密钥管理权限。",
};

export function ApiKeysPage() {
  const { t } = useI18n();
  const [listing, setListing] = useState<ApiKeyListing | null>(null);
  const [loading, setLoading] = useState(true);
  const [label, setLabel] = useState("");
  const [memoryScope, setMemoryScope] = useState<ApiKeyMemoryScope>("shared");
  const [updatingScope, setUpdatingScope] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [created, setCreated] = useState<CreatedApiKey | null>(null);
  const [copied, setCopied] = useState(false);
  const [revoking, setRevoking] = useState<ApiKeyInfo | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [deleting, setDeleting] = useState<ApiKeyInfo[] | null>(null);
  const [deleteBusy, setDeleteBusy] = useState(false);

  const load = async () => {
    setLoading(true);
    try {
      const next = await getApiKeys();
      setListing(next);
      // Drop selections for keys that no longer exist.
      const ids = new Set(next.keys.map((key) => key.id));
      setSelected((current) => {
        const kept = new Set([...current].filter((id) => ids.has(id)));
        return kept.size === current.size ? current : kept;
      });
    } catch (error) {
      toast.error(t("API 密钥加载失败"), {
        description: error instanceof Error ? error.message : t("未知错误"),
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void load();
  }, []);

  const issue = async () => {
    const normalized = label.trim();
    if (!normalized) return;
    setCreating(true);
    try {
      const key = await createApiKey(normalized, memoryScope);
      setCreated(key);
      setLabel("");
      setMemoryScope("shared");
      setCopied(false);
      await load();
      toast.success(t("API 密钥已签发"));
    } catch (error) {
      toast.error(t("API 密钥签发失败"), {
        description: error instanceof Error ? error.message : t("未知错误"),
      });
    } finally {
      setCreating(false);
    }
  };

  const changeMemoryScope = async (key: ApiKeyInfo, scope: ApiKeyMemoryScope) => {
    if (scope === key.memory_scope || updatingScope) return;
    setUpdatingScope(key.id);
    try {
      const updated = await updateApiKeyMemoryScope(key.id, scope);
      setListing((current) =>
        current
          ? {
              ...current,
              revision: updated.revision,
              keys: current.keys.map((item) =>
                item.id === updated.id ? updated : item,
              ),
            }
          : current,
      );
      toast.success(t("记忆模式已更新"), { description: key.label });
    } catch (error) {
      toast.error(t("记忆模式更新失败"), {
        description: error instanceof Error ? error.message : t("未知错误"),
      });
    } finally {
      setUpdatingScope(null);
    }
  };

  const copyToken = async () => {
    if (!created) return;
    await navigator.clipboard.writeText(created.token);
    setCopied(true);
    toast.success(t("密钥已复制"));
  };

  const revoke = async () => {
    if (!revoking) return;
    const target = revoking;
    try {
      await revokeApiKey(target.id);
      setRevoking(null);
      await load();
      toast.success(t("API 密钥已吊销"), { description: target.label });
    } catch (error) {
      toast.error(t("API 密钥吊销失败"), {
        description: error instanceof Error ? error.message : t("未知错误"),
      });
    }
  };

  const remove = async () => {
    if (!deleting?.length) return;
    const targets = deleting;
    setDeleteBusy(true);
    try {
      const result = await deleteApiKeys(targets.map((key) => key.id));
      setDeleting(null);
      setSelected(new Set());
      await load();
      toast.success(t("API 密钥已删除"), {
        description: t("已删除 {count} 个密钥", {
          count: result.deleted.length,
        }),
      });
    } catch (error) {
      toast.error(t("API 密钥删除失败"), {
        description: error instanceof Error ? error.message : t("未知错误"),
      });
    } finally {
      setDeleteBusy(false);
    }
  };

  const keys = listing?.keys ?? [];
  const activeCount = keys.filter((key) => !key.revoked_at).length;
  const revokedIds = useMemo(
    () => keys.filter((key) => key.revoked_at).map((key) => key.id),
    [keys],
  );
  const allSelected = keys.length > 0 && keys.every((key) => selected.has(key.id));
  const selectedKeys = keys.filter((key) => selected.has(key.id));

  const toggleOne = (id: string, checked: boolean) => {
    setSelected((current) => {
      const next = new Set(current);
      if (checked) next.add(id);
      else next.delete(id);
      return next;
    });
  };

  const toggleAll = (checked: boolean) => {
    setSelected(checked ? new Set(keys.map((key) => key.id)) : new Set());
  };

  return (
    <div className="page-frame">
      <PageHeader
        eyebrow={t("访问控制")}
        title={t("API 密钥")}
        description={t(
          "签发、吊销和删除用于 DuckGPT Responses、上下文压缩与模型列表的 Bearer 密钥。明文只在签发时显示一次。",
        )}
      />

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_340px]">
        <Card>
          <CardHeader className="border-b">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div>
                <CardTitle>{t("已签发密钥")}</CardTitle>
                <CardDescription>
                  {t("服务每次请求读取持久化密钥表；签发和吊销无需重启模型。")}
                </CardDescription>
                <CardDescription className="mt-2">
                  {t("共享：只读写共享记忆；私有＋共享：读取共享及本密钥私有记忆，只写自己的私有记忆；仅私有：仅读写自己的私有记忆，不读取共享记忆、共享提示或共享原生记忆；超级读取：读取共享及所有私有记忆，只写自己的私有记忆，不授予其他记忆的修改权或密钥管理权限。")}
                </CardDescription>
                <CardDescription className="mt-2">
                  {t("切换模式仅影响后续请求，不迁移或公开已有私有记忆；旧模式下的历史响应可能无法继续或重放。密钥 ID 和令牌保持不变。")}
                </CardDescription>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <Badge variant="secondary">
                  {t("{count} 个有效", { count: activeCount })}
                </Badge>
                <Button
                  variant="outline"
                  size="sm"
                  disabled={revokedIds.length === 0}
                  onClick={() => setSelected(new Set(revokedIds))}
                >
                  {t("选中全部已吊销")}
                </Button>
                <Button
                  variant="destructive"
                  size="sm"
                  disabled={selectedKeys.length === 0}
                  onClick={() => setDeleting(selectedKeys)}
                >
                  <Trash2 />
                  {t("删除选中")} ({selectedKeys.length})
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent className="p-0">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-10">
                    <input
                      type="checkbox"
                      className={CHECKBOX_CLASS}
                      aria-label={t("全选")}
                      checked={allSelected}
                      disabled={keys.length === 0}
                      onChange={(event) => toggleAll(event.target.checked)}
                    />
                  </TableHead>
                  <TableHead>{t("名称")}</TableHead>
                  <TableHead>{t("密钥 ID")}</TableHead>
                  <TableHead>{t("创建时间")}</TableHead>
                  <TableHead>{t("状态")}</TableHead>
                  <TableHead>{t("记忆模式")}</TableHead>
                  <TableHead className="text-right">{t("操作")}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {loading ? (
                  <TableRow>
                    <TableCell
                      colSpan={7}
                      className="h-28 text-center text-muted-foreground"
                    >
                      <LoaderCircle className="mx-auto h-5 w-5 animate-spin" />
                    </TableCell>
                  </TableRow>
                ) : keys.length ? (
                  keys.map((key) => {
                    const active = !key.revoked_at;
                    return (
                      <TableRow
                        key={key.id}
                        data-state={selected.has(key.id) ? "selected" : undefined}
                      >
                        <TableCell>
                          <input
                            type="checkbox"
                            className={CHECKBOX_CLASS}
                            aria-label={key.label}
                            checked={selected.has(key.id)}
                            onChange={(event) =>
                              toggleOne(key.id, event.target.checked)
                            }
                          />
                        </TableCell>
                        <TableCell className="font-medium">
                          {key.label}
                        </TableCell>
                        <TableCell className="font-mono text-xs">
                          {key.id}
                        </TableCell>
                        <TableCell className="text-xs text-muted-foreground">
                          {formatTime(key.created_at)}
                        </TableCell>
                        <TableCell>
                          <Badge variant={active ? "success" : "secondary"}>
                            {active ? <Check /> : <X />}
                            {active ? t("有效") : t("已吊销")}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            <Select
                              value={key.memory_scope}
                              disabled={!active || updatingScope !== null}
                              onValueChange={(value) =>
                                void changeMemoryScope(key, value as ApiKeyMemoryScope)
                              }
                            >
                              <SelectTrigger
                                className="w-36"
                                aria-label={t("{name} 的记忆模式", { name: key.label })}
                              >
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="shared">{t("共享")}</SelectItem>
                                <SelectItem value="private">{t("私有＋共享")}</SelectItem>
                                <SelectItem value="private_only">{t("仅私有")}</SelectItem>
                                <SelectItem value="super">{t("超级读取")}</SelectItem>
                              </SelectContent>
                            </Select>
                            {updatingScope === key.id && (
                              <LoaderCircle className="h-4 w-4 animate-spin" />
                            )}
                          </div>
                        </TableCell>
                        <TableCell className="text-right">
                          <Button
                            variant="ghost"
                            size="sm"
                            disabled={!active}
                            onClick={() => setRevoking(key)}
                          >
                            {t("吊销")}
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="text-destructive hover:text-destructive"
                            onClick={() => setDeleting([key])}
                          >
                            {t("删除")}
                          </Button>
                        </TableCell>
                      </TableRow>
                    );
                  })
                ) : (
                  <TableRow>
                    <TableCell
                      colSpan={7}
                      className="h-28 text-center text-muted-foreground"
                    >
                      {t("尚未签发 API 密钥")}
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </CardContent>
        </Card>

        <Card className="h-fit">
          <CardHeader>
            <div className="mb-2 grid h-10 w-10 place-items-center rounded-md bg-slate-950 text-white">
              <KeyRound className="h-5 w-5" />
            </div>
            <CardTitle>{t("签发新密钥")}</CardTitle>
            <CardDescription>
              {t("使用可识别的用途名称；系统只保存 SHA-256 摘要，不保存明文。")}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <Input
              value={label}
              maxLength={80}
              placeholder={t("例如：OpenCode 工作站")}
              onChange={(event) => setLabel(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") void issue();
              }}
            />
            <div className="space-y-2">
              <label htmlFor="new-key-memory-scope" className="text-sm font-medium">
                {t("记忆模式")}
              </label>
              <Select
                value={memoryScope}
                disabled={creating}
                onValueChange={(value) => setMemoryScope(value as ApiKeyMemoryScope)}
              >
                <SelectTrigger id="new-key-memory-scope">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="shared">{t("共享")}</SelectItem>
                  <SelectItem value="private">{t("私有＋共享")}</SelectItem>
                  <SelectItem value="private_only">{t("仅私有")}</SelectItem>
                  <SelectItem value="super">{t("超级读取")}</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs leading-5 text-muted-foreground">
                {t(MEMORY_SCOPE_DESCRIPTIONS[memoryScope])}
              </p>
            </div>
            <Button
              className="w-full"
              disabled={creating || !label.trim()}
              onClick={() => void issue()}
            >
              {creating ? <LoaderCircle className="animate-spin" /> : <Plus />}
              {t("签发密钥")}
            </Button>
            <div className="flex gap-2 rounded-md border bg-muted/40 p-3 text-xs leading-5 text-muted-foreground">
              <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0" />
              <span>
                {t("密钥可立即用于公网入口；吊销后下一次请求立即失效。")}
              </span>
            </div>
          </CardContent>
        </Card>
      </div>

      <Dialog
        open={Boolean(created)}
        onOpenChange={(open) => !open && setCreated(null)}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t("保存 API 密钥")}</DialogTitle>
            <DialogDescription>
              {t(
                "这是唯一一次显示完整密钥。关闭后无法再次读取，只能吊销并重新签发。",
              )}
            </DialogDescription>
          </DialogHeader>
          <div className="rounded-md border bg-slate-950 p-4 font-mono text-xs leading-5 text-slate-100 break-all select-all">
            {created?.token}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setCreated(null)}>
              {t("我已保存")}
            </Button>
            <Button onClick={() => void copyToken()}>
              {copied ? <Check /> : <Copy />}
              {copied ? t("已复制") : t("复制密钥")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={Boolean(revoking)}
        onOpenChange={(open) => !open && setRevoking(null)}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t("吊销 API 密钥")}</DialogTitle>
            <DialogDescription>
              {t("吊销后使用该密钥的客户端将立即收到 401，且无法恢复。")}
            </DialogDescription>
          </DialogHeader>
          <div className="rounded-md border bg-muted/40 p-3">
            <div className="font-medium">{revoking?.label}</div>
            <div className="mt-1 font-mono text-xs text-muted-foreground">
              {revoking?.id}
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setRevoking(null)}>
              {t("取消")}
            </Button>
            <Button variant="destructive" onClick={() => void revoke()}>
              {t("确认吊销")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={Boolean(deleting)}
        onOpenChange={(open) => !open && !deleteBusy && setDeleting(null)}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t("删除 API 密钥")}</DialogTitle>
            <DialogDescription>
              {t(
                "删除会同时移除记录和访问权限：有效密钥立即失效，且无法恢复。",
              )}
            </DialogDescription>
          </DialogHeader>
          <div className="max-h-64 overflow-y-auto rounded-md border bg-muted/40 p-3 text-sm">
            <div className="mb-2 font-medium">
              {t("将删除 {count} 个密钥", { count: deleting?.length ?? 0 })}
            </div>
            <ul className="space-y-1">
              {deleting?.map((key) => (
                <li key={key.id} className="flex flex-wrap items-baseline gap-2">
                  <span className="font-medium">{key.label}</span>
                  <span className="font-mono text-xs text-muted-foreground">
                    {key.id}
                  </span>
                  {!key.revoked_at && (
                    <Badge variant="success">{t("有效")}</Badge>
                  )}
                </li>
              ))}
            </ul>
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              disabled={deleteBusy}
              onClick={() => setDeleting(null)}
            >
              {t("取消")}
            </Button>
            <Button
              variant="destructive"
              disabled={deleteBusy}
              onClick={() => void remove()}
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
