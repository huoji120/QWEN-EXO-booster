import { useEffect, useState } from "react";
import {
  Check,
  Copy,
  KeyRound,
  LoaderCircle,
  Plus,
  ShieldCheck,
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
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { createApiKey, getApiKeys, revokeApiKey } from "@/lib/api";
import { useI18n } from "@/lib/i18n";
import type { ApiKeyInfo, ApiKeyListing, CreatedApiKey } from "@/lib/types";
import { formatTime } from "@/lib/utils";

export function ApiKeysPage() {
  const { t } = useI18n();
  const [listing, setListing] = useState<ApiKeyListing | null>(null);
  const [loading, setLoading] = useState(true);
  const [label, setLabel] = useState("");
  const [creating, setCreating] = useState(false);
  const [created, setCreated] = useState<CreatedApiKey | null>(null);
  const [copied, setCopied] = useState(false);
  const [revoking, setRevoking] = useState<ApiKeyInfo | null>(null);

  const load = async () => {
    setLoading(true);
    try {
      setListing(await getApiKeys());
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
      const key = await createApiKey(normalized);
      setCreated(key);
      setLabel("");
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

  const activeCount =
    listing?.keys.filter((key) => !key.revoked_at).length || 0;

  return (
    <div className="page-frame">
      <PageHeader
        eyebrow={t("访问控制")}
        title={t("API 密钥")}
        description={t(
          "签发和吊销用于 DuckGPT Responses、上下文压缩与模型列表的 Bearer 密钥。明文只在签发时显示一次。",
        )}
      />

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_340px]">
        <Card>
          <CardHeader className="border-b">
            <div className="flex items-start justify-between gap-4">
              <div>
                <CardTitle>{t("已签发密钥")}</CardTitle>
                <CardDescription>
                  {t("服务每次请求读取持久化密钥表；签发和吊销无需重启模型。")}
                </CardDescription>
              </div>
              <Badge variant="secondary">
                {t("{count} 个有效", { count: activeCount })}
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="p-0">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>{t("名称")}</TableHead>
                  <TableHead>{t("密钥 ID")}</TableHead>
                  <TableHead>{t("创建时间")}</TableHead>
                  <TableHead>{t("状态")}</TableHead>
                  <TableHead className="text-right">{t("操作")}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {loading ? (
                  <TableRow>
                    <TableCell
                      colSpan={5}
                      className="h-28 text-center text-muted-foreground"
                    >
                      <LoaderCircle className="mx-auto h-5 w-5 animate-spin" />
                    </TableCell>
                  </TableRow>
                ) : listing?.keys.length ? (
                  listing.keys.map((key) => {
                    const active = !key.revoked_at;
                    return (
                      <TableRow key={key.id}>
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
                        <TableCell className="text-right">
                          <Button
                            variant="ghost"
                            size="sm"
                            disabled={!active}
                            onClick={() => setRevoking(key)}
                          >
                            {t("吊销")}
                          </Button>
                        </TableCell>
                      </TableRow>
                    );
                  })
                ) : (
                  <TableRow>
                    <TableCell
                      colSpan={5}
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
    </div>
  );
}
