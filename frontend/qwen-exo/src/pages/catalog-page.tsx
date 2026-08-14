import { useEffect, useState } from "react";
import {
  Boxes,
  CheckCircle2,
  Cpu,
  Database,
  Layers3,
  LoaderCircle,
  RefreshCw,
  Server,
  Sparkles,
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
import { Separator } from "@/components/ui/separator";
import {
  getModelCatalog,
  getModelId,
  getStatus,
  selectActiveModel,
} from "@/lib/api";
import type { CatalogModel, ModelCatalog, RuntimeStatus } from "@/lib/types";
import { runtimeStateSource, translate as t } from "@/lib/i18n";
import { formatNumber, shortHash } from "@/lib/utils";

export function CatalogPage({ status }: { status: RuntimeStatus | null }) {
  const [modelId, setModelId] = useState<string | null>(null);
  const [catalog, setCatalog] = useState<ModelCatalog | null>(null);
  const [catalogLoading, setCatalogLoading] = useState(true);
  const [switching, setSwitching] = useState(false);
  const [selectedModel, setSelectedModel] = useState<CatalogModel | null>(null);

  const loadCatalog = async () => {
    setCatalogLoading(true);
    try {
      const [nextCatalog, nextModelId] = await Promise.all([
        getModelCatalog(),
        getModelId().catch(() => null),
      ]);
      setCatalog(nextCatalog);
      setModelId(nextModelId);
    } catch (error) {
      toast.error(t("模型目录加载失败"), {
        description: error instanceof Error ? error.message : t("未知错误"),
      });
    } finally {
      setCatalogLoading(false);
    }
  };

  useEffect(() => {
    void loadCatalog();
  }, []);

  const model = status?.model;
  const modelPath = String(
    model?.model_path || status?.model_path || modelId || t("未知模型"),
  );
  const capabilities = [
    { label: "Responses API", active: Boolean(modelId), icon: Server },
    {
      label: t("TP 推理"),
      active: Number(status?.tp_size || status?.hybrid_state?.tp_size || 0) > 1,
      icon: Layers3,
    },
    {
      label: "Hybrid KV / GDN",
      active: Boolean(status?.hybrid_state?.atomic_full_gdn_lifecycle),
      icon: Cpu,
    },
    {
      label: "Native Tensor Bank",
      active: Boolean(status?.internal_services?.tensor_bank),
      icon: Database,
    },
    {
      label: "In-flight Observer",
      active: status?.observer_mode !== "off",
      icon: Sparkles,
    },
  ];

  const confirmSwitch = async () => {
    if (!catalog || !selectedModel) return;
    const targetModel = selectedModel;
    setSwitching(true);
    try {
      await selectActiveModel(
        targetModel.model_fingerprint,
        catalog.revision,
      );
      toast.success(t("模型切换已提交"), {
        description: t(
          "服务将立即进入受管重启。Knowledge、PolicyData、Cognition 与轨迹保持共享，目标模型会独立编译并使用自己的 Native Bank。",
        ),
      });
      setSelectedModel(null);
      const deadline = Date.now() + 15 * 60 * 1000;
      let completed = false;
      while (Date.now() < deadline) {
        const { promise, resolve } = Promise.withResolvers<void>();
        window.setTimeout(resolve, 5000);
        await promise;
        try {
          const [nextCatalog, nextStatus] = await Promise.all([
            getModelCatalog(),
            getStatus(),
          ]);
          setCatalog(nextCatalog);
          if (
            nextCatalog.active_model_fingerprint === targetModel.model_fingerprint &&
            nextCatalog.applied_model_fingerprint === targetModel.model_fingerprint &&
            nextCatalog.healthy_model_fingerprint === targetModel.model_fingerprint &&
            nextCatalog.running_model_fingerprint === targetModel.model_fingerprint &&
            nextStatus.runtime_state === "ready"
          ) {
            toast.success(t("模型切换完成"));
            completed = true;
            break;
          }
          if (
            nextCatalog.last_failed_model_fingerprint ===
            targetModel.model_fingerprint
          ) {
            toast.error(t("模型启动失败，已回滚"));
            completed = true;
            break;
          }
        } catch {
          // Model weights are loading; the loopback API is temporarily unavailable.
        }
      }
      if (!completed) {
        toast.error(t("模型切换等待超时"), {
          description: t("服务仍在加载，请稍后刷新模型目录检查最终状态。"),
        });
      }
    } catch (error) {
      toast.error(t("模型切换失败"), {
        description: error instanceof Error ? error.message : t("未知错误"),
      });
    } finally {
      setSwitching(false);
      void loadCatalog();
    }
  };

  return (
    <div className="page-frame">
      <PageHeader
        eyebrow={t("部署目录")}
        title={t("模型目录")}
        description={t(
          "发现兼容模型并查看模型专属 Native Bank。Knowledge、PolicyData、Cognition 与轨迹使用同一套 Markdown 或 JSON 源；切换后立即受管重启，由目标模型重新编译原生状态。",
        )}
        actions={
          <Button variant="outline" onClick={() => void loadCatalog()}>
            <RefreshCw className={catalogLoading ? "animate-spin" : ""} />
            {t("刷新")}
          </Button>
        }
      />

      <Card className="overflow-hidden">
        <CardHeader className="border-b bg-card">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
            <div className="flex min-w-0 items-start gap-4">
              <div className="grid h-12 w-12 shrink-0 place-items-center rounded-md border bg-slate-950 text-white">
                <Boxes className="h-5 w-5" />
              </div>
              <div className="min-w-0">
                <div className="eyebrow mb-1">{t("当前部署")}</div>
                <CardTitle className="break-all text-base">
                  {modelId || modelPath.split(/[\\/]/).filter(Boolean).at(-1)}
                </CardTitle>
                <CardDescription className="mt-1 break-all font-mono text-xs">
                  {modelPath}
                </CardDescription>
              </div>
            </div>
            <Badge
              variant={
                status?.runtime_state === "ready" ? "success" : "warning"
              }
            >
              <span
                className="status-dot"
                data-state={status?.runtime_state || "starting"}
              />
              {t(runtimeStateSource(status?.runtime_state, "连接中"))}
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          <div className="grid lg:grid-cols-[minmax(0,1fr)_320px]">
            <div className="p-5">
              <div className="mb-4 text-xs font-semibold">{t("运行能力")}</div>
              <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
                {capabilities.map((capability) => (
                  <div
                    key={capability.label}
                    className="flex items-center gap-3 rounded-md border p-3"
                  >
                    <div className="grid h-8 w-8 place-items-center rounded-md bg-muted">
                      <capability.icon className="h-4 w-4" />
                    </div>
                    <div className="min-w-0 flex-1 text-xs font-medium">
                      {capability.label}
                    </div>
                    {capability.active ? (
                      <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                    ) : (
                      <span className="h-2 w-2 rounded-full bg-slate-300" />
                    )}
                  </div>
                ))}
              </div>
            </div>
            <div className="border-t bg-muted/25 p-5 lg:border-l lg:border-t-0">
              <div className="text-xs font-semibold">{t("模型身份")}</div>
              <div className="mt-4 space-y-4 text-xs">
                <div>
                  <div className="text-muted-foreground">
                    {t("API 模型 ID")}
                  </div>
                  <div className="mt-1 break-all font-mono">
                    {modelId || "—"}
                  </div>
                </div>
                <Separator />
                <div>
                  <div className="text-muted-foreground">{t("模型指纹")}</div>
                  <div className="mt-1 font-mono">
                    {shortHash(String(model?.fingerprint || ""), 24)}
                  </div>
                </div>
                <Separator />
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-muted-foreground">{t("TP 规模")}</div>
                    <div className="mt-1 font-mono">
                      {String(
                        status?.tp_size || status?.hybrid_state?.tp_size || "—",
                      )}
                    </div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">{t("数据类型")}</div>
                    <div className="mt-1 font-mono">
                      {status?.hybrid_state?.dtype || "—"}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="mt-5 grid gap-4 xl:grid-cols-2">
        {catalog && !catalog.managed_restart ? (
          <Card className="border-amber-300/70 bg-amber-50/60 xl:col-span-2 dark:bg-amber-950/15">
            <CardContent className="p-4 text-xs text-amber-950 dark:text-amber-100">
              {t(
                "当前进程未启用受管重启，模型目录仅可查看，不能从控制台切换。",
              )}
            </CardContent>
          </Card>
        ) : null}
        {catalogLoading && !catalog ? (
          <Card className="xl:col-span-2">
            <CardContent className="grid min-h-48 place-items-center">
              <LoaderCircle className="h-5 w-5 animate-spin text-muted-foreground" />
            </CardContent>
          </Card>
        ) : null}
        {catalog?.models.map((entry) => (
          <Card key={entry.model_fingerprint} className="overflow-hidden">
            <CardHeader className="border-b">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <CardTitle className="text-sm">{entry.name}</CardTitle>
                    {entry.running ? (
                      <Badge variant="success">{t("运行中")}</Badge>
                    ) : null}
                    {entry.active && !entry.running ? (
                      <Badge variant="warning">{t("等待重启")}</Badge>
                    ) : null}
                  </div>
                  <CardDescription className="mt-1 break-all font-mono text-[10px]">
                    {entry.model_path}
                  </CardDescription>
                </div>
                <div className="flex flex-wrap justify-end gap-1">
                  <Badge variant="outline">{entry.variant}</Badge>
                  <Badge variant="outline">
                    {entry.checkpoint_quantization === "gptq"
                      ? `W${entry.checkpoint_quantization_bits || 4}A16`
                      : entry.checkpoint_quantization === "fp8"
                        ? t("FP8 权重")
                        : t("BF16 权重")}
                  </Badge>
                  {entry.runtime_quantization ? (
                    <Badge variant="outline">
                      {t("加载器 {value}", {
                        value: entry.runtime_quantization,
                      })}
                    </Badge>
                  ) : null}
                  <Badge variant="outline">FP8 KV</Badge>
                </div>
              </div>
            </CardHeader>
            <CardContent className="p-4">
              <div className="grid grid-cols-2 gap-3 text-xs sm:grid-cols-4">
                <div>
                  <div className="text-muted-foreground">Knowledge</div>
                  <div className="mt-1 font-mono font-semibold">
                    {formatNumber(entry.knowledge_document_count)}
                  </div>
                </div>
                <div>
                  <div className="text-muted-foreground">PolicyData</div>
                  <div className="mt-1 font-mono font-semibold">
                    {formatNumber(entry.policy_document_count)}
                  </div>
                </div>
                <div>
                  <div className="text-muted-foreground">Cognition</div>
                  <div className="mt-1 font-mono font-semibold">
                    {formatNumber(entry.cognition_document_count)}
                  </div>
                </div>
                <div>
                  <div className="text-muted-foreground">Native Bank</div>
                  <div className="mt-1 font-semibold">
                    {entry.native_bank_ready ? t("本模型已编译") : t("等待本模型编译")}
                  </div>
                </div>
              </div>
              <div className="mt-3 rounded-md border bg-muted/30 px-3 py-2 text-[11px] text-muted-foreground">
                {t("前三项来自所有模型共用的源目录；Native Bank 按模型指纹与运行拓扑独立存储。")}
              </div>
              <div className="mt-4 flex items-center justify-between border-t pt-3">
                <div className="font-mono text-[10px] text-muted-foreground">
                  {shortHash(entry.model_fingerprint, 18)} ·{" "}
                  {formatNumber(entry.layer_count)} layers
                </div>
                <Button
                  size="sm"
                  disabled={
                    entry.running || switching || !catalog.managed_restart
                  }
                  onClick={() => setSelectedModel(entry)}
                >
                  {entry.running ? t("当前模型") : t("切换到此模型")}
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <Dialog
        open={Boolean(selectedModel)}
        onOpenChange={(open) => !open && setSelectedModel(null)}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t("确认切换模型")}</DialogTitle>
            <DialogDescription>
              {t(
                "切换会中断当前推理并立即重启模型服务。Knowledge、PolicyData、Cognition 与轨迹不复制、不分叉；目标模型使用共享源重新编译自己的 Native Bank，其他模型的编译产物不会被覆盖。",
              )}
            </DialogDescription>
          </DialogHeader>
          <div className="rounded-md border bg-muted/35 p-3">
            <div className="text-sm font-semibold">{selectedModel?.name}</div>
            <div className="mt-1 break-all font-mono text-[10px] text-muted-foreground">
              {selectedModel?.model_path}
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setSelectedModel(null)}>
              {t("取消")}
            </Button>
            <Button onClick={() => void confirmSwitch()} disabled={switching}>
              {switching ? <LoaderCircle className="animate-spin" /> : null}
              {t("重启并切换")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
