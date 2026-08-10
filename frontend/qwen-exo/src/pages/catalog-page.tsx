import { useEffect, useState } from "react";
import {
  Boxes,
  CheckCircle2,
  Cpu,
  Database,
  Layers3,
  Server,
  Sparkles,
} from "lucide-react";
import { EmptyState } from "@/components/empty-state";
import { PageHeader } from "@/components/page-header";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { getModelId } from "@/lib/api";
import type { RuntimeStatus } from "@/lib/types";
import { runtimeStateSource, translate as t } from "@/lib/i18n";
import { shortHash } from "@/lib/utils";

export function CatalogPage({ status }: { status: RuntimeStatus | null }) {
  const [modelId, setModelId] = useState<string | null>(null);

  useEffect(() => {
    void getModelId()
      .then(setModelId)
      .catch(() => setModelId(null));
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

  return (
    <div className="page-frame">
      <PageHeader
        eyebrow={t("部署目录")}
        title={t("模型目录")}
        description={t(
          "查看当前服务实际加载的模型身份与后端能力。模型切换属于部署变更，不在运行参数热配置范围内。",
        )}
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

      <div className="mt-5">
        <EmptyState
          icon={Boxes}
          title={t("未公开其他部署槽位")}
          description={t(
            "当前 API 只公开正在服务的模型。要增加可切换模型，需要单独的镜像、权重挂载、Native Bank 隔离与部署级资源准入。",
          )}
        />
      </div>
    </div>
  );
}
