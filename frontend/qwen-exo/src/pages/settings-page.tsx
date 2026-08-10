import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  LoaderCircle,
  RefreshCw,
  RotateCcw,
  Save,
  ServerCog,
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
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { getServiceConfig, getStatus, updateServiceConfig } from "@/lib/api";
import {
  runtimeStateSource,
  translate as t,
  type TranslationValues,
} from "@/lib/i18n";
import type { RuntimeStatus, ServiceConfig, ServiceSetting } from "@/lib/types";
import { settingCopy } from "@/lib/settings-copy";
import { cn, formatNumber, formatTime, shortHash } from "@/lib/utils";

const RESTART_STEPS = [
  "写入原子配置",
  "停止当前进程",
  "Docker 托管拉起",
  "模型与 Tensor Bank 就绪",
];

const SETTING_UNIT_SOURCES: Record<string, string> = {
  attempts: "次",
  events: "个事件",
  seconds: "秒",
  tokens: "token",
  turns: "轮",
};

type LocalizedMessage =
  | { source: string; values?: TranslationValues }
  | { text: string };

function localizedMessageText(message: LocalizedMessage) {
  if ("text" in message) return message.text;
  const values = message.values
    ? Object.fromEntries(
        Object.entries(message.values).map(([key, value]) => [
          key,
          typeof value === "string" ? t(value) : value,
        ]),
      )
    : undefined;
  return t(message.source, values);
}

function SettingControl({
  id,
  setting,
  value,
  disabled,
  onChange,
}: {
  id: string;
  setting: ServiceSetting;
  value: boolean | number | string;
  disabled: boolean;
  onChange: (value: boolean | number | string) => void;
}) {
  if (setting.type === "boolean") {
    return (
      <Switch
        id={id}
        checked={Boolean(value)}
        onCheckedChange={onChange}
        disabled={disabled}
        aria-label={t(setting.label)}
      />
    );
  }
  if (setting.choices?.length) {
    return (
      <Select
        value={String(value)}
        onValueChange={onChange}
        disabled={disabled}
      >
        <SelectTrigger id={id} aria-label={t(setting.label)}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {setting.choices.map((choice) => (
            <SelectItem key={choice} value={choice}>
              {t(setting.choice_labels?.[choice] ?? choice)}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    );
  }
  return (
    <div className="relative">
      <Input
        id={id}
        type="number"
        value={Number(value)}
        min={setting.minimum}
        max={setting.maximum}
        step={setting.step || (setting.type === "integer" ? 1 : "any")}
        onChange={(event) =>
          onChange(
            setting.type === "integer"
              ? Number.parseInt(event.target.value || "0", 10)
              : Number.parseFloat(event.target.value || "0"),
          )
        }
        disabled={disabled}
        className={setting.unit ? "pr-16" : undefined}
      />
      {setting.unit ? (
        <span className="pointer-events-none absolute inset-y-0 right-3 flex items-center text-[10px] text-muted-foreground">
          {t(SETTING_UNIT_SOURCES[setting.unit] || setting.unit)}
        </span>
      ) : null}
    </div>
  );
}

function delay(milliseconds: number) {
  return new Promise<void>((resolve) =>
    window.setTimeout(resolve, milliseconds),
  );
}

class ManagedRollbackError extends Error {}

export function SettingsPage({
  status,
  onStatusRefresh,
}: {
  status: RuntimeStatus | null;
  onStatusRefresh: () => Promise<void>;
}) {
  const [config, setConfig] = useState<ServiceConfig | null>(null);
  const [draft, setDraft] = useState<Record<string, boolean | number | string>>(
    {},
  );
  const [activeGroup, setActiveGroup] = useState("capacity");
  const [loading, setLoading] = useState(true);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [restarting, setRestarting] = useState(false);
  const [restartStep, setRestartStep] = useState(0);
  const [restartMessage, setRestartMessage] = useState<LocalizedMessage>({
    source: "",
  });

  const load = async () => {
    setLoading(true);
    try {
      const next = await getServiceConfig();
      setConfig(next);
      setDraft(next.values);
    } catch (error) {
      toast.error(t("服务配置加载失败"), {
        description: error instanceof Error ? error.message : t("未知错误"),
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void load();
  }, []);

  const changedKeys = useMemo(() => {
    if (!config) return [];
    return config.settings
      .filter((setting) => draft[setting.key] !== config.values[setting.key])
      .map((setting) => setting.key);
  }, [config, draft]);

  const visibleSettings =
    config?.settings.filter((setting) => setting.group === activeGroup) || [];
  const dirty = changedKeys.length > 0;

  const saveAndRestart = async () => {
    if (!config || !dirty) return;
    setRestarting(true);
    setRestartStep(0);
    setRestartMessage({ source: "正在验证并写入 service-config.json" });
    try {
      const accepted = await updateServiceConfig(draft, config.revision);
      const targetRevision = accepted.revision;
      setConfig(accepted);
      setRestartStep(1);
      setRestartMessage({
        source: "当前 HTTP 连接即将关闭；Docker 会使用新配置重新拉起服务",
      });
      await delay(1600);
      setRestartStep(2);

      let lastError = "等待服务监听端口";
      for (let attempt = 0; attempt < 180; attempt += 1) {
        try {
          const [nextConfig, nextStatus] = await Promise.all([
            getServiceConfig(),
            getStatus(),
          ]);
          setRestartMessage({
            source: "运行时 {state} · 配置 {revision}",
            values: {
              state: runtimeStateSource(nextStatus.runtime_state),
              revision: shortHash(nextConfig.applied_revision, 12),
            },
          });
          if (
            nextConfig.last_failed_revision === targetRevision &&
            nextConfig.revision !== targetRevision
          ) {
            setConfig(nextConfig);
            setDraft(nextConfig.values);
            throw new ManagedRollbackError(
              t("新配置启动失败，托管启动器已自动回滚到上一健康版本"),
            );
          }
          if (
            nextConfig.healthy_revision === targetRevision &&
            nextStatus.runtime_state === "ready"
          ) {
            setRestartStep(3);
            setConfig(nextConfig);
            setDraft(nextConfig.values);
            await onStatusRefresh();
            toast.success(t("服务已使用新配置恢复就绪"), {
              description: `revision ${targetRevision}`,
            });
            await delay(700);
            setConfirmOpen(false);
            setRestarting(false);
            return;
          }
          lastError = t("运行时仍为 {state}", {
            state: nextStatus.runtime_state,
          });
        } catch (error) {
          lastError =
            error instanceof Error ? error.message : t("服务正在启动");
          if (error instanceof ManagedRollbackError) throw error;
        }
        await delay(2000);
      }
      throw new Error(t("等待服务恢复超时：{error}", { error: lastError }));
    } catch (error) {
      const message = error instanceof Error ? error.message : t("未知错误");
      setRestartMessage({ text: message });
      toast.error(t("配置未能生效"), { description: message, duration: 10000 });
      setRestarting(false);
    }
  };

  const defaultsForGroup = () => {
    if (!config) return;
    setDraft((current) => ({
      ...current,
      ...Object.fromEntries(
        visibleSettings.map((setting) => [setting.key, setting.default]),
      ),
    }));
  };

  return (
    <div className="page-frame">
      <PageHeader
        eyebrow={t("托管运行时")}
        title={t("服务配置")}
        description={t(
          "所有参数写入同一 revision 配置并通过托管重启生效；配置列表页仅展示当前生效值，不提供覆盖式编辑。",
        )}
        actions={
          <>
            <Button
              variant="outline"
              size="sm"
              onClick={() => void load()}
              disabled={loading || restarting}
            >
              <RefreshCw className={loading ? "animate-spin" : ""} />
              {t("重新读取")}
            </Button>
            <Button
              size="sm"
              onClick={() => setConfirmOpen(true)}
              disabled={
                !dirty || loading || restarting || !config?.managed_restart
              }
            >
              <Save />
              {t("保存并重启")}
              {dirty ? ` (${formatNumber(changedKeys.length)})` : ""}
            </Button>
          </>
        }
      />

      {!config && loading ? (
        <div className="grid gap-4 lg:grid-cols-[240px_minmax(0,1fr)]">
          <Skeleton className="h-96" />
          <Skeleton className="h-[560px]" />
        </div>
      ) : null}

      {config ? (
        <>
          {!config.managed_restart ? (
            <div className="mb-4 flex items-start gap-3 border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
              <div>
                <div className="font-semibold">
                  {t("当前进程未启用托管重启")}
                </div>
                <p className="mt-1 text-xs leading-5">
                  {t(
                    "配置保持只读。使用 qwen_exo_booster.service_launcher 且设置 QWEN_EXO_MANAGED_RESTART=1 后才能从控制台保存。",
                  )}
                </p>
              </div>
            </div>
          ) : null}

          <div className="grid gap-5 xl:grid-cols-[244px_minmax(0,1fr)]">
            <aside className="space-y-4">
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle>{t("配置分组")}</CardTitle>
                  <CardDescription>
                    {t("{count} 项受控参数", {
                      count: formatNumber(config.settings.length),
                    })}
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-1">
                  {config.groups.map((group) => {
                    const changed = config.settings.filter(
                      (setting) =>
                        setting.group === group.id &&
                        changedKeys.includes(setting.key),
                    ).length;
                    return (
                      <button
                        key={group.id}
                        onClick={() => setActiveGroup(group.id)}
                        className={cn(
                          "flex w-full items-center justify-between rounded-md px-3 py-2 text-left text-xs font-medium",
                          activeGroup === group.id
                            ? "bg-foreground text-background"
                            : "text-muted-foreground hover:bg-muted hover:text-foreground",
                        )}
                      >
                        <span>{t(group.label)}</span>
                        {changed ? (
                          <span
                            className={cn(
                              "font-mono text-[10px]",
                              activeGroup === group.id
                                ? "text-slate-300"
                                : "text-primary",
                            )}
                          >
                            +{formatNumber(changed)}
                          </span>
                        ) : null}
                      </button>
                    );
                  })}
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-3">
                  <CardTitle>{t("部署状态")}</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3 text-xs">
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">{t("运行时")}</span>
                    <Badge
                      variant={
                        status?.runtime_state === "ready"
                          ? "success"
                          : "warning"
                      }
                    >
                      {t(runtimeStateSource(status?.runtime_state, "不可用"))}
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">
                      {t("当前 revision")}
                    </span>
                    <span className="font-mono">
                      {shortHash(config.revision, 12)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">
                      {t("健康 revision")}
                    </span>
                    <span className="font-mono">
                      {shortHash(config.healthy_revision, 12)}
                    </span>
                  </div>
                  <div className="border-t pt-3">
                    <div className="text-muted-foreground">{t("最近应用")}</div>
                    <div className="mt-1">{formatTime(config.applied_at)}</div>
                  </div>
                  {config.last_rollback_at ? (
                    <div className="border-t pt-3 text-amber-700">
                      <div className="font-medium">{t("检测到自动回滚")}</div>
                      <div className="mt-1 font-mono">
                        {t("失败")} {shortHash(config.last_failed_revision, 12)}
                      </div>
                    </div>
                  ) : null}
                </CardContent>
              </Card>
            </aside>

            <Card>
              <CardHeader className="border-b">
                <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-start">
                  <div>
                    <CardTitle>
                      {t(
                        config.groups.find((group) => group.id === activeGroup)
                          ?.label || "",
                      )}
                    </CardTitle>
                    <CardDescription>
                      {t(
                        config.groups.find((group) => group.id === activeGroup)
                          ?.description || "",
                      )}
                    </CardDescription>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={defaultsForGroup}
                    disabled={restarting}
                  >
                    <RotateCcw />
                    {t("恢复本组默认值")}
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="px-5 py-0">
                {visibleSettings.map((setting) => {
                  const copy = settingCopy(
                    setting.key,
                    setting.label,
                    setting.description,
                  );
                  return (
                    <div key={setting.key} className="field-grid">
                      <div className="min-w-0">
                        <div className="flex items-center gap-2">
                          <Label htmlFor={setting.key}>{copy.label}</Label>
                          {changedKeys.includes(setting.key) ? (
                            <Badge>{t("已修改")}</Badge>
                          ) : null}
                        </div>
                        <p className="mt-1.5 text-xs leading-5 text-muted-foreground">
                          {copy.summary}
                        </p>
                        <div className="mt-1 font-mono text-[10px] text-slate-400">
                          {setting.key}
                        </div>
                      </div>
                      <div>
                        <SettingControl
                          id={setting.key}
                          setting={setting}
                          value={draft[setting.key]}
                          disabled={restarting || !config.managed_restart}
                          onChange={(value) =>
                            setDraft((current) => ({
                              ...current,
                              [setting.key]: value,
                            }))
                          }
                        />
                      </div>
                    </div>
                  );
                })}
              </CardContent>
            </Card>
          </div>
        </>
      ) : null}

      <Dialog
        open={confirmOpen}
        onOpenChange={(open) => {
          if (!restarting) setConfirmOpen(open);
        }}
      >
        <DialogContent className="max-w-xl">
          <DialogHeader>
            <DialogTitle>
              {restarting ? t("正在重启 QWEN EXO") : t("应用服务配置")}
            </DialogTitle>
            <DialogDescription>
              {restarting
                ? t(
                    "请保持此页面打开。连接中断属于预期行为，控制台会自动等待新进程通过健康检查。",
                  )
                : t(
                    "将原子写入 {count} 项变更，并终止当前进程触发 Docker 托管重启。",
                    { count: formatNumber(changedKeys.length) },
                  )}
            </DialogDescription>
          </DialogHeader>
          {restarting ? (
            <div className="space-y-4 py-2">
              <div className="flex items-center gap-3 border bg-muted/40 p-3">
                <LoaderCircle className="h-5 w-5 animate-spin text-primary" />
                <div>
                  <div className="text-sm font-medium">
                    {t(RESTART_STEPS[restartStep])}
                  </div>
                  <div className="mt-1 text-xs text-muted-foreground">
                    {localizedMessageText(restartMessage)}
                  </div>
                </div>
              </div>
              <div className="grid grid-cols-4 gap-2">
                {RESTART_STEPS.map((step, index) => (
                  <div key={step}>
                    <div
                      className={cn(
                        "h-1 bg-muted",
                        index <= restartStep && "bg-primary",
                      )}
                    />
                    <div className="mt-2 hidden text-[9px] text-muted-foreground sm:block">
                      {t(step)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="space-y-3">
              <div className="max-h-60 overflow-auto rounded-md border">
                <div className="divide-y">
                  {changedKeys.map((key) => {
                    const setting = config?.settings.find(
                      (item) => item.key === key,
                    );
                    return (
                      <div
                        key={key}
                        className="grid grid-cols-[1fr_auto] gap-4 px-3 py-2 text-xs"
                      >
                        <div>
                          <div className="font-medium">
                            {setting
                              ? settingCopy(
                                  setting.key,
                                  setting.label,
                                  setting.description,
                                ).label
                              : key}
                          </div>
                          <div className="font-mono text-[10px] text-muted-foreground">
                            {key}
                          </div>
                        </div>
                        <div className="text-right font-mono">
                          <div className="text-muted-foreground line-through">
                            {String(config?.values[key])}
                          </div>
                          <div className="text-primary">
                            {String(draft[key])}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
              <div className="flex gap-2 bg-amber-50 p-3 text-xs leading-5 text-amber-900">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                {t(
                  "运行中的请求会被中断。若新进程在到达 runtime.ready 前退出，启动器将在下一次拉起时回滚到上一健康 revision。",
                )}
              </div>
            </div>
          )}
          <DialogFooter>
            {!restarting ? (
              <>
                <Button variant="outline" onClick={() => setConfirmOpen(false)}>
                  {t("取消")}
                </Button>
                <Button onClick={() => void saveAndRestart()}>
                  <ServerCog />
                  {t("确认保存并重启")}
                </Button>
              </>
            ) : (
              <div className="flex w-full items-center gap-2 text-xs text-muted-foreground">
                <CheckCircle2 className="h-4 w-4" />
                {t("配置保存后由服务端独立完成，不依赖浏览器保持网络连接。")}
              </div>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
