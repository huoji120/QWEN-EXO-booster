import { useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  BrainCircuit,
  BookOpen,
  Check,
  Copy,
  Download,
  LoaderCircle,
  Pencil,
  Plus,
  RefreshCw,
  ServerCog,
  SlidersHorizontal,
  Trash2,
  Upload,
} from "lucide-react";
import { toast } from "sonner";
import { EmptyState } from "@/components/empty-state";
import { PageHeader } from "@/components/page-header";
import { TagInput } from "@/components/tag-input";
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
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Textarea } from "@/components/ui/textarea";
import {
  ApiError,
  createTrajectory,
  deleteTrajectory,
  getEditorTraining,
  getServiceConfig,
  getTrainingSelection,
  getTrajectory,
  listTrajectories,
  previewTrajectory,
  saveTrajectory,
  trainSelectedTrajectories,
  updateServiceConfig,
  updateTrainingSelection,
} from "@/lib/api";
import { translate as t, useI18n } from "@/lib/i18n";
import type {
  EditorTrainingStatus,
  TrainingSelectionStatus,
  TrajectoryInfo,
} from "@/lib/types";
import { formatNumber } from "@/lib/utils";

const RESTART_PENDING_KEY = "qwen-exo-trajectory-restart-pending";
function trajectoryTemplate() {
  return JSON.stringify(
    {
      format: "chatml-v1",
      session: {
        messages: [
          {
            role: "system",
            content: t("你是一个严谨的软件工程助手。"),
          },
          {
            role: "user",
            content: t("请检查项目中的问题，完成修改并验证结果。"),
          },
          {
            role: "assistant",
            content: t("我会先读取相关文件，确认现有约束后再修改。"),
          },
          {
            role: "tool",
            content: JSON.stringify({
              tool: "read",
              result: t("这里放工具返回的原始文本"),
            }),
          },
          {
            role: "assistant",
            content: t("问题已定位并修复；验证命令通过。"),
          },
        ],
      },
    },
    null,
    2,
  );
}

function downloadTrajectoryTemplate() {
  const blob = new Blob([`${trajectoryTemplate()}\n`], {
    type: "application/json;charset=utf-8",
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "qwen-exo-trajectory-template.json";
  anchor.click();
  URL.revokeObjectURL(url);
}

function describeError(error: unknown, fallback = t("未知错误")) {
  if (error instanceof ApiError && error.status === 404) {
    return t("该功能将在服务重启后可用");
  }
  return error instanceof Error ? error.message : fallback;
}

function readFileAsBase64(file: File) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () =>
      reject(new Error(t("无法读取 {name}", { name: file.name })));
    reader.onload = () => {
      const result = reader.result;
      if (typeof result !== "string" || !result.includes(",")) {
        reject(new Error(t("无法编码 {name}", { name: file.name })));
        return;
      }
      resolve(result.slice(result.indexOf(",") + 1));
    };
    reader.readAsDataURL(file);
  });
}

function delay(milliseconds: number) {
  return new Promise<void>((resolve) =>
    window.setTimeout(resolve, milliseconds),
  );
}

export function EditorPage() {
  const { locale } = useI18n();
  const [selection, setSelection] = useState<TrainingSelectionStatus>({
    names: [],
    updated_at: null,
    sources: [],
    editor: null,
    up_to_date: false,
    applied: false,
  });
  const [selectionLoading, setSelectionLoading] = useState(true);
  const [selectionUpdating, setSelectionUpdating] = useState<string | null>(
    null,
  );

  const [trajectories, setTrajectories] = useState<TrajectoryInfo[]>([]);
  const [trajectoriesLoading, setTrajectoriesLoading] = useState(true);
  const [editorOpen, setEditorOpen] = useState(false);
  const [editorName, setEditorName] = useState("");
  const [editorContent, setEditorContent] = useState("");
  const [editorTags, setEditorTags] = useState<string[]>([]);
  const [editorMode, setEditorMode] = useState<"create" | "edit">("edit");
  const [suggestedName, setSuggestedName] = useState("");
  const [editorLoading, setEditorLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<TrajectoryInfo | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [originalName, setOriginalName] = useState("");
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [training, setTraining] = useState<EditorTrainingStatus>({
    status: "idle",
    job: null,
  });
  const [trainingOpen, setTrainingOpen] = useState(false);
  const [trainingSubmitting, setTrainingSubmitting] = useState(false);
  const [trainingMessage, setTrainingMessage] = useState("");
  const [formatOpen, setFormatOpen] = useState(false);
  const [restartOpen, setRestartOpen] = useState(false);
  const [restarting, setRestarting] = useState(false);
  const [restartPending, setRestartPending] = useState(
    () => window.localStorage.getItem(RESTART_PENDING_KEY) === "1",
  );

  const selectedNameSet = useMemo(
    () => new Set(selection.names),
    [selection.names],
  );
  const nameCollator = useMemo(
    () => new Intl.Collator(locale, { numeric: true, sensitivity: "base" }),
    [locale],
  );
  const assets = useMemo(
    () =>
      [...trajectories].sort((left, right) => {
        const selectedDelta =
          Number(selectedNameSet.has(right.name)) -
          Number(selectedNameSet.has(left.name));
        return selectedDelta || nameCollator.compare(left.name, right.name);
      }),
    [nameCollator, selectedNameSet, trajectories],
  );

  const loadTrajectories = async () => {
    setTrajectoriesLoading(true);
    try {
      const payload = await listTrajectories();
      setTrajectories(payload.trajectories || []);
    } catch (error) {
      toast.error(t("轨迹列表加载失败"), {
        description: describeError(error),
      });
    } finally {
      setTrajectoriesLoading(false);
    }
  };
  const loadSelection = async () => {
    setSelectionLoading(true);
    try {
      setSelection(await getTrainingSelection());
    } catch (error) {
      toast.error(t("训练成员状态加载失败"), {
        description: describeError(error),
      });
    } finally {
      setSelectionLoading(false);
    }
  };

  const loadTraining = async () => {
    try {
      const next = await getEditorTraining();
      setTraining(next);
      if (next.status === "succeeded") {
        window.localStorage.removeItem(RESTART_PENDING_KEY);
        setRestartPending(false);
      }
    } catch (error) {
      toast.error(t("训练任务状态加载失败"), {
        description: describeError(error),
      });
    }
  };

  useEffect(() => {
    void loadTrajectories();
    void loadSelection();
    void loadTraining();
  }, []);

  const markRestartPending = () => {
    window.localStorage.setItem(RESTART_PENDING_KEY, "1");
    setRestartPending(true);
  };

  const restartServer = async () => {
    setRestarting(true);
    try {
      const config = await getServiceConfig();
      if (!config.managed_restart) {
        throw new Error(
          t("当前服务未由 Docker 托管启动器管理，不能从控制台重启"),
        );
      }
      await updateServiceConfig(config.values, config.revision);
      window.localStorage.removeItem(RESTART_PENDING_KEY);
      setRestartPending(false);
      setRestartOpen(false);
      toast.success(t("服务端重启请求已提交"), {
        description: t("当前连接将短暂中断；服务会由托管启动器自动恢复。"),
        duration: 10000,
      });
    } catch (error) {
      toast.error(t("服务端重启失败"), {
        description: describeError(error),
        duration: 10000,
      });
    } finally {
      setRestarting(false);
    }
  };

  const toggleTrainingMember = async (name: string) => {
    const wasSelected = selectedNameSet.has(name);
    const nextNames = wasSelected
      ? selection.names.filter((item) => item !== name)
      : [...selection.names, name];
    setSelectionUpdating(name);
    try {
      const next = await updateTrainingSelection(nextNames);
      setSelection(next);
      toast.success(
        wasSelected ? t("已停用该轨迹训练") : t("已激活该轨迹训练"),
        {
          description: wasSelected
            ? t("{name} 不再进入下一次联合训练", { name })
            : t("{name} 已加入联合训练集；各轨迹会话边界保持独立", { name }),
        },
      );
    } catch (error) {
      toast.error(wasSelected ? t("停用训练失败") : t("激活训练失败"), {
        description: describeError(error),
      });
    } finally {
      setSelectionUpdating(null);
    }
  };

  const startTraining = async () => {
    if (!selection.names.length) return;
    const sourceCount = selection.names.length;
    setTrainingSubmitting(true);
    setTrainingMessage("正在持久化联合训练任务并释放 GPU");
    try {
      const accepted = await trainSelectedTrajectories();
      setTraining(accepted);
      setTrainingMessage("推理服务即将离线；正在训练一个联合低秩编辑器");
      await delay(2500);
      let lastError = "等待训练服务恢复";
      for (let attempt = 0; attempt < 900; attempt += 1) {
        let next: EditorTrainingStatus;
        try {
          next = await getEditorTraining();
        } catch (error) {
          lastError =
            error instanceof Error ? error.message : t("服务正在训练");
          setTrainingMessage("服务离线联合训练中；完成后会自动恢复推理服务");
          await delay(2000);
          continue;
        }
        setTraining(next);
        if (next.status === "failed") {
          throw new Error(next.job?.error || t("联合轨迹编辑器训练失败"));
        }
        if (next.status === "succeeded") {
          setTrainingMessage("联合训练完成；单个编辑器已发布并应用");
          window.localStorage.removeItem(RESTART_PENDING_KEY);
          setRestartPending(false);
          await Promise.all([loadTrajectories(), loadSelection()]);
          toast.success(t("联合轨迹编辑器训练完成"), {
            description: t(
              "{count} 条轨迹已训练为一个编辑器；作用强度由设置页当前档位决定",
              { count: formatNumber(sourceCount) },
            ),
            duration: 10000,
          });
          await delay(700);
          setTrainingOpen(false);
          setTrainingSubmitting(false);
          return;
        }
        setTrainingMessage(
          next.status === "running"
            ? "模型已加载，正在联合训练并计算质量报告"
            : "训练任务已排队，等待托管进程释放 GPU",
        );
        lastError = t("训练状态仍为 {status}", { status: next.status });
        await delay(2000);
      }
      throw new Error(t("等待训练完成超时：{error}", { error: lastError }));
    } catch (error) {
      const message = error instanceof Error ? error.message : t("未知错误");
      setTrainingMessage(message);
      toast.error(t("联合轨迹编辑器训练失败"), {
        description: message,
        duration: 12000,
      });
      setTrainingSubmitting(false);
    }
  };

  const openTrajectory = async (trajectory: TrajectoryInfo) => {
    setEditorOpen(true);
    setEditorMode("edit");
    setSuggestedName("");
    setEditorLoading(true);
    setOriginalName(trajectory.name);
    setEditorName(trajectory.name);
    setEditorTags(trajectory.tags || []);
    setEditorContent("");
    try {
      const payload = await getTrajectory(trajectory.name);
      setEditorContent(payload.content);
      setEditorTags(payload.tags || []);
    } catch (error) {
      toast.error(t("轨迹读取失败"), {
        description: describeError(error),
      });
      setEditorOpen(false);
    } finally {
      setEditorLoading(false);
    }
  };

  const save = async () => {
    const name = editorName.trim().toLowerCase();
    if (!name) {
      toast.error(t("请填写轨迹名称，不能直接使用上传文件名入库"));
      return;
    }
    if (!editorTags.length) {
      toast.error(t("至少填写一个轨迹标签"));
      return;
    }
    if (!editorContent.trim()) {
      toast.error(t("轨迹内容不能为空"));
      return;
    }
    setSaving(true);
    try {
      const result =
        editorMode === "create"
          ? await createTrajectory(name, editorContent, editorTags)
          : await saveTrajectory(
              originalName || name,
              editorContent,
              editorTags,
              name,
            );
      await Promise.all([loadTrajectories(), loadSelection()]);
      setEditorOpen(false);
      markRestartPending();
      toast.success(
        editorMode === "create"
          ? t("轨迹已创建，重启后生效")
          : t("轨迹已保存，重启后生效"),
        {
          description: t("{name} · {count} 条消息 · {tags}", {
            name: result.name,
            count: formatNumber(result.messages),
            tags: result.tags.join(t("、")),
          }),
          duration: 8000,
        },
      );
    } catch (error) {
      toast.error(editorMode === "create" ? t("创建失败") : t("保存失败"), {
        description: describeError(error),
        duration: 8000,
      });
    } finally {
      setSaving(false);
    }
  };

  const remove = async () => {
    if (!deleteTarget) return;
    setDeleting(true);
    try {
      await deleteTrajectory(deleteTarget.name);
      setDeleteTarget(null);
      await Promise.all([loadTrajectories(), loadSelection()]);
      toast.success(t("轨迹已删除"));
    } catch (error) {
      toast.error(t("删除失败"), {
        description: describeError(error),
      });
    } finally {
      setDeleting(false);
    }
  };

  const importFile = async (file: File) => {
    setUploading(true);
    try {
      const contentBase64 = await readFileAsBase64(file);
      const draft = await previewTrajectory(file.name, contentBase64);
      setEditorMode("create");
      setEditorName("");
      setSuggestedName(draft.suggested_name);
      setEditorTags(draft.tags || []);
      setEditorContent(draft.content);
      setEditorLoading(false);
      setOriginalName("");
      setEditorOpen(true);
      toast.success(t("文件已解析为草稿，尚未保存"), {
        description: t("请填写自定义名称和标签，并检查内容后再创建。"),
      });
    } catch (error) {
      toast.error(t("轨迹解析失败，未写入服务端"), {
        description: describeError(error),
        duration: 10000,
      });
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="page-frame">
      <PageHeader
        title={t("轨迹微调")}
        actions={
          <>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setFormatOpen(true)}
              title={t("轨迹格式")}
            >
              <BookOpen />
              <span className="sr-only">{t("轨迹格式")}</span>
            </Button>
            <Button
              variant="ghost"
              size="icon"
              disabled={trajectoriesLoading || selectionLoading}
              onClick={() => {
                void loadTrajectories();
                void loadSelection();
              }}
              title={t("刷新")}
            >
              {trajectoriesLoading || selectionLoading ? (
                <LoaderCircle className="animate-spin" />
              ) : (
                <RefreshCw />
              )}
              <span className="sr-only">{t("刷新")}</span>
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={uploading}
              onClick={() => fileInputRef.current?.click()}
            >
              {uploading ? (
                <LoaderCircle className="animate-spin" />
              ) : (
                <Upload />
              )}
              {t("导入")}
            </Button>
            <Button
              size="sm"
              disabled={
                !selection.names.length ||
                trainingSubmitting ||
                training.status === "queued" ||
                training.status === "running"
              }
              onClick={() => setTrainingOpen(true)}
            >
              {training.status === "queued" || training.status === "running" ? (
                <LoaderCircle className="animate-spin" />
              ) : (
                <BrainCircuit />
              )}
              {selection.up_to_date ? t("重新训练") : t("训练")}
              {selection.names.length
                ? ` (${formatNumber(selection.names.length)})`
                : ""}
            </Button>
          </>
        }
      />

      <input
        ref={fileInputRef}
        type="file"
        accept=".json,.jsonl,.gz,.zip,application/json,application/gzip,application/zip"
        hidden
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) void importFile(file);
          event.target.value = "";
        }}
      />

      {restartPending ? (
        <div className="mb-4 flex items-center justify-between gap-3 rounded-lg border bg-muted/40 px-4 py-3">
          <span className="text-sm font-medium">
            {t("修改已保存，重启后生效")}
          </span>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setRestartOpen(true)}
          >
            <ServerCog />
            {t("重启")}
          </Button>
        </div>
      ) : null}

      <Card>
        <CardContent className="p-0">
          {assets.length ? (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>{t("轨迹")}</TableHead>
                  <TableHead className="w-40">{t("消息")}</TableHead>
                  <TableHead className="w-72 text-right">{t("操作")}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {assets.map((trajectory) => {
                  const isSelected = selectedNameSet.has(trajectory.name);
                  const isTraining = Boolean(
                    training.job?.trajectories?.includes(trajectory.name) &&
                      (training.status === "queued" ||
                        training.status === "running"),
                  );
                  return (
                    <TableRow key={trajectory.name}>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-xs font-medium">
                            {trajectory.name}
                          </span>
                          {!trajectory.valid ? (
                            <Badge variant="destructive">{t("无效")}</Badge>
                          ) : null}
                        </div>
                      </TableCell>
                      <TableCell className="font-mono text-xs">
                        {formatNumber(trajectory.messages)}
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex items-center justify-end gap-1">
                          <Button
                            variant={isSelected ? "secondary" : "outline"}
                            size="sm"
                            disabled={
                              !trajectory.valid ||
                              selectionUpdating !== null ||
                              trainingSubmitting ||
                              training.status === "queued" ||
                              training.status === "running"
                            }
                            onClick={() =>
                              void toggleTrainingMember(trajectory.name)
                            }
                          >
                            {selectionUpdating === trajectory.name ||
                            isTraining ? (
                              <LoaderCircle className="animate-spin" />
                            ) : isSelected ? (
                              <Check />
                            ) : (
                              <Plus />
                            )}
                            {isSelected ? t("已选择") : t("选择")}
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => void openTrajectory(trajectory)}
                          >
                            <Pencil />
                            {t("编辑")}
                          </Button>
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => setDeleteTarget(trajectory)}
                          >
                            <Trash2 />
                            <span className="sr-only">{t("删除轨迹")}</span>
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          ) : (
            <EmptyState
              icon={
                trajectoriesLoading || selectionLoading
                  ? LoaderCircle
                  : SlidersHorizontal
              }
              title={
                trajectoriesLoading || selectionLoading
                  ? t("正在读取")
                  : t("暂无轨迹")
              }
              description={t("导入一份轨迹，或新建轨迹后再开始训练。")}
              actionLabel={
                !trajectoriesLoading && !selectionLoading
                  ? t("导入")
                  : undefined
              }
              onAction={
                !trajectoriesLoading && !selectionLoading
                  ? () => fileInputRef.current?.click()
                  : undefined
              }
            />
          )}
        </CardContent>
      </Card>
      <Dialog open={formatOpen} onOpenChange={setFormatOpen}>
        <DialogContent className="max-h-[90vh] max-w-5xl overflow-y-auto">
          <DialogHeader>
            <DialogTitle>{t("阅读轨迹格式")}</DialogTitle>
            <DialogDescription>
              {t(
                "服务端按 ChatML v1 校验并规范化轨迹。下面的模板可直接下载、修改和上传。",
              )}
            </DialogDescription>
          </DialogHeader>

          <div className="grid gap-5 lg:grid-cols-[290px_minmax(0,1fr)]">
            <div className="space-y-3">
              <div className="rounded-lg border p-4">
                <div className="text-sm font-semibold">{t("必须满足")}</div>
                <ol className="mt-3 space-y-3 text-xs leading-5 text-muted-foreground">
                  <li className="flex gap-2">
                    <Badge variant="secondary" className="h-5 shrink-0">
                      1
                    </Badge>
                    <span>
                      <code>messages</code>{" "}
                      {t("必须是数组，至少包含 2 条消息。")}
                    </span>
                  </li>
                  <li className="flex gap-2">
                    <Badge variant="secondary" className="h-5 shrink-0">
                      2
                    </Badge>
                    <span>
                      {t("至少包含 1 条")} <code>assistant</code> {t("消息。")}
                    </span>
                  </li>
                  <li className="flex gap-2">
                    <Badge variant="secondary" className="h-5 shrink-0">
                      3
                    </Badge>
                    <span>
                      {t("每条消息使用")} <code>role</code> {t("和")}{" "}
                      <code>content</code> {t("字段。")}
                    </span>
                  </li>
                </ol>
              </div>

              <div className="rounded-lg border p-4">
                <div className="text-sm font-semibold">{t("允许的角色")}</div>
                <div className="mt-3 flex flex-wrap gap-1.5">
                  {["system", "user", "assistant", "tool"].map((role) => (
                    <Badge key={role} variant="outline" className="font-mono">
                      {role}
                    </Badge>
                  ))}
                </div>
                <p className="mt-3 text-xs leading-5 text-muted-foreground">
                  <code>content</code>{" "}
                  {t(
                    "建议使用字符串；若上传 JSON 对象，服务端会将其规范化为 JSON 字符串。",
                  )}
                </p>
              </div>

              <div className="rounded-lg border p-4 text-xs leading-5 text-muted-foreground">
                <div className="text-sm font-semibold text-foreground">
                  {t("文件规则")}
                </div>
                <ul className="mt-2 list-disc space-y-1 pl-4">
                  <li>{t("支持 .json、.jsonl、.gz、.zip。")}</li>
                  <li>{t("ZIP 内只能有 1 个轨迹文件。")}</li>
                  <li>{t("解压后必须是 UTF-8，最大 8 MB。")}</li>
                  <li>{t("轨迹名只使用小写字母、数字、点、横线和下划线。")}</li>
                </ul>
              </div>
            </div>

            <div className="min-w-0 space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <div className="text-sm font-semibold">
                    {t("ChatML JSON 模板")}
                  </div>
                  <p className="mt-1 text-[11px] text-muted-foreground">
                    {t("推荐使用")} <code>session.messages</code>{" "}
                    {t("容器；它与服务端最终保存格式一致。")}
                  </p>
                </div>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      void navigator.clipboard.writeText(trajectoryTemplate());
                      toast.success(t("轨迹模板已复制"));
                    }}
                  >
                    <Copy />
                    {t("复制")}
                  </Button>
                  <Button size="sm" onClick={downloadTrajectoryTemplate}>
                    <Download />
                    {t("下载 JSON 模板")}
                  </Button>
                </div>
              </div>

              <pre className="max-h-[52vh] overflow-auto rounded-lg border bg-slate-950 p-4 font-mono text-[11px] leading-5 text-slate-200">
                {trajectoryTemplate()}
              </pre>

              <div className="grid gap-3 sm:grid-cols-2">
                <div className="rounded-lg border bg-muted/30 p-3">
                  <div className="text-xs font-semibold">
                    {t("JSON 也接受")}
                  </div>
                  <p className="mt-1 font-mono text-[10px] leading-5 text-muted-foreground">
                    {t("顶层 messages 数组，或直接上传消息对象数组。")}
                  </p>
                </div>
                <div className="rounded-lg border bg-muted/30 p-3">
                  <div className="text-xs font-semibold">{t("JSONL 规则")}</div>
                  <p className="mt-1 font-mono text-[10px] leading-5 text-muted-foreground">
                    {t(
                      "每行一条 role/content 消息；也可每行放一个 messages 数组。",
                    )}
                  </p>
                </div>
              </div>
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setFormatOpen(false)}>
              {t("关闭")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={restartOpen}
        onOpenChange={(open) => {
          if (!restarting) setRestartOpen(open);
        }}
      >
        <DialogContent className="max-w-xl">
          <DialogHeader>
            <DialogTitle>
              {restarting ? t("正在提交重启请求") : t("重启 QWEN EXO 服务端？")}
            </DialogTitle>
            <DialogDescription>
              {t(
                "重启后服务端会重新加载轨迹文件与运行配置。请只在当前推理结束后执行。",
              )}
            </DialogDescription>
          </DialogHeader>

          <div className="flex gap-3 rounded-lg border border-red-300 bg-red-50 p-4 text-red-950 dark:border-red-900 dark:bg-red-950/45 dark:text-red-100">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
            <div>
              <div className="text-sm font-semibold">
                {t("运行中的推理会被立即中断")}
              </div>
              <p className="mt-1 text-xs leading-5 opacity-80">
                {t(
                  "确认后，Docker 托管启动器会结束当前进程并自动拉起新进程。模型重新就绪前控制台会短暂断开。",
                )}
              </p>
            </div>
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              disabled={restarting}
              onClick={() => setRestartOpen(false)}
            >
              {t("取消")}
            </Button>
            <Button
              variant="destructive"
              disabled={restarting}
              onClick={() => void restartServer()}
            >
              {restarting ? (
                <LoaderCircle className="animate-spin" />
              ) : (
                <ServerCog />
              )}
              {restarting ? t("正在提交…") : t("确认重启服务端")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={trainingOpen}
        onOpenChange={(open) => {
          if (!trainingSubmitting) setTrainingOpen(open);
        }}
      >
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>
              {trainingSubmitting
                ? t("正在训练")
                : t("训练 {count} 条轨迹？", {
                    count: formatNumber(selection.names.length),
                  })}
            </DialogTitle>
            {!trainingSubmitting ? (
              <DialogDescription>
                {t("训练期间服务会暂时停止，完成后自动恢复。")}
              </DialogDescription>
            ) : null}
          </DialogHeader>

          {trainingSubmitting ? (
            <div className="flex items-center gap-3 rounded-lg border bg-muted/40 p-4">
              <LoaderCircle className="h-5 w-5 shrink-0 animate-spin" />
              <span className="text-sm font-medium">{t(trainingMessage)}</span>
            </div>
          ) : null}

          {!trainingSubmitting ? (
            <DialogFooter>
              <Button variant="outline" onClick={() => setTrainingOpen(false)}>
                {t("取消")}
              </Button>
              <Button onClick={() => void startTraining()}>
                <BrainCircuit />
                {t("开始训练")}
              </Button>
            </DialogFooter>
          ) : null}
        </DialogContent>
      </Dialog>

      <Dialog
        open={editorOpen}
        onOpenChange={(open) => {
          if (!saving) setEditorOpen(open);
        }}
      >
        <DialogContent className="max-h-[92vh] max-w-5xl overflow-y-auto">
          <DialogHeader>
            <DialogTitle>
              {editorMode === "create"
                ? t("检查并创建轨迹")
                : t("编辑轨迹资产")}
            </DialogTitle>
            <DialogDescription>
              {editorMode === "create"
                ? t(
                    "上传文件只解析为草稿；必须填写自定义名称和标签，确认后才会写入服务端。",
                  )
                : t(
                    "可修改轨迹名称、ChatML 内容和标签；训练成员引用会同步，已有联合产物将等待重训。",
                  )}
            </DialogDescription>
          </DialogHeader>

          <div className="grid gap-4 sm:grid-cols-2">
            <div className="grid gap-2">
              <Label htmlFor="trajectory-name">{t("轨迹名称")}</Label>
              <Input
                id="trajectory-name"
                value={editorName}
                onChange={(event) => setEditorName(event.target.value)}
                placeholder={
                  suggestedName
                    ? t("例如 {name}", { name: suggestedName })
                    : t("例如 coding-agent-success-run")
                }
                disabled={editorLoading || saving}
              />
              <p className="text-[10px] leading-4 text-muted-foreground">
                {t(
                  "只允许小写字母、数字、点、横线和下划线；已激活训练的轨迹改名后会同步成员引用。",
                )}
              </p>
            </div>
            <div className="grid gap-2">
              <Label>{t("轨迹标签（至少一个）")}</Label>
              <TagInput
                value={editorTags}
                onChange={setEditorTags}
                disabled={editorLoading || saving}
                placeholder={t("例如 修复成功、工具调用、验证通过")}
              />
            </div>
          </div>

          <div className="grid gap-2">
            <div className="flex items-center justify-between">
              <Label htmlFor="trajectory-content">
                {t("ChatML JSON 内容")}
              </Label>
              <span className="text-[11px] text-muted-foreground">
                {t("{count} 字符", {
                  count: formatNumber(editorContent.length),
                })}
              </span>
            </div>
            {editorLoading ? (
              <div className="grid h-96 place-items-center border">
                <LoaderCircle className="animate-spin text-muted-foreground" />
              </div>
            ) : (
              <Textarea
                id="trajectory-content"
                value={editorContent}
                onChange={(event) => setEditorContent(event.target.value)}
                className="source-editor min-h-[48vh] resize-y"
                spellCheck={false}
                disabled={saving}
              />
            )}
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setEditorOpen(false)}
              disabled={saving}
            >
              {t("取消")}
            </Button>
            <Button
              onClick={() => void save()}
              disabled={saving || editorLoading}
            >
              {saving ? <LoaderCircle className="animate-spin" /> : null}
              {editorMode === "create" ? t("创建轨迹") : t("保存内容与标签")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={Boolean(deleteTarget)}
        onOpenChange={(open) => {
          if (!open && !deleting) setDeleteTarget(null);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t("删除轨迹")}</DialogTitle>
            <DialogDescription>
              {t("将删除")}{" "}
              <span className="font-mono text-foreground">
                {deleteTarget?.name}
              </span>
              {t("。此操作无法撤销。")}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setDeleteTarget(null)}
              disabled={deleting}
            >
              {t("取消")}
            </Button>
            <Button
              variant="destructive"
              onClick={() => void remove()}
              disabled={deleting}
            >
              {deleting ? <LoaderCircle className="animate-spin" /> : null}
              {t("删除")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
