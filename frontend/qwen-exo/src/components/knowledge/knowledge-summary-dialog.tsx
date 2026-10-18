import { useEffect, useRef, useState } from "react";
import { LoaderCircle, RotateCcw, Save, Sparkles, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Skeleton } from "@/components/ui/skeleton";
import { Textarea } from "@/components/ui/textarea";
import {
  deleteKnowledgeSummary,
  generateKnowledgeSummaries,
  getKnowledgeSummary,
  saveKnowledgeSummary,
} from "@/lib/api";
import { translate as t } from "@/lib/i18n";
import type { KnowledgeSummaryRecord } from "@/lib/types";
import { formatTime } from "@/lib/utils";
import { errorText, type SummaryView } from "./knowledge-utils";

type Busy = "save" | "generate" | "delete" | null;

export function KnowledgeSummaryDialog({
  path,
  view,
  onClose,
  onChanged,
}: {
  path: string | null;
  /** Live status of this file (running / queued / failed …). */
  view: SummaryView | null;
  onClose: () => void;
  onChanged: () => void;
}) {
  const [record, setRecord] = useState<KnowledgeSummaryRecord | null>(null);
  const [draft, setDraft] = useState("");
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState<Busy>(null);

  const reload = async (target: string) => {
    setLoading(true);
    try {
      const next = await getKnowledgeSummary(target);
      setRecord(next);
      setDraft(next?.summary ?? "");
    } catch (error) {
      toast.error(t("总结读取失败"), { description: errorText(error, t("未知错误")) });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    setRecord(null);
    setDraft("");
    if (path) void reload(path);
  }, [path]);

  // A background run for this file just finished: show the new text.
  const running = view?.state === "running" || view?.state === "queued";
  const wasRunning = useRef(false);
  useEffect(() => {
    if (running) wasRunning.current = true;
    else if (wasRunning.current) {
      wasRunning.current = false;
      if (path) void reload(path);
    }
  }, [running]);

  const act = async (kind: Exclude<Busy, null>) => {
    if (!path || busy) return;
    setBusy(kind);
    try {
      if (kind === "save") {
        const saved = await saveKnowledgeSummary(path, draft);
        setRecord(saved);
        setDraft(saved.summary);
        toast.success(t("总结已保存"));
      } else if (kind === "generate") {
        await generateKnowledgeSummaries([path], true);
        toast.success(t("已加入总结队列"));
      } else {
        await deleteKnowledgeSummary(path);
        setRecord(null);
        setDraft("");
        toast.success(t("总结已删除"));
      }
      onChanged();
    } catch (error) {
      toast.error(
        kind === "save"
          ? t("保存失败")
          : kind === "generate"
            ? t("生成总结失败")
            : t("删除失败"),
        { description: errorText(error, t("未知错误")) },
      );
    } finally {
      setBusy(null);
    }
  };

  const dirty = draft.trim() !== (record?.summary ?? "").trim();

  return (
    <Dialog open={Boolean(path)} onOpenChange={(open) => !open && !busy && onClose()}>
      <DialogContent className="max-w-3xl">
        <DialogHeader>
          <DialogTitle className="break-all font-mono text-sm">{path}</DialogTitle>
          <DialogDescription asChild>
            <div className="flex flex-wrap items-center gap-2">
              {record ? (
                <>
                  <span className="text-foreground">{record.title}</span>
                  <Badge variant={record.origin === "manual" ? "success" : "secondary"}>
                    {record.origin === "manual" ? t("手动") : t("自动生成")}
                  </Badge>
                  {record.sampled ? <Badge variant="outline">{t("抽样")}</Badge> : null}
                  <span className="text-xs">
                    {formatTime(record.updated_ns / 1_000_000_000)}
                  </span>
                </>
              ) : loading ? null : (
                <span>{t("这个文件还没有总结；可以手动填写，或让模型生成。")}</span>
              )}
              {view && (view.state === "running" || view.state === "queued") ? (
                <Badge variant={view.variant}>
                  {view.state === "running" ? (
                    <LoaderCircle className="h-3 w-3 animate-spin" />
                  ) : null}
                  {t(view.label)}
                  {view.progress ? ` ${view.progress}` : ""}
                </Badge>
              ) : null}
              {view?.state === "failed" ? (
                <Badge variant="destructive" title={view.detail}>
                  {t("上次生成失败")}
                </Badge>
              ) : null}
            </div>
          </DialogDescription>
        </DialogHeader>
        {view?.state === "failed" && view.detail ? (
          <p className="break-words rounded-md border border-destructive/40 bg-destructive/5 p-2 font-mono text-xs text-destructive">
            {view.detail}
          </p>
        ) : null}
        {loading ? (
          <Skeleton className="h-[40vh] w-full" />
        ) : (
          <Textarea
            className="h-[45vh] resize-y font-mono text-xs leading-5"
            value={draft}
            placeholder={t("在这里输入总结；保存后标记为手动，自动生成不会覆盖。")}
            onChange={(event) => setDraft(event.target.value)}
          />
        )}
        <DialogFooter className="sm:justify-between">
          <Button
            variant="outline"
            className="text-destructive hover:text-destructive"
            disabled={!record || Boolean(busy)}
            onClick={() => void act("delete")}
          >
            {busy === "delete" ? <LoaderCircle className="animate-spin" /> : <Trash2 />}
            {t("删除总结")}
          </Button>
          <div className="flex flex-col-reverse gap-2 sm:flex-row">
            <Button
              variant="outline"
              disabled={Boolean(busy) || running}
              onClick={() => void act("generate")}
            >
              {busy === "generate" ? (
                <LoaderCircle className="animate-spin" />
              ) : record ? (
                <RotateCcw />
              ) : (
                <Sparkles />
              )}
              {record ? t("重新生成") : t("生成总结")}
            </Button>
            <Button
              disabled={!draft.trim() || !dirty || Boolean(busy) || loading}
              onClick={() => void act("save")}
            >
              {busy === "save" ? <LoaderCircle className="animate-spin" /> : <Save />}
              {t("保存")}
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
