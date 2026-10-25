import { useCallback, useState } from "react";
import { LoaderCircle } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useKnowledgeApi } from "./knowledge-scope";
import { translate as t } from "@/lib/i18n";
import { formatNumber } from "@/lib/utils";
import { errorText } from "./knowledge-utils";

const VIEWER_WINDOW = 262_144;

type ViewerState = {
  path: string;
  text: string;
  endOffset: number;
  totalBytes: number;
  eof: boolean;
};

/** Paged raw-text viewer backed by ``/knowledge/read``. */
export function useKnowledgeViewer(onChanged: () => void) {
  const { readKnowledge, uploadKnowledgeFileRaw, signal } = useKnowledgeApi();
  const [viewer, setViewer] = useState<ViewerState | null>(null);
  const [loading, setLoading] = useState(false);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const [saving, setSaving] = useState(false);

  const open = useCallback(async (path: string) => {
    setLoading(true);
    setEditing(false);
    try {
      const result = await readKnowledge(path, 0, VIEWER_WINDOW);
      if (signal.aborted) return;
      setViewer({
        path: result.path,
        text: result.text,
        endOffset: result.end_offset,
        totalBytes: result.total_bytes,
        eof: result.eof,
      });
    } catch (error) {
      if (signal.aborted) return;
      toast.error(t("文件读取失败"), { description: errorText(error, t("未知错误")) });
    } finally {
      setLoading(false);
    }
  }, [readKnowledge, signal]);

  const loadMore = async () => {
    if (!viewer || viewer.eof || loading) return;
    setLoading(true);
    try {
      const result = await readKnowledge(viewer.path, viewer.endOffset, VIEWER_WINDOW);
      if (signal.aborted) return;
      setViewer({
        ...viewer,
        text: viewer.text + result.text,
        endOffset: result.end_offset,
        eof: result.eof,
      });
    } catch (error) {
      if (signal.aborted) return;
      toast.error(t("文件读取失败"), { description: errorText(error, t("未知错误")) });
    } finally {
      setLoading(false);
    }
  };

  const save = async () => {
    if (!viewer?.eof || saving || signal.aborted) return;
    setSaving(true);
    try {
      const content = new Blob([draft], { type: "text/plain;charset=utf-8" });
      await uploadKnowledgeFileRaw(viewer.path, content);
      if (signal.aborted) return;
      setViewer({ ...viewer, text: draft, endOffset: content.size, totalBytes: content.size });
      setEditing(false);
      toast.success(t("文件已保存"));
      onChanged();
    } catch (error) {
      if (!signal.aborted) toast.error(t("保存失败"), { description: errorText(error, t("未知错误")) });
    } finally {
      setSaving(false);
    }
  };

  const dialog = (
    <Dialog open={Boolean(viewer)} onOpenChange={(next) => !next && !saving && setViewer(null)}>
      <DialogContent className="max-w-3xl">
        <DialogHeader>
          <DialogTitle className="break-all font-mono text-sm">{viewer?.path}</DialogTitle>
          <DialogDescription>
            {viewer
              ? viewer.eof
                ? t("完整内容，共 {total} 字节", {
                    total: formatNumber(viewer.totalBytes),
                  })
                : t("已显示前 {end} 字节，共 {total} 字节", {
                    end: formatNumber(viewer.endOffset),
                    total: formatNumber(viewer.totalBytes),
                  })
              : null}
          </DialogDescription>
        </DialogHeader>
        {editing ? <Textarea className="h-[55vh] font-mono text-xs" value={draft} disabled={saving} onChange={(event) => setDraft(event.target.value)} /> : <div className="max-h-[60vh] overflow-auto rounded-md border bg-muted/40 p-3">
          <pre className="whitespace-pre-wrap break-words font-mono text-xs leading-5">
            {viewer?.text}
          </pre>
        </div>}
        {viewer && !viewer.eof ? (
          <DialogFooter>
            <Button variant="outline" disabled={loading} onClick={() => void loadMore()}>
              {loading ? <LoaderCircle className="animate-spin" /> : null}
              {t("继续加载")}
            </Button>
          </DialogFooter>
        ) : null}
        {viewer?.eof ? <DialogFooter>
          {editing ? <>
            <Button variant="outline" disabled={saving} onClick={() => setEditing(false)}>{t("取消")}</Button>
            <Button disabled={saving || draft === viewer.text} onClick={() => void save()}>
              {saving ? <LoaderCircle className="animate-spin" /> : null}{t("保存")}
            </Button>
          </> : <Button variant="outline" onClick={() => { setDraft(viewer.text); setEditing(true); }}>{t("编辑原文")}</Button>}
        </DialogFooter> : null}
      </DialogContent>
    </Dialog>
  );

  return { open, loading, dialog };
}
