import { useCallback, useState } from "react";
import { LoaderCircle } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { readKnowledge } from "@/lib/api";
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
export function useKnowledgeViewer() {
  const [viewer, setViewer] = useState<ViewerState | null>(null);
  const [loading, setLoading] = useState(false);

  const open = useCallback(async (path: string) => {
    setLoading(true);
    try {
      const result = await readKnowledge(path, 0, VIEWER_WINDOW);
      setViewer({
        path: result.path,
        text: result.text,
        endOffset: result.end_offset,
        totalBytes: result.total_bytes,
        eof: result.eof,
      });
    } catch (error) {
      toast.error(t("文件读取失败"), { description: errorText(error, t("未知错误")) });
    } finally {
      setLoading(false);
    }
  }, []);

  const loadMore = async () => {
    if (!viewer || viewer.eof || loading) return;
    setLoading(true);
    try {
      const result = await readKnowledge(viewer.path, viewer.endOffset, VIEWER_WINDOW);
      setViewer({
        ...viewer,
        text: viewer.text + result.text,
        endOffset: result.end_offset,
        eof: result.eof,
      });
    } catch (error) {
      toast.error(t("文件读取失败"), { description: errorText(error, t("未知错误")) });
    } finally {
      setLoading(false);
    }
  };

  const dialog = (
    <Dialog open={Boolean(viewer)} onOpenChange={(next) => !next && setViewer(null)}>
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
        <div className="max-h-[60vh] overflow-auto rounded-md border bg-muted/40 p-3">
          <pre className="whitespace-pre-wrap break-words font-mono text-xs leading-5">
            {viewer?.text}
          </pre>
        </div>
        {viewer && !viewer.eof ? (
          <DialogFooter>
            <Button variant="outline" disabled={loading} onClick={() => void loadMore()}>
              {loading ? <LoaderCircle className="animate-spin" /> : null}
              {t("继续加载")}
            </Button>
          </DialogFooter>
        ) : null}
      </DialogContent>
    </Dialog>
  );

  return { open, loading, dialog };
}
