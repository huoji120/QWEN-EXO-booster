import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ChevronDown,
  ChevronRight,
  LoaderCircle,
  RefreshCw,
  RotateCcw,
  Trash2,
} from "lucide-react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { getWorkTrash, restoreWorkTrash } from "@/lib/api";
import { translate as t } from "@/lib/i18n";
import type { WorkTrashEntry } from "@/lib/types";
import { formatBytes, formatNumber } from "@/lib/utils";
import { errorText } from "./knowledge-utils";

/** Newest batch first: ``dream-12`` before ``dream-3``. */
function compareBatches(a: string, b: string) {
  const number = (value: string) => Number(/(\d+)$/.exec(value)?.[1] ?? -1);
  return number(b) - number(a) || b.localeCompare(a);
}

/** Collapsible list of deleted work notes, grouped by batch, each restorable. */
export function KnowledgeWorkTrash({
  refreshKey,
  onRestored,
}: {
  /** Bump to reload (e.g. after a dream run finished). */
  refreshKey: number;
  onRestored: () => void | Promise<void>;
}) {
  const [open, setOpen] = useState(false);
  const [entries, setEntries] = useState<WorkTrashEntry[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [restoring, setRestoring] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setEntries((await getWorkTrash()).entries);
      setError(null);
    } catch (cause) {
      setError(errorText(cause, t("未知错误")));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load, refreshKey]);

  const groups = useMemo(() => {
    const map = new Map<string, WorkTrashEntry[]>();
    for (const entry of entries ?? []) {
      const list = map.get(entry.batch) ?? [];
      list.push(entry);
      map.set(entry.batch, list);
    }
    return Array.from(map.entries()).sort(([a], [b]) => compareBatches(a, b));
  }, [entries]);

  const restore = async (entry: WorkTrashEntry) => {
    const key = `${entry.batch}:${entry.path}`;
    setRestoring(key);
    try {
      const result = await restoreWorkTrash(entry.batch, entry.path);
      toast.success(t("已恢复"), { description: result.restored || entry.path });
      await load();
      await onRestored();
    } catch (cause) {
      toast.error(t("恢复失败"), { description: errorText(cause, t("未知错误")) });
    } finally {
      setRestoring(null);
    }
  };

  return (
    <Card>
      <CardContent className="space-y-3 pt-6 text-xs">
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            className="flex items-center gap-1.5 text-sm font-medium"
            aria-expanded={open}
            onClick={() => setOpen((value) => !value)}
          >
            {open ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
            <Trash2 className="h-4 w-4 text-muted-foreground" />
            {t("回收站")}
          </button>
          {entries ? (
            <Badge variant="secondary">
              {t("{count} 个文件", { count: formatNumber(entries.length) })}
            </Badge>
          ) : null}
          {open ? (
            <Button
              className="ml-auto"
              variant="ghost"
              size="sm"
              disabled={loading}
              onClick={() => void load()}
            >
              {loading ? <LoaderCircle className="animate-spin" /> : <RefreshCw />}
              {t("刷新")}
            </Button>
          ) : null}
        </div>
        {!open ? null : error ? (
          <p className="break-words text-destructive">{error}</p>
        ) : entries === null ? (
          <p className="text-muted-foreground">{t("加载中")}</p>
        ) : !groups.length ? (
          <p className="py-4 text-center text-muted-foreground">{t("回收站是空的")}</p>
        ) : (
          <div className="space-y-3">
            {groups.map(([batch, items]) => (
              <div key={batch} className="rounded-md border">
                <div className="flex items-center gap-2 border-b bg-muted/30 px-2 py-1.5">
                  <span className="font-mono font-medium">{batch}</span>
                  <span className="text-muted-foreground">
                    {t("{count} 个文件", { count: formatNumber(items.length) })}
                  </span>
                </div>
                <ul className="divide-y">
                  {items.map((entry) => {
                    const key = `${entry.batch}:${entry.path}`;
                    return (
                      <li key={key} className="flex items-center gap-2 px-2 py-1.5">
                        <span className="min-w-0 flex-1 break-all font-mono">{entry.path}</span>
                        <span className="shrink-0 text-muted-foreground">
                          {formatBytes(entry.bytes)}
                        </span>
                        <Button
                          variant="outline"
                          size="sm"
                          disabled={restoring !== null}
                          onClick={() => void restore(entry)}
                        >
                          {restoring === key ? (
                            <LoaderCircle className="animate-spin" />
                          ) : (
                            <RotateCcw />
                          )}
                          {t("恢复")}
                        </Button>
                      </li>
                    );
                  })}
                </ul>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
