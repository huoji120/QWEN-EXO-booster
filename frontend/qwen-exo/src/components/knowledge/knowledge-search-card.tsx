import { useState } from "react";
import { LoaderCircle, Search } from "lucide-react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { searchKnowledge } from "@/lib/api";
import { translate as t } from "@/lib/i18n";
import type { KnowledgeSearchResponse } from "@/lib/types";
import { formatNumber } from "@/lib/utils";
import { errorText } from "./knowledge-utils";

/** Full-text search over the knowledge files (``/knowledge/search``). */
export function KnowledgeSearchCard({
  onOpen,
  resetKey,
}: {
  onOpen: (path: string) => void;
  /** Changing this clears stale results (e.g. after an import). */
  resetKey: number;
}) {
  const [pattern, setPattern] = useState("");
  const [pathGlob, setPathGlob] = useState("**/*");
  const [caseSensitive, setCaseSensitive] = useState(false);
  const [regex, setRegex] = useState(false);
  const [searching, setSearching] = useState(false);
  const [result, setResult] = useState<{
    key: number;
    response: KnowledgeSearchResponse;
  } | null>(null);
  const searchResult = result && result.key === resetKey ? result.response : null;

  const runSearch = async () => {
    if (!pattern.trim() || searching) return;
    setSearching(true);
    try {
      const response = await searchKnowledge({
        pattern: pattern.trim(),
        pathGlob: pathGlob.trim() || "**/*",
        caseSensitive,
        regex,
      });
      setResult({ key: resetKey, response });
    } catch (error) {
      toast.error(t("搜索失败"), { description: errorText(error, t("未知错误")) });
    } finally {
      setSearching(false);
    }
  };

  return (
    <Card>
      <CardContent className="flex flex-col gap-3 pt-6">
        <div className="grid gap-3 sm:grid-cols-[minmax(0,2fr)_minmax(0,1fr)_auto]">
          <div className="space-y-2">
            <Label htmlFor="knowledge-pattern">{t("搜索原文")}</Label>
            <Input
              id="knowledge-pattern"
              value={pattern}
              placeholder={t("输入要检索的文本")}
              onChange={(event) => setPattern(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") void runSearch();
              }}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="knowledge-glob">{t("路径过滤")}</Label>
            <Input
              id="knowledge-glob"
              value={pathGlob}
              placeholder="**/*"
              onChange={(event) => setPathGlob(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") void runSearch();
              }}
            />
          </div>
          <div className="flex items-end gap-4">
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={caseSensitive}
                onChange={(event) => setCaseSensitive(event.target.checked)}
                className="h-4 w-4 accent-primary"
              />
              {t("区分大小写")}
            </label>
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={regex}
                onChange={(event) => setRegex(event.target.checked)}
                className="h-4 w-4 accent-primary"
              />
              {t("正则表达式")}
            </label>
            <Button disabled={!pattern.trim() || searching} onClick={() => void runSearch()}>
              {searching ? <LoaderCircle className="animate-spin" /> : <Search />}
              {t("搜索")}
            </Button>
          </div>
        </div>
        {searchResult ? (
          searchResult.results.length ? (
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <span>
                  {t("{count} 条结果", { count: formatNumber(searchResult.count) })}
                </span>
                {searchResult.truncated ? (
                  <Badge variant="outline">{t("结果已截断")}</Badge>
                ) : null}
              </div>
              <div className="max-h-[420px] overflow-auto rounded-md border">
                <ul className="divide-y">
                  {searchResult.results.map((item, index) => (
                    <li key={`${item.path}:${item.line}:${index}`}>
                      <button
                        type="button"
                        className="w-full px-3 py-2 text-left hover:bg-muted"
                        onClick={() => onOpen(item.path)}
                      >
                        <div className="font-mono text-xs text-muted-foreground">
                          {item.path}:{item.line}
                        </div>
                        {item.before.map((line, beforeIndex) => (
                          <div
                            key={`before-${beforeIndex}`}
                            className="mt-1 whitespace-pre-wrap break-words font-mono text-xs text-muted-foreground"
                          >
                            {line}
                          </div>
                        ))}
                        <div className="mt-1 whitespace-pre-wrap break-words font-mono text-xs">
                          {item.line_text ?? item.text}
                        </div>
                        {item.after.map((line, afterIndex) => (
                          <div
                            key={`after-${afterIndex}`}
                            className="whitespace-pre-wrap break-words font-mono text-xs text-muted-foreground"
                          >
                            {line}
                          </div>
                        ))}
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">{t("没有匹配结果")}</p>
          )
        ) : null}
      </CardContent>
    </Card>
  );
}
