import {
  FileText,
  FolderInput,
  FolderOpen,
  LoaderCircle,
  MoreHorizontal,
  Pencil,
  ScrollText,
  Sparkles,
  Trash2,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { translate as t } from "@/lib/i18n";
import type { KnowledgeFile } from "@/lib/types";
import { formatBytes, formatNumber, formatTime, shortHash } from "@/lib/utils";
import { kindLabel, relativeTo, type SummaryView } from "./knowledge-utils";

export type FileAction =
  | "view"
  | "summary"
  | "generate"
  | "rename"
  | "move"
  | "delete";

export function SummaryBadge({ view }: { view: SummaryView }) {
  const badge = (
    <Badge variant={view.variant} className="whitespace-nowrap">
      {view.state === "running" ? <LoaderCircle className="h-3 w-3 animate-spin" /> : null}
      {t(view.label)}
      {view.progress ? <span className="font-mono">{view.progress}</span> : null}
    </Badge>
  );
  if (!view.detail) return badge;
  // Failure details are raw server text; the rest are Chinese source strings.
  const detail = view.state === "failed" ? view.detail : t(view.detail);
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className="inline-flex cursor-help">{badge}</span>
      </TooltipTrigger>
      <TooltipContent className="max-w-sm break-words">{detail}</TooltipContent>
    </Tooltip>
  );
}

export function KnowledgeFileTable({
  files,
  folder,
  views,
  selected,
  viewerLoading,
  onToggle,
  onToggleAll,
  onAction,
}: {
  files: KnowledgeFile[];
  folder: string;
  views: Map<string, SummaryView>;
  selected: Set<string>;
  viewerLoading: boolean;
  onToggle: (path: string, checked: boolean) => void;
  onToggleAll: (checked: boolean) => void;
  onAction: (action: FileAction, file: KnowledgeFile) => void;
}) {
  const allChecked = files.length > 0 && files.every((file) => selected.has(file.relative_path));
  const someChecked = !allChecked && files.some((file) => selected.has(file.relative_path));

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead className="w-8">
            <input
              type="checkbox"
              aria-label={t("全选")}
              className="h-4 w-4 accent-primary"
              checked={allChecked}
              ref={(element) => {
                if (element) element.indeterminate = someChecked;
              }}
              onChange={(event) => onToggleAll(event.target.checked)}
            />
          </TableHead>
          <TableHead>{t("名称")}</TableHead>
          <TableHead className="w-24">{t("类型")}</TableHead>
          <TableHead>{t("标题")}</TableHead>
          <TableHead className="w-24">{t("大小")}</TableHead>
          <TableHead className="w-20">{t("行数")}</TableHead>
          <TableHead className="w-28">{t("总结")}</TableHead>
          <TableHead className="w-44 text-right">{t("操作")}</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {files.map((file) => {
          const path = file.relative_path;
          const view = views.get(path);
          const readable = file.encoding === "utf-8";
          const hasSummary = Boolean(file.summary);
          return (
            <TableRow key={path} data-state={selected.has(path) ? "selected" : undefined}>
              <TableCell>
                <input
                  type="checkbox"
                  aria-label={t("选择")}
                  className="h-4 w-4 accent-primary"
                  checked={selected.has(path)}
                  onChange={(event) => onToggle(path, event.target.checked)}
                />
              </TableCell>
              <TableCell
                className="break-all font-mono text-xs"
                title={`${path}\nSHA-256 ${shortHash(file.sha256, 16)}\n${t("索引时间")} ${formatTime(file.indexed_ns / 1_000_000_000)}`}
              >
                {relativeTo(path, folder)}
                {!readable ? (
                  <Badge variant="outline" className="ml-2">
                    {t("非 UTF-8，不可检索")}
                  </Badge>
                ) : null}
              </TableCell>
              <TableCell>
                <Badge variant="secondary" className="whitespace-nowrap">
                  {t(kindLabel(file.kind))}
                </Badge>
              </TableCell>
              <TableCell className="max-w-[18rem]">
                <div className="truncate text-xs" title={file.title}>
                  {file.title || "—"}
                </div>
              </TableCell>
              <TableCell className="whitespace-nowrap font-mono text-xs">
                {formatBytes(file.byte_count)}
              </TableCell>
              <TableCell className="font-mono text-xs">
                {formatNumber(file.line_count)}
              </TableCell>
              <TableCell>{view ? <SummaryBadge view={view} /> : null}</TableCell>
              <TableCell className="text-right">
                <div className="flex justify-end gap-1">
                  <Button
                    variant="ghost"
                    size="sm"
                    disabled={viewerLoading || !readable}
                    onClick={() => onAction("view", file)}
                  >
                    <FolderOpen />
                    {t("查看")}
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    disabled={!readable}
                    onClick={() => onAction(hasSummary ? "summary" : "generate", file)}
                  >
                    {hasSummary ? <ScrollText /> : <Sparkles />}
                    {hasSummary ? t("查看总结") : t("生成总结")}
                  </Button>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" size="icon" className="h-8 w-8" aria-label={t("更多操作")}>
                        <MoreHorizontal />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem
                        disabled={!readable}
                        onSelect={() => onAction("summary", file)}
                      >
                        <FileText className="h-4 w-4" />
                        {hasSummary ? t("查看总结") : t("编辑总结")}
                      </DropdownMenuItem>
                      <DropdownMenuItem onSelect={() => onAction("rename", file)}>
                        <Pencil className="h-4 w-4" />
                        {t("重命名")}
                      </DropdownMenuItem>
                      <DropdownMenuItem onSelect={() => onAction("move", file)}>
                        <FolderInput className="h-4 w-4" />
                        {t("移动到…")}
                      </DropdownMenuItem>
                      <DropdownMenuSeparator />
                      <DropdownMenuItem
                        className="text-destructive focus:text-destructive"
                        onSelect={() => onAction("delete", file)}
                      >
                        <Trash2 className="h-4 w-4" />
                        {t("删除")}
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              </TableCell>
            </TableRow>
          );
        })}
      </TableBody>
    </Table>
  );
}
