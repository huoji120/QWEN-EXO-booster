import { useState } from "react";
import {
  ChevronDown,
  ChevronRight,
  Folder,
  FolderInput,
  FolderOpen,
  FolderPlus,
  MoreHorizontal,
  Pencil,
  Trash2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { translate as t } from "@/lib/i18n";
import { cn, formatNumber } from "@/lib/utils";
import type { FolderNode } from "./knowledge-utils";

export type FolderAction = "create" | "rename" | "move" | "delete";

type FolderTreeProps = {
  root: FolderNode;
  /** Chinese source label for the root node. */
  rootLabel: string;
  selected: string;
  onSelect: (path: string) => void;
  onAction: (action: FolderAction, path: string) => void;
};

export function KnowledgeFolderTree({
  root,
  rootLabel,
  selected,
  onSelect,
  onAction,
}: FolderTreeProps) {
  // Collapsed folders are tracked, so new folders start expanded.
  const [collapsed, setCollapsed] = useState<Set<string>>(() => new Set());

  const toggle = (path: string) =>
    setCollapsed((previous) => {
      const next = new Set(previous);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });

  const renderNode = (node: FolderNode, depth: number) => {
    const isRoot = node.path === root.path;
    const open = !collapsed.has(node.path);
    const active = selected === node.path;
    const Icon = active ? FolderOpen : Folder;
    return (
      <li key={node.path || "/"}>
        <div
          className={cn(
            "group flex items-center gap-1 rounded-md pr-1 text-sm",
            active ? "bg-primary/10 text-primary" : "hover:bg-muted",
          )}
          style={{ paddingLeft: depth * 14 }}
        >
          <button
            type="button"
            className={cn(
              "flex h-7 w-5 shrink-0 items-center justify-center text-muted-foreground",
              !node.children.length && "invisible",
            )}
            aria-label={open ? t("折叠") : t("展开")}
            onClick={() => toggle(node.path)}
          >
            {open ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
          </button>
          <button
            type="button"
            className="flex min-w-0 flex-1 items-center gap-1.5 py-1 text-left"
            title={node.path || t(rootLabel)}
            onClick={() => onSelect(node.path)}
          >
            <Icon className="h-4 w-4 shrink-0" />
            <span className="truncate">{isRoot ? t(rootLabel) : node.name}</span>
            <span className="ml-auto shrink-0 pl-1 font-mono text-[11px] text-muted-foreground">
              {formatNumber(node.fileCount)}
            </span>
          </button>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6 shrink-0 opacity-60 group-hover:opacity-100"
                aria-label={t("文件夹操作")}
              >
                <MoreHorizontal />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onSelect={() => onAction("create", node.path)}>
                <FolderPlus className="h-4 w-4" />
                {t("新建文件夹")}
              </DropdownMenuItem>
              {isRoot ? null : (
                <>
                  <DropdownMenuItem onSelect={() => onAction("rename", node.path)}>
                    <Pencil className="h-4 w-4" />
                    {t("重命名")}
                  </DropdownMenuItem>
                  <DropdownMenuItem onSelect={() => onAction("move", node.path)}>
                    <FolderInput className="h-4 w-4" />
                    {t("移动到…")}
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem
                    className="text-destructive focus:text-destructive"
                    onSelect={() => onAction("delete", node.path)}
                  >
                    <Trash2 className="h-4 w-4" />
                    {t("删除文件夹")}
                  </DropdownMenuItem>
                </>
              )}
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
        {open && node.children.length ? (
          <ul>{node.children.map((child) => renderNode(child, depth + 1))}</ul>
        ) : null}
      </li>
    );
  };

  return <ul className="space-y-0.5">{renderNode(root, 0)}</ul>;
}
