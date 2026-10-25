import { useState } from "react";
import { ChevronDown, ChevronRight, Folder, FolderInput, FolderOpen, FolderPlus, MoreHorizontal, Pencil, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { translate as t } from "@/lib/i18n";
import { cn, formatNumber } from "@/lib/utils";
import { useMemoryDrag, useMemoryDrop } from "./knowledge-drag";
import type { FolderNode } from "./knowledge-utils";

export type FolderAction = "create" | "rename" | "move" | "delete";

type FolderTreeProps = {
  root: FolderNode;
  rootLabel: string;
  rootTitle?: string;
  selected: string;
  memoryKeyId?: string;
  onSelect: (path: string) => void;
  onAction?: (action: FolderAction, path: string) => void;
  onMigrate?: () => void;
  hideRoot?: boolean;
};

function FolderRow({ node, depth, isRoot, props }: { node: FolderNode; depth: number; isRoot: boolean; props: FolderTreeProps }) {
  const [open, setOpen] = useState(true);
  const active = props.selected === node.path;
  const drag = useMemoryDrag({ keyId: props.memoryKeyId ?? null, path: node.path, kind: "folder" });
  const drop = useMemoryDrop(props.memoryKeyId ?? null, node.path);
  const Icon = active ? FolderOpen : Folder;
  return <li>
    <div {...drop.handlers} {...(!isRoot ? drag : {})}
      className={cn("group flex min-w-0 items-center gap-1 rounded-md pr-1 text-sm", active ? "bg-primary/10 text-primary" : "hover:bg-muted", drop.className)}
      style={{ paddingLeft: depth * 14 }}>
      <button type="button" className={cn("flex h-7 w-5 shrink-0 items-center justify-center text-muted-foreground", !node.children.length && "invisible")}
        aria-label={open ? t("折叠") : t("展开")} onClick={() => setOpen(!open)}>
        {open ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
      </button>
      <button type="button" className="flex min-w-0 flex-1 items-center gap-1.5 py-1 text-left"
        title={isRoot ? props.rootTitle ?? props.rootLabel : node.path} onClick={() => props.onSelect(node.path)}>
        <Icon className="h-4 w-4 shrink-0" />
        <span className="min-w-0 truncate whitespace-nowrap">{isRoot ? props.rootLabel : node.name}</span>
        <span className="ml-auto shrink-0 pl-1 font-mono text-[11px] text-muted-foreground">{formatNumber(node.fileCount)}</span>
      </button>
      {props.onAction || (isRoot && props.onMigrate) ? <DropdownMenu>
        <DropdownMenuTrigger asChild><Button variant="ghost" size="icon" className="h-6 w-6 shrink-0 opacity-60 group-hover:opacity-100" aria-label={t("文件夹操作")}><MoreHorizontal /></Button></DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          {props.onAction ? <DropdownMenuItem onSelect={() => props.onAction?.("create", node.path)}><FolderPlus className="h-4 w-4" />{t("新建文件夹")}</DropdownMenuItem> : null}
          {isRoot && props.onMigrate ? <DropdownMenuItem onSelect={props.onMigrate}><FolderInput className="h-4 w-4" />{t("迁移记忆")}</DropdownMenuItem> : null}
          {!isRoot && props.onAction ? <>
            <DropdownMenuItem onSelect={() => props.onAction?.("rename", node.path)}><Pencil className="h-4 w-4" />{t("重命名")}</DropdownMenuItem>
            <DropdownMenuItem onSelect={() => props.onAction?.("move", node.path)}><FolderInput className="h-4 w-4" />{t("移动到…")}</DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem className="text-destructive focus:text-destructive" onSelect={() => props.onAction?.("delete", node.path)}><Trash2 className="h-4 w-4" />{t("删除文件夹")}</DropdownMenuItem>
          </> : null}
        </DropdownMenuContent>
      </DropdownMenu> : null}
    </div>
    {open && node.children.length ? <ul>{node.children.map((child) => <FolderRow key={child.path} node={child} depth={depth + 1} isRoot={false} props={props} />)}</ul> : null}
  </li>;
}

export function KnowledgeFolderTree(props: FolderTreeProps) {
  return <ul className="min-w-0 space-y-0.5">{props.hideRoot
    ? props.root.children.map((node) => <FolderRow key={node.path} node={node} depth={0} isRoot={false} props={props} />)
    : <FolderRow node={props.root} depth={0} isRoot props={props} />}</ul>;
}
