import type {
  KnowledgeFile,
  KnowledgeFileKind,
  KnowledgeSummaryStatus,
} from "@/lib/types";

/** Chinese source strings; pass through ``t()`` when rendering. */
export const KIND_LABELS: Record<KnowledgeFileKind, string> = {
  summary: "总结",
  note: "笔记",
  code: "代码",
  other: "其他",
  data: "数据",
  log: "日志",
  session: "会话记录",
  trajectory: "轨迹",
  work: "工作记忆",
};

/** Top-level folder holding the notes the model writes itself. */
export const WORK_ROOT = "work-memory";

export type MemoryArea = "imported" | "work";

/** Chinese source labels for the two memory areas. */
export const AREA_LABELS: Record<MemoryArea, string> = {
  imported: "导入的记忆",
  work: "日常工作记忆",
};

export function areaOf(path: string): MemoryArea {
  return path === WORK_ROOT || path.startsWith(`${WORK_ROOT}/`) ? "work" : "imported";
}

export function areaRoot(area: MemoryArea) {
  return area === "work" ? WORK_ROOT : "";
}

export function kindLabel(kind: string) {
  return KIND_LABELS[kind as KnowledgeFileKind] ?? kind;
}

export function normalizePath(path: string) {
  return path
    .trim()
    .replace(/\\/g, "/")
    .replace(/\/{2,}/g, "/")
    .replace(/^\/+|\/+$/g, "");
}

export function joinPath(directory: string, path: string) {
  const prefix = normalizePath(directory);
  const rest = normalizePath(path);
  if (!prefix) return rest;
  return rest ? `${prefix}/${rest}` : prefix;
}

export function parentOf(path: string) {
  const index = path.lastIndexOf("/");
  return index < 0 ? "" : path.slice(0, index);
}

export function baseName(path: string) {
  return path.slice(path.lastIndexOf("/") + 1);
}

/** True when ``path`` is ``folder`` itself or lies anywhere below it. */
export function isWithin(path: string, folder: string) {
  return !folder || path === folder || path.startsWith(`${folder}/`);
}

/** Path shown relative to the selected folder. */
export function relativeTo(path: string, folder: string) {
  return folder && path.startsWith(`${folder}/`)
    ? path.slice(folder.length + 1)
    : path;
}

/** Rewrite ``path`` after ``from`` was moved to ``to`` (file or folder). */
export function rebasePath(path: string, from: string, to: string) {
  if (path === from) return to;
  if (from && path.startsWith(`${from}/`)) return to + path.slice(from.length);
  return path;
}

export type SummaryState =
  | "none"
  | "running"
  | "queued"
  | "ready"
  | "stale"
  | "manual"
  | "failed";

export type SummaryView = {
  state: SummaryState;
  label: string;
  variant: "outline" | "secondary" | "success" | "warning" | "destructive" | "default";
  /** Chinese source tooltip or raw server text (error). */
  detail?: string;
  progress?: string;
};

export function summaryView(
  file: KnowledgeFile,
  status: KnowledgeSummaryStatus | null,
  failures: Map<string, string>,
  queued: Set<string>,
): SummaryView {
  const path = file.relative_path;
  const current = status?.current;
  if (current && current.path === path) {
    return {
      state: "running",
      label: "生成中",
      variant: "default",
      progress: current.chunks > 1 ? `${current.done}/${current.chunks}` : undefined,
    };
  }
  if (queued.has(path)) {
    return { state: "queued", label: "排队中", variant: "secondary" };
  }
  const error = failures.get(path);
  if (error !== undefined) {
    return { state: "failed", label: "失败", variant: "destructive", detail: error };
  }
  const summary = file.summary;
  if (!summary) return { state: "none", label: "无", variant: "outline" };
  if (summary.origin === "manual") {
    return {
      state: "manual",
      label: "手动",
      variant: summary.stale ? "warning" : "success",
      detail: summary.stale ? "文件已变更，手动总结不会被自动覆盖" : undefined,
    };
  }
  if (summary.stale) {
    return {
      state: "stale",
      label: "已过期",
      variant: "warning",
      detail: "文件已变更，总结待重新生成",
    };
  }
  return {
    state: "ready",
    label: "已生成",
    variant: "success",
    detail: summary.sampled ? "基于均匀采样片段生成，未逐字读完" : undefined,
  };
}

export type FolderNode = {
  path: string;
  name: string;
  children: FolderNode[];
  /** Files anywhere below this folder. */
  fileCount: number;
};

/**
 * Build the folder tree from the server directory list plus file parents.
 * Callers pass only the entries of one area; ``rootPath`` is that area's
 * root folder (``""`` for the knowledge root) and is never listed itself.
 */
export function buildFolderTree(
  directories: string[],
  files: KnowledgeFile[] = [],
  rootPath = "",
  counts?: Record<string, { files: number }>,
) {
  const all = new Set<string>();
  const addWithParents = (path: string) => {
    let current = normalizePath(path);
    while (current && current !== rootPath && !all.has(current)) {
      all.add(current);
      current = parentOf(current);
    }
  };
  directories.forEach(addWithParents);
  if (!counts) files.forEach((file) => addWithParents(parentOf(file.relative_path)));

  const root: FolderNode = {
    path: rootPath,
    name: baseName(rootPath),
    children: [],
    fileCount: counts ? counts[rootPath]?.files ?? 0 : files.length,
  };
  const nodes = new Map<string, FolderNode>([[rootPath, root]]);
  for (const path of Array.from(all).sort((a, b) => a.localeCompare(b))) {
    const node: FolderNode = {
      path,
      name: baseName(path),
      children: [],
      fileCount: counts ? counts[path]?.files ?? 0 : 0,
    };
    nodes.set(path, node);
  }
  for (const [path, node] of nodes) {
    if (path === rootPath) continue;
    nodes.get(parentOf(path))?.children.push(node);
  }
  if (!counts) {
    for (const file of files) {
      let folder = parentOf(file.relative_path);
      while (folder && folder !== rootPath) {
        const node = nodes.get(folder);
        if (node) node.fileCount += 1;
        folder = parentOf(folder);
      }
    }
  }
  const sortChildren = (node: FolderNode) => {
    node.children.sort((a, b) => a.name.localeCompare(b.name));
    node.children.forEach(sortChildren);
  };
  sortChildren(root);
  return { root, folders: Array.from(all).sort((a, b) => a.localeCompare(b)) };
}

export type PendingFile = { file: File; relativePath: string };

export const isZip = (item: PendingFile) => /\.zip$/i.test(item.file.name);

/** Folder-relative path for picked/dropped files; bare name otherwise. */
export function pickedPath(file: File) {
  const relative = (file as File & { webkitRelativePath?: string })
    .webkitRelativePath;
  return (relative || file.name).replace(/\\/g, "/").replace(/^\/+/, "");
}

/** Expand dropped items, walking dropped folders and keeping their paths. */
export async function collectDropped(
  items: DataTransferItemList,
): Promise<PendingFile[]> {
  const out: PendingFile[] = [];
  const walk = async (entry: FileSystemEntry): Promise<void> => {
    if (entry.isFile) {
      const file = await new Promise<File>((resolve, reject) =>
        (entry as FileSystemFileEntry).file(resolve, reject),
      );
      out.push({ file, relativePath: entry.fullPath.replace(/^\/+/, "") });
    } else if (entry.isDirectory) {
      const reader = (entry as FileSystemDirectoryEntry).createReader();
      // readEntries returns batches; an empty batch means done.
      for (;;) {
        const batch = await new Promise<FileSystemEntry[]>((resolve, reject) =>
          reader.readEntries(resolve, reject),
        );
        if (!batch.length) break;
        for (const child of batch) await walk(child);
      }
    }
  };
  const entries = Array.from(items)
    .map((item) => item.webkitGetAsEntry())
    .filter((entry): entry is FileSystemEntry => entry !== null);
  for (const entry of entries) await walk(entry);
  return out;
}

export function errorText(error: unknown, fallback: string) {
  return error instanceof Error ? error.message : fallback;
}
