import { createContext, useContext, useState, type DragEvent, type ReactNode } from "react";
import { baseName, joinPath } from "./knowledge-utils";

export const KNOWLEDGE_DRAG_TYPE = "application/x-qwen-memory-entry";
export type MemoryEntry = { keyId: string | null; path: string; kind: "file" | "folder" };
export type TransferSelection = { source: MemoryEntry; destinationKeyId: string | null; folder: string };

export function canDropMemory(source: MemoryEntry, keyId: string | null, folder: string) {
  if (!source.path) return false; // Whole spaces use the named-folder migration dialog.
  if (source.keyId !== keyId) return true;
  const destination = joinPath(folder, baseName(source.path));
  return destination !== source.path && !(source.kind === "folder" &&
    (folder === source.path || folder.startsWith(`${source.path}/`)));
}

const DragContext = createContext<{
  source: MemoryEntry | null;
  setSource: (source: MemoryEntry | null) => void;
  onTransfer: (selection: TransferSelection) => void;
} | null>(null);

export function KnowledgeDragProvider({ children, onTransfer }: {
  children: ReactNode;
  onTransfer: (selection: TransferSelection) => void;
}) {
  const [source, setSource] = useState<MemoryEntry | null>(null);
  return <DragContext.Provider value={{ source, setSource, onTransfer }}>{children}</DragContext.Provider>;
}

export function useMemoryDrag(entry: MemoryEntry) {
  const context = useContext(DragContext);
  return {
    draggable: Boolean(entry.path),
    onDragStart: (event: DragEvent) => {
      event.stopPropagation();
      if (!context || !entry.path) { event.preventDefault(); return; }
      event.dataTransfer.effectAllowed = "move";
      event.dataTransfer.setData(KNOWLEDGE_DRAG_TYPE, JSON.stringify(entry));
      context.setSource(entry);
    },
    onDragEnd: () => context?.setSource(null),
  };
}

export function useMemoryDrop(keyId: string | null, folder: string) {
  const context = useContext(DragContext);
  const [hovered, setHovered] = useState(false);
  const valid = Boolean(context?.source && canDropMemory(context.source, keyId, folder));
  return {
    className: hovered && context?.source
      ? valid ? "ring-2 ring-primary bg-primary/10" : "ring-2 ring-destructive bg-destructive/10"
      : "",
    handlers: {
      onDragOver: (event: DragEvent) => {
        if (!event.dataTransfer.types.includes(KNOWLEDGE_DRAG_TYPE)) return;
        event.preventDefault();
        event.stopPropagation();
        event.dataTransfer.dropEffect = valid ? "move" : "none";
        setHovered(true);
      },
      onDragLeave: (event: DragEvent) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setHovered(false);
      },
      onDrop: (event: DragEvent) => {
        if (!event.dataTransfer.types.includes(KNOWLEDGE_DRAG_TYPE)) return;
        event.preventDefault();
        event.stopPropagation();
        setHovered(false);
        // Only this mounted browser's captured owner can initiate a move. External
        // or stale payloads never acquire ownership by naming another KEY.
        const source = context?.source;
        if (source && valid && event.dataTransfer.getData(KNOWLEDGE_DRAG_TYPE) === JSON.stringify(source)) {
          context?.onTransfer({ source: { ...source }, destinationKeyId: keyId, folder });
        }
        context?.setSource(null);
      },
    },
  };
}
