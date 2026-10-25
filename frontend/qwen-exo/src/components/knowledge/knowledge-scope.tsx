import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { createKnowledgeApi, type KnowledgeApi } from "@/lib/api";

const KnowledgeApiContext = createContext<KnowledgeApi | null>(null);

/** Remounted by scope: aborts all old reads and mutations, including child dialogs. */
export function KnowledgeScope({ memoryKeyId, children }: {
  memoryKeyId?: string;
  children: ReactNode;
}) {
  const [api, setApi] = useState<KnowledgeApi | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    setApi(createKnowledgeApi(memoryKeyId, controller.signal));
    return () => controller.abort();
  }, [memoryKeyId]);
  return api ? <KnowledgeApiContext.Provider value={api}>{children}</KnowledgeApiContext.Provider> : null;
}

export function useKnowledgeApi() {
  const api = useContext(KnowledgeApiContext);
  if (!api) throw new Error("Knowledge API requires a mounted memory scope");
  return api;
}
