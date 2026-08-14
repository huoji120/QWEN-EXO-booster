import { useCallback, useEffect, useState } from "react";
import { Toaster, toast } from "sonner";
import { AppShell, NAV_ITEMS, type ViewId } from "@/components/app-shell";
import { getStatus } from "@/lib/api";
import { useI18n } from "@/lib/i18n";
import { useTheme } from "@/lib/theme";
import type { RuntimeStatus } from "@/lib/types";
import { ApiKeysPage } from "@/pages/api-keys-page";
import { CatalogPage } from "@/pages/catalog-page";
import { ChatPage } from "@/pages/chat-page";
import { EditorPage } from "@/pages/editor-page";
import { KnowledgePage } from "@/pages/knowledge-page";
import { OverviewPage } from "@/pages/overview-page";
import { ReflectionPage } from "@/pages/reflection-page";
import { SettingsPage } from "@/pages/settings-page";
import { TracePage } from "@/pages/trace-page";

const VALID_VIEWS: Record<string, true> = Object.fromEntries(
  NAV_ITEMS.map((item) => [item.id, true]),
);

function initialView(): ViewId {
  const hash = window.location.hash.replace(/^#\/?/, "");
  return VALID_VIEWS[hash] ? (hash as ViewId) : "overview";
}

export default function App() {
  const [view, setView] = useState<ViewId>(initialView);
  const [status, setStatus] = useState<RuntimeStatus | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const { dark } = useTheme();
  const { t } = useI18n();

  const loadStatus = useCallback(async () => {
    try {
      setStatus(await getStatus());
      setStatusError(null);
    } catch (error) {
      setStatusError(
        error instanceof Error ? error.message : t("状态请求失败"),
      );
    }
  }, [t]);

  useEffect(() => {
    void loadStatus();
    if (view === "trace") return;
    const timer = window.setInterval(() => void loadStatus(), 5000);
    return () => window.clearInterval(timer);
  }, [loadStatus, view]);

  useEffect(() => {
    const onHashChange = () => setView(initialView());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  useEffect(() => {
    if (statusError)
      toast.error(t("控制面连接失败"), {
        description: statusError,
        id: "status-error",
      });
  }, [statusError, t]);

  const navigate = (next: ViewId) => {
    setView(next);
    window.history.replaceState(null, "", `#/${next}`);
  };

  const pages: Record<ViewId, React.ReactNode> = {
    overview: <OverviewPage status={status} onRefresh={loadStatus} />,
    chat: <ChatPage />,
    trace: <TracePage />,
    knowledge: <KnowledgePage />,
    reflection: <ReflectionPage />,
    editor: <EditorPage />,
    catalog: <CatalogPage status={status} />,
    "api-keys": <ApiKeysPage />,
    settings: <SettingsPage status={status} onStatusRefresh={loadStatus} />,
  };

  return (
    <AppShell view={view} onView={navigate} status={status}>
      {pages[view]}
      <Toaster
        richColors
        closeButton
        position="bottom-right"
        theme={dark ? "dark" : "light"}
      />
    </AppShell>
  );
}
