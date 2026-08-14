import type { ReactNode } from "react";
import {
  Activity,
  BookOpen,
  Bot,
  BrainCircuit,
  Boxes,
  KeyRound,
  ChevronDown,
  Gauge,
  MessageSquareText,
  Languages,
  Monitor,
  Moon,
  PanelLeft,
  Settings,
  SlidersHorizontal,
  Sun,
  UserRound,
} from "lucide-react";
import type { RuntimeStatus } from "@/lib/types";
import {
  runtimeStateSource,
  useI18n,
  type LanguagePreference,
} from "@/lib/i18n";
import { useTheme, type ThemeMode } from "@/lib/theme";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

export const NAV_ITEMS = [
  { id: "overview", label: "总览", icon: Gauge },
  { id: "chat", label: "对话", icon: MessageSquareText },
  { id: "trace", label: "召回轨迹", icon: Activity },
  { id: "knowledge", label: "知识库", icon: BookOpen },
  { id: "reflection", label: "反思记忆", icon: BrainCircuit },
  { id: "editor", label: "轨迹微调", icon: SlidersHorizontal },
  { id: "catalog", label: "模型目录", icon: Boxes },
  { id: "api-keys", label: "API 密钥", icon: KeyRound },
  { id: "settings", label: "设置", icon: Settings },
] as const;

export type ViewId = (typeof NAV_ITEMS)[number]["id"];

export function AppShell({
  view,
  onView,
  status,
  children,
}: {
  view: ViewId;
  onView: (view: ViewId) => void;
  status: RuntimeStatus | null;
  children: ReactNode;
}) {
  const ready = status?.runtime_state === "ready";
  const current = NAV_ITEMS.find((item) => item.id === view) || NAV_ITEMS[0];
  const { mode, setMode, dark } = useTheme();
  const { language, preference, setPreference, t } = useI18n();

  return (
    <div className="app-shell">
      <aside className="app-sidebar">
        <div className="flex h-16 items-center gap-3 border-b px-4">
          <div className="brand-mark" aria-hidden="true">
            QE
          </div>
          <div className="min-w-0">
            <div className="text-sm font-semibold tracking-tight">QWEN EXO</div>
            <div className="truncate text-[11px] text-muted-foreground">
              {t("模型原生记忆")}
            </div>
          </div>
        </div>

        <nav className="flex-1 space-y-1 px-3 py-4" aria-label={t("主导航")}>
          <div className="eyebrow mb-3 px-3">{t("工作区")}</div>
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                className="nav-item"
                data-active={view === item.id}
                onClick={() => onView(item.id)}
              >
                <Icon className="h-4 w-4" />
                <span>{t(item.label)}</span>
              </button>
            );
          })}
        </nav>

        <div className="border-t p-3">
          <div className="mb-2 flex items-center justify-between rounded-md border bg-muted/40 px-3 py-2.5">
            <div className="flex min-w-0 items-center gap-2">
              <span
                className="status-dot"
                data-state={
                  ready ? "ready" : status?.runtime_state || "unavailable"
                }
              />
              <div className="min-w-0">
                <div className="text-xs font-medium">
                  {ready ? t("服务就绪") : t("服务未就绪")}
                </div>
                <div className="truncate font-mono text-[10px] text-muted-foreground">
                  TP {status?.tp_size ?? status?.hybrid_state?.tp_size ?? "—"}
                </div>
              </div>
            </div>
            <Badge variant={ready ? "success" : "warning"}>
              {t(runtimeStateSource(status?.runtime_state, "离线"))}
            </Badge>
          </div>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button className="flex w-full items-center gap-2 rounded-md px-2 py-2 text-left hover:bg-muted">
                <div className="grid h-8 w-8 place-items-center rounded-md bg-foreground text-background">
                  <UserRound className="h-4 w-4" />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="text-xs font-medium">{t("本机管理员")}</div>
                  <div className="truncate text-[10px] text-muted-foreground">
                    {t("运维工作区")}
                  </div>
                </div>
                <ChevronDown className="h-4 w-4 text-muted-foreground" />
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" side="right" className="w-56">
              <DropdownMenuLabel>QWEN EXO Console</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem onSelect={() => onView("settings")}>
                <Settings />
                {t("服务配置")}
              </DropdownMenuItem>
              <DropdownMenuItem onSelect={() => window.open("/docs", "_blank")}>
                <BookOpen />
                {t("API 文档")}
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </aside>

      <main className="app-main">
        <header className="sticky top-0 z-20 flex h-14 items-center justify-between border-b bg-background/95 px-4 backdrop-blur sm:px-8">
          <div className="flex items-center gap-3">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="icon" className="mobile-only">
                  <PanelLeft />
                  <span className="sr-only">{t("打开导航")}</span>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="start" className="w-52">
                {NAV_ITEMS.map((item) => (
                  <DropdownMenuItem
                    key={item.id}
                    onSelect={() => onView(item.id)}
                  >
                    {<item.icon />}
                    {t(item.label)}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
            <div>
              <div className="text-xs font-semibold">{t(current.label)}</div>
              <div className="hidden text-[10px] text-muted-foreground sm:block">
                QWEN EXO /{" "}
                {t(runtimeStateSource(status?.runtime_state, "连接中"))}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="outline"
                  size="sm"
                  className="h-9 gap-1.5 px-2.5"
                  aria-label={t("切换语言")}
                >
                  <Languages />
                  <span>{language === "en-US" ? "EN" : "中"}</span>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-48">
                <DropdownMenuRadioGroup
                  value={preference}
                  onValueChange={(value) =>
                    setPreference(value as LanguagePreference)
                  }
                >
                  <DropdownMenuRadioItem value="browser">
                    <Monitor />
                    {t("跟随浏览器")}
                  </DropdownMenuRadioItem>
                  <DropdownMenuRadioItem value="zh-CN">
                    {t("中文")}
                  </DropdownMenuRadioItem>
                  <DropdownMenuRadioItem value="en-US">
                    {t("英语")}
                  </DropdownMenuRadioItem>
                </DropdownMenuRadioGroup>
              </DropdownMenuContent>
            </DropdownMenu>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="icon">
                  {dark ? <Moon /> : <Sun />}
                  <span className="sr-only">{t("切换主题")}</span>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-40">
                <DropdownMenuRadioGroup
                  value={mode}
                  onValueChange={(value) => setMode(value as ThemeMode)}
                >
                  <DropdownMenuRadioItem value="light">
                    <Sun />
                    {t("浅色")}
                  </DropdownMenuRadioItem>
                  <DropdownMenuRadioItem value="dark">
                    <Moon />
                    {t("深色")}
                  </DropdownMenuRadioItem>
                  <DropdownMenuRadioItem value="system">
                    <Monitor />
                    {t("跟随系统")}
                  </DropdownMenuRadioItem>
                </DropdownMenuRadioGroup>
              </DropdownMenuContent>
            </DropdownMenu>
            <Bot className="h-4 w-4" />
            <span className="hidden max-w-64 truncate sm:inline">
              {String(
                status?.model?.model_path ||
                  status?.model_path ||
                  t("等待模型信息"),
              )}
            </span>
          </div>
        </header>
        {children}
      </main>
    </div>
  );
}
