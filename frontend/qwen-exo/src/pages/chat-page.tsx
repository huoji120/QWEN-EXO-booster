import { useEffect, useMemo, useRef, useState } from "react";
import {
  Check,
  ChevronDown,
  Clipboard,
  CornerDownLeft,
  History,
  LoaderCircle,
  MessageSquarePlus,
  RotateCcw,
  Square,
  Trash2,
  Wrench,
} from "lucide-react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { getModelId, getRecallTrace, streamResponse } from "@/lib/api";
import { translate as t, translateFor } from "@/lib/i18n";
import { createSession, loadSessions, saveSessions } from "@/lib/session-store";
import type {
  ChatMessage,
  ChatSession,
  RecallTrace,
  ToolCall,
} from "@/lib/types";
import { cn, makeId } from "@/lib/utils";

const CONSOLE_MAX_OUTPUT_TOKENS = 8192;

const CONSOLE_TEXT_SOURCES = [
  "未命名工具",
  "本轮在返回文本前达到输出上限。",
  "本轮没有返回可显示文本。",
  "本轮生成已由你停止。",
] as const;
const CONSOLE_TEXT_SOURCE_BY_VALUE: Record<string, string> =
  Object.create(null);
for (const source of CONSOLE_TEXT_SOURCES) {
  CONSOLE_TEXT_SOURCE_BY_VALUE[source] = source;
  CONSOLE_TEXT_SOURCE_BY_VALUE[translateFor("en-US", source)] = source;
}

function toolFor(
  message: ChatMessage,
  item: Record<string, any>,
  itemId?: string,
): ToolCall {
  const id = String(itemId || item.id || item.call_id || makeId("tool"));
  let tool = message.tools.find((candidate) => candidate.id === id);
  if (!tool) {
    tool = {
      id,
      name: String(item.name || "未命名工具"),
      arguments: String(item.arguments || ""),
      done: false,
    };
    message.tools.push(tool);
  }
  if (item.name) tool.name = String(item.name);
  return tool;
}

function MessageCard({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  const [copied, setCopied] = useState(false);
  const contentSource = CONSOLE_TEXT_SOURCE_BY_VALUE[message.content];
  const content = contentSource ? t(contentSource) : message.content;

  const copy = async () => {
    await navigator.clipboard.writeText(content);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  };

  return (
    <article className={cn("group flex w-full", isUser && "justify-end")}>
      <div
        className={cn(
          "relative min-w-0 max-w-[min(800px,88%)]",
          isUser
            ? "rounded-2xl rounded-br-md bg-muted px-4 py-3"
            : "w-full pl-10",
          message.error && "text-destructive",
        )}
      >
        {!isUser ? (
          <div className="absolute left-0 top-0 grid h-7 w-7 place-items-center rounded-full bg-foreground font-mono text-[11px] font-semibold text-background">
            Q
          </div>
        ) : null}
        {!isUser &&
        (message.status === "in_progress" ||
          message.status === "incomplete" ||
          message.status === "failed") ? (
          <div className="mb-2 flex items-center gap-2">
            {message.status === "in_progress" ? (
              <LoaderCircle className="h-3.5 w-3.5 animate-spin text-muted-foreground" />
            ) : null}
            {message.status === "incomplete" ? (
              <Badge variant="warning">{t("达到上限")}</Badge>
            ) : null}
            {message.status === "failed" ? (
              <Badge variant="destructive">{t("失败")}</Badge>
            ) : null}
          </div>
        ) : null}
        {message.reasoning ? (
          <details
            className="mb-3 border-l-2 border-border pl-3 text-xs text-muted-foreground"
            open={message.status === "in_progress"}
          >
            <summary className="cursor-pointer select-none font-medium">
              {t("思考")} <ChevronDown className="ml-1 inline h-3 w-3" />
            </summary>
            <pre className="mt-2 max-h-72 overflow-auto whitespace-pre-wrap font-sans text-xs leading-6">
              {message.reasoning}
            </pre>
          </details>
        ) : null}
        {message.tools.length ? (
          <div className="mb-3 space-y-2">
            {message.tools.map((tool) => (
              <details key={tool.id} className="rounded-lg border bg-muted/40">
                <summary className="flex cursor-pointer list-none items-center gap-2 px-3 py-2 text-xs font-medium">
                  <Wrench className="h-3.5 w-3.5" />
                  <span className="min-w-0 flex-1 truncate">
                    {CONSOLE_TEXT_SOURCE_BY_VALUE[tool.name]
                      ? t(CONSOLE_TEXT_SOURCE_BY_VALUE[tool.name])
                      : tool.name}
                  </span>
                  {tool.done ? <Check className="h-3.5 w-3.5" /> : null}
                </summary>
                {tool.arguments ? (
                  <pre className="border-t px-3 py-2 font-mono text-[11px] leading-5 text-muted-foreground">
                    {tool.arguments}
                  </pre>
                ) : null}
              </details>
            ))}
          </div>
        ) : null}
        <div className="chat-copy">
          {content ||
            (message.status === "in_progress"
              ? t("正在生成…")
              : t("本轮没有可显示文本"))}
        </div>
        {!isUser && message.content ? (
          <button
            className="absolute right-0 top-0 rounded-md p-2 text-muted-foreground opacity-0 transition-opacity hover:bg-muted group-hover:opacity-100"
            onClick={() => void copy()}
            aria-label={t("复制回答")}
          >
            {copied ? (
              <Check className="h-4 w-4 text-emerald-500" />
            ) : (
              <Clipboard className="h-4 w-4" />
            )}
          </button>
        ) : null}
      </div>
    </article>
  );
}

export function ChatPage() {
  const [sessions, setSessions] = useState<ChatSession[]>(() => {
    const loaded = loadSessions();
    return loaded.length ? loaded : [createSession()];
  });
  const [activeId, setActiveId] = useState(() => sessions[0].id);
  const [prompt, setPrompt] = useState("");
  const [reasoningEffort, setReasoningEffort] = useState("high");
  const [busy, setBusy] = useState(false);
  const [phase, setPhase] = useState("等待输入");
  const [recall, setRecall] = useState<RecallTrace | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const responseIdRef = useRef<string | null>(null);
  const terminalResponseIdRef = useRef<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const active = useMemo(
    () => sessions.find((session) => session.id === activeId) || sessions[0],
    [activeId, sessions],
  );

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: busy ? "auto" : "smooth",
      block: "end",
    });
  }, [active?.messages, busy]);

  const commit = (mutate: (draft: ChatSession[]) => void) => {
    setSessions((current) => {
      const next = current.map((session) => ({
        ...session,
        messages: [...session.messages],
      }));
      mutate(next);
      next.sort((left, right) => right.updatedAt.localeCompare(left.updatedAt));
      saveSessions(next);
      return next;
    });
  };

  const newSession = () => {
    const session = createSession();
    setSessions((current) => {
      const next = [session, ...current];
      saveSessions(next);
      return next;
    });
    setActiveId(session.id);
    setRecall(null);
  };

  const deleteSession = (sessionId: string) => {
    if (busy && sessionId === activeId) return;
    setSessions((current) => {
      let next = current.filter((session) => session.id !== sessionId);
      if (!next.length) next = [createSession()];
      if (sessionId === activeId) setActiveId(next[0].id);
      saveSessions(next);
      return next;
    });
  };

  const resetContext = () => {
    if (!active || busy) return;
    commit((draft) => {
      const session = draft.find((item) => item.id === active.id);
      if (!session) return;
      session.lastResponseId = null;
      session.updatedAt = new Date().toISOString();
    });
    toast.success(t("已断开 previous_response_id"), {
      description: t("本地消息保留，下一轮从新上下文开始。"),
    });
  };

  const loadRecallForResponse = async (responseId: string) => {
    for (const delayMs of [0, 350, 900, 1800]) {
      if (delayMs) {
        await new Promise<void>((resolve) =>
          window.setTimeout(resolve, delayMs),
        );
      }
      try {
        const payload = await getRecallTrace(100);
        const turn = payload.turns.find(
          (item) =>
            item.response_id === responseId || item.turn_id === responseId,
        );
        if (turn) {
          setRecall({ ...payload, turns: [turn] });
          return;
        }
      } catch {
        // The answer remains valid when observability is temporarily unavailable.
        return;
      }
    }
  };

  const send = async () => {
    const text = prompt.trim();
    if (!text || !active || busy) return;
    setPrompt("");
    setBusy(true);
    setPhase("准备记忆与调度");
    setRecall(null);

    const createdAt = new Date().toISOString();
    const userMessage: ChatMessage = {
      id: makeId("msg"),
      role: "user",
      content: text,
      reasoning: "",
      tools: [],
      createdAt,
      status: "completed",
    };
    const assistantMessage: ChatMessage = {
      id: makeId("msg"),
      role: "assistant",
      content: "",
      reasoning: "",
      tools: [],
      createdAt,
      status: "in_progress",
    };
    const sessionId = active.id;
    const previousResponseId = active.lastResponseId;

    commit((draft) => {
      const session = draft.find((item) => item.id === sessionId);
      if (!session) return;
      if (!session.messages.length)
        session.title = text.replace(/\s+/g, " ").slice(0, 28) || "新对话";
      session.messages.push(userMessage, assistantMessage);
      session.updatedAt = createdAt;
    });

    const updateAssistant = () => {
      commit((draft) => {
        const session = draft.find((item) => item.id === sessionId);
        if (!session) return;
        const index = session.messages.findIndex(
          (item) => item.id === assistantMessage.id,
        );
        if (index >= 0)
          session.messages[index] = {
            ...assistantMessage,
            tools: assistantMessage.tools.map((tool) => ({ ...tool })),
          };
        session.updatedAt = new Date().toISOString();
      });
    };

    const controller = new AbortController();
    abortRef.current = controller;
    responseIdRef.current = null;
    terminalResponseIdRef.current = null;
    let completed = false;
    let incomplete = false;

    try {
      const body: Record<string, unknown> = {
        model: await getModelId(),
        input: text,
        stream: true,
        store: true,
        reasoning: { effort: reasoningEffort },
        max_output_tokens: CONSOLE_MAX_OUTPUT_TOKENS,
        metadata: { client: "qwen-exo-console", local_session_id: sessionId },
      };
      if (previousResponseId) body.previous_response_id = previousResponseId;

      await streamResponse(body, controller.signal, {
        onEvent(eventType, payload) {
          if (eventType === "response.created") {
            responseIdRef.current =
              payload.response?.id || payload.id || responseIdRef.current;
            setPhase("连续批处理调度中");
          } else if (
            eventType === "response.reasoning_text.delta" ||
            eventType === "response.reasoning_summary_text.delta"
          ) {
            assistantMessage.reasoning += String(payload.delta || "");
            setPhase("模型推理中");
          } else if (eventType === "response.output_text.delta") {
            assistantMessage.content += String(payload.delta || "");
            setPhase("输出回答中");
          } else if (
            eventType === "response.output_item.added" &&
            payload.item?.type === "function_call"
          ) {
            toolFor(assistantMessage, payload.item);
            setPhase("生成工具调用");
          } else if (eventType === "response.function_call_arguments.delta") {
            toolFor(assistantMessage, {}, payload.item_id).arguments += String(
              payload.delta || "",
            );
          } else if (eventType === "response.function_call_arguments.done") {
            const tool = toolFor(assistantMessage, {}, payload.item_id);
            if (payload.arguments) tool.arguments = String(payload.arguments);
            tool.done = true;
          } else if (eventType === "response.completed") {
            completed = true;
            responseIdRef.current =
              payload.response?.id || responseIdRef.current;
          } else if (eventType === "response.incomplete") {
            completed = true;
            incomplete = true;
            responseIdRef.current =
              payload.response?.id || responseIdRef.current;
          } else if (eventType === "response.failed" || eventType === "error") {
            throw new Error(
              String(
                payload.error?.message || payload.message || t("模型生成失败"),
              ),
            );
          }
          updateAssistant();
        },
      });

      if (!completed) throw new Error(t("流式响应在完成事件前结束"));
      if (!assistantMessage.content && !assistantMessage.tools.length)
        assistantMessage.content = incomplete
          ? "本轮在返回文本前达到输出上限。"
          : "本轮没有返回可显示文本。";
      assistantMessage.status = incomplete ? "incomplete" : "completed";
      const finalResponseId = responseIdRef.current;
      commit((draft) => {
        const session = draft.find((item) => item.id === sessionId);
        if (!session) return;
        if (finalResponseId) session.lastResponseId = finalResponseId;
      });
      updateAssistant();
      setPhase(incomplete ? "达到输出上限" : "生成完成");
      if (finalResponseId) void loadRecallForResponse(finalResponseId);
    } catch (error) {
      if (controller.signal.aborted) {
        const terminalOnServer =
          Boolean(responseIdRef.current) &&
          terminalResponseIdRef.current === responseIdRef.current;
        if (terminalOnServer) {
          assistantMessage.status = "completed";
          assistantMessage.content ||= "本轮没有返回可显示文本。";
          setPhase("生成完成");
        } else {
          assistantMessage.status = "cancelled";
          assistantMessage.content ||= "本轮生成已由你停止。";
          setPhase("已停止");
        }
      } else {
        assistantMessage.status = "failed";
        assistantMessage.error = true;
        const message = error instanceof Error ? error.message : t("未知错误");
        assistantMessage.content = assistantMessage.content
          ? `${assistantMessage.content}\n\n${t("请求未完成：{message}", { message })}`
          : t("请求未完成：{message}", { message });
        setPhase("请求失败");
        toast.error(t("生成失败"), { description: message });
      }
      updateAssistant();
    } finally {
      if (abortRef.current === controller) {
        setBusy(false);
        abortRef.current = null;
        responseIdRef.current = null;
      }
      terminalResponseIdRef.current = null;
    }
  };

  const stop = async () => {
    const controller = abortRef.current;
    const responseId = responseIdRef.current;
    if (!responseId) {
      controller?.abort();
      return;
    }
    try {
      const response = await fetch(
        `/v1/responses/${encodeURIComponent(responseId)}/cancel`,
        { method: "POST" },
      );
      if (!response.ok) {
        const detail = await response.text();
        if (
          response.status === 400 &&
          detail.toLowerCase().includes("cannot cancel a terminal response")
        ) {
          terminalResponseIdRef.current = responseId;
          return;
        }
        throw new Error(detail || `HTTP ${response.status}`);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : t("未知错误");
      toast.error(t("停止请求失败"), { description: message });
    } finally {
      controller?.abort();
    }
  };

  return (
    <div className="h-[calc(100vh-3.5rem)] p-3 sm:p-5">
      <section className="mx-auto flex h-full max-w-6xl flex-col overflow-hidden rounded-xl border bg-card shadow-sm">
        <header className="flex h-14 shrink-0 items-center justify-between gap-3 border-b px-3 sm:px-4">
          <div className="flex min-w-0 items-center gap-3">
            <span
              className="status-dot shrink-0"
              data-state={busy ? "starting" : "ready"}
              title={t(phase)}
              aria-label={t(phase)}
            />
            <h1 className="truncate text-sm font-semibold">
              {active.title === "新对话" ? t("新对话") : active.title}
            </h1>
          </div>
          <div className="flex shrink-0 items-center gap-1">
            {recall?.turns?.length ? (
              <span
                className="mr-1 h-2 w-2 rounded-full bg-emerald-500"
                title={t("本轮有召回记录")}
                aria-label={t("本轮有召回记录")}
              />
            ) : null}
            <Select
              value={reasoningEffort}
              onValueChange={setReasoningEffort}
              disabled={busy}
            >
              <SelectTrigger
                className="h-8 w-24 text-xs"
                aria-label={t("推理强度")}
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="none">{t("关闭")}</SelectItem>
                <SelectItem value="low">{t("轻量")}</SelectItem>
                <SelectItem value="medium">{t("标准")}</SelectItem>
                <SelectItem value="high">{t("深入")}</SelectItem>
              </SelectContent>
            </Select>
            <Button
              variant="ghost"
              size="icon"
              onClick={resetContext}
              disabled={busy || !active.lastResponseId}
              title={t("断开上下文")}
            >
              <RotateCcw />
              <span className="sr-only">{t("断开上下文")}</span>
            </Button>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="icon" title={t("对话记录")}>
                  <History />
                  <span className="sr-only">{t("对话记录")}</span>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-64">
                {sessions.map((session) => (
                  <DropdownMenuItem
                    key={session.id}
                    onSelect={() => {
                      setActiveId(session.id);
                      setRecall(null);
                    }}
                    className={cn(
                      "truncate",
                      session.id === active.id && "bg-muted font-medium",
                    )}
                  >
                    <span className="truncate">
                      {session.title === "新对话" ? t("新对话") : session.title}
                    </span>
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
            <Button
              variant="ghost"
              size="icon"
              onClick={newSession}
              title={t("新对话")}
            >
              <MessageSquarePlus />
              <span className="sr-only">{t("新对话")}</span>
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => deleteSession(active.id)}
              disabled={busy}
              title={t("删除当前对话")}
            >
              <Trash2 />
              <span className="sr-only">{t("删除当前对话")}</span>
            </Button>
          </div>
        </header>

        <ScrollArea className="min-h-0 flex-1 bg-card">
          <div className="mx-auto flex min-h-full max-w-4xl flex-col px-4 py-7 sm:px-8">
            {!active.messages.length ? (
              <div className="flex flex-1 items-center justify-center text-sm text-muted-foreground">
                {t("输入消息开始")}
              </div>
            ) : (
              <div className="space-y-7">
                {active.messages.map((message) => (
                  <MessageCard key={message.id} message={message} />
                ))}
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </ScrollArea>

        <footer className="shrink-0 border-t bg-card p-3 sm:p-4">
          <div className="chat-composer mx-auto max-w-4xl rounded-xl border bg-muted/30 p-2 transition-shadow">
            <Textarea
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void send();
                }
              }}
              placeholder={t("输入消息…")}
              className="min-h-[76px] resize-none border-0 bg-transparent p-2 shadow-none focus-visible:ring-0"
              disabled={busy}
            />
            <div className="flex justify-end px-1 pt-1">
              {busy ? (
                <Button variant="outline" size="sm" onClick={() => void stop()}>
                  <Square className="fill-current" />
                  {t("停止")}
                </Button>
              ) : (
                <Button
                  size="sm"
                  onClick={() => void send()}
                  disabled={!prompt.trim()}
                >
                  <CornerDownLeft />
                  {t("发送")}
                </Button>
              )}
            </div>
          </div>
        </footer>
      </section>
    </div>
  );
}
