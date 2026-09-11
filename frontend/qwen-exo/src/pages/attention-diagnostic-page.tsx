import { useEffect, useMemo, useRef, useState } from "react";
import { LoaderCircle, Play, RefreshCw, Square, Upload } from "lucide-react";
import { PageHeader } from "@/components/page-header";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { apiFetch, ApiError } from "@/lib/api";
import { loadSessions } from "@/lib/session-store";
import type { ChatSession } from "@/lib/types";
import { useI18n } from "@/lib/i18n";
import { AttentionBlockHeatmap } from "@/components/attention-block-heatmap";
import {
  classifyAttentionTokens,
  visibleTokenText,
  type AttentionMessage,
  type AttentionReport,
  type AttributionCategory,
} from "@/lib/attention-diagnostic-view";

type Conversation = {
  id: string;
  title: string;
  updated_at: number;
  message_count: number;
  source: string;
  partial: boolean;
};
type ConversationPage = {
  conversations: Conversation[];
  total: number;
  limit: number;
  offset: number;
};
type ConversationDetail = {
  content: string;
  filename: string;
  warnings: string[];
};
type TokenBudget = {
  end_message: number;
  prompt_tokens: number;
  max_prompt_tokens: number;
};
type Preview = {
  messages: AttentionMessage[];
  format: string;
  warnings: string[];
  token_budget: TokenBudget;
  available_layer_ids: number[];
  default_layer_ids: number[];
  max_layers: number;
};
const MAX_BYTES = 2 * 1024 * 1024;

export function AttentionDiagnosticPage() {
  const { t } = useI18n();
  const [content, setContent] = useState("");
  const [filename, setFilename] = useState("");
  const [preview, setPreview] = useState<Preview | null>(null);
  const [endMessage, setEndMessage] = useState(1);
  const [tokenBudget, setTokenBudget] = useState<TokenBudget | null>(null);
  const [endToken, setEndToken] = useState<string | null>(null);
  const [sampleCount, setSampleCount] = useState(1);
  const [selectedLayers, setSelectedLayers] = useState<number[] | null>(null);
  const [report, setReport] = useState<AttentionReport | null>(null);
  const [sampleIndex, setSampleIndex] = useState(0);
  const [layerIndex, setLayerIndex] = useState(0);
  const [page, setPage] = useState(0);
  const [busy, setBusy] = useState<"preview" | "run" | null>(null);
  const [error, setError] = useState("");
  const controller = useRef<AbortController | null>(null);
  const fileVersion = useRef(0);
  const [serverConversations, setServerConversations] = useState<
    Conversation[]
  >([]);
  const [serverTotal, setServerTotal] = useState(0);
  const [serverOffset, setServerOffset] = useState(0);
  const [selectedServer, setSelectedServer] = useState("");
  const [selectedLocal, setSelectedLocal] = useState("");
  const [localSessions, setLocalSessions] = useState<ChatSession[]>([]);
  const [listBusy, setListBusy] = useState(false);
  const [listError, setListError] = useState("");
  const [importBusy, setImportBusy] = useState(false);
  const [sourceWarnings, setSourceWarnings] = useState<string[]>([]);
  const listController = useRef<AbortController | null>(null);
  const importController = useRef<AbortController | null>(null);
  useEffect(
    () => () => {
      controller.current?.abort();
      listController.current?.abort();
      importController.current?.abort();
      fileVersion.current++;
    },
    [],
  );

  useEffect(() => {
    setLocalSessions(loadSessions());
    void loadConversations(0);
  }, []);

  async function loadConversations(offset: number) {
    listController.current?.abort();
    const current = new AbortController();
    listController.current = current;
    setListBusy(true);
    setListError("");
    try {
      const response = await apiFetch(
        `/attention-diagnostics/conversations?limit=25&offset=${offset}`,
        { signal: current.signal },
      );
      const data = (await response.json()) as ConversationPage;
      if (current.signal.aborted) return;
      setServerConversations(data.conversations);
      setServerTotal(data.total);
      setServerOffset(data.offset);
      setSelectedServer("");
    } catch (cause) {
      if (!current.signal.aborted)
        setListError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      if (listController.current === current) {
        listController.current = null;
        setListBusy(false);
      }
    }
  }

  function stopRequests() {
    controller.current?.abort();
    importController.current?.abort();
    controller.current = null;
    importController.current = null;
    fileVersion.current++;
    setBusy(null);
    setImportBusy(false);
  }

  async function importConversation(kind: "server" | "local", id: string) {
    stopRequests();
    if (!id) return;
    setError("");
    setImportBusy(true);
    const current = new AbortController();
    importController.current = current;
    try {
      let detail: ConversationDetail;
      if (kind === "local") {
        const session = localSessions.find((item) => item.id === id);
        if (!session) throw new Error(t("本地会话已不存在"));
        const messages = session.messages.map((message) => ({
          role: message.role,
          content: [
            message.reasoning ? `<think>\n${message.reasoning}\n</think>` : "",
            message.content,
            message.tools?.length
              ? `[Browser tool calls: inert recorded data]\n${JSON.stringify(message.tools)}`
              : "",
          ]
            .filter(Boolean)
            .join("\n\n"),
        }));
        detail = {
          content: JSON.stringify({ messages }),
          filename: `browser-session-${session.id}.json`,
          warnings: [
            t(
              "浏览器会话最多保留 160 条本地消息，不含系统提示或工具结果来源。推理与工具调用按记录投影为惰性文本，可能不完整。",
            ),
          ],
        };
        if (
          session.messages.some((message) => message.status !== "completed")
        ) {
          detail.warnings.push(
            t("此会话含未完成、取消或失败的消息，仅导入已保留的内容。"),
          );
        }
      } else {
        const response = await apiFetch(
          `/attention-diagnostics/conversations/${encodeURIComponent(id)}`,
          { signal: current.signal },
        );
        detail = (await response.json()) as ConversationDetail;
      }
      if (current.signal.aborted) return;
      if (new TextEncoder().encode(detail.content).length > MAX_BYTES)
        throw new Error(t("文件不能超过 2 MiB"));
      const response = await apiFetch("/attention-diagnostics/preview", {
        method: "POST",
        signal: current.signal,
        body: JSON.stringify({
          content: detail.content,
          filename: detail.filename,
        }),
      });
      const data = (await response.json()) as Preview;
      if (current.signal.aborted) return;
      const nextEndMessage = data.token_budget.end_message;
      setContent(detail.content);
      setFilename(detail.filename);
      setSourceWarnings(detail.warnings || []);
      setPreview(data);
      setSelectedLayers(null);
      setEndMessage(nextEndMessage);
      setTokenBudget(data.token_budget);
      setEndToken(null);
      setReport(null);
    } catch (cause) {
      if (!current.signal.aborted)
        setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      if (importController.current === current) {
        importController.current = null;
        setImportBusy(false);
      }
    }
  }

  function invalidate() {
    stopRequests();
    setError("");
    setReport(null);
  }
  function replaceContent(next: string, name = "") {
    invalidate();
    setContent(next);
    setFilename(name);
    setPreview(null);
    setSelectedLayers(null);
    setTokenBudget(null);
    setEndToken(null);
    setSourceWarnings([]);
    setSelectedServer("");
    setSelectedLocal("");
  }
  async function loadFile(file?: File) {
    if (!file) return;
    stopRequests();
    const version = fileVersion.current;
    if (file.size > MAX_BYTES) {
      setError(t("文件不能超过 2 MiB"));
      return;
    }
    try {
      const text = await file.text();
      if (fileVersion.current === version) replaceContent(text, file.name);
    } catch (cause) {
      if (fileVersion.current === version) setError(String(cause));
    }
  }
  async function request(mode: "preview" | "run", selectedEndMessage?: number) {
    if (mode === "run" && !canRun) return;
    if (new TextEncoder().encode(content).length > MAX_BYTES) {
      setError(t("文件不能超过 2 MiB"));
      return;
    }
    invalidate();
    if (mode === "preview") {
      setTokenBudget(null);
      setEndToken(null);
      if (selectedEndMessage !== undefined) setEndMessage(selectedEndMessage);
    }
    const current = new AbortController();
    controller.current = current;
    setBusy(mode);
    try {
      const response = await apiFetch(`/attention-diagnostics/${mode}`, {
        method: "POST",
        signal: current.signal,
        body: JSON.stringify({
          content,
          filename: filename || undefined,
          ...(mode === "run"
            ? {
                end_message: endMessage,
                sample_count: sampleCount,
                layer_ids: layerIds,
                ...(endToken !== null ? { end_token: Number(endToken) } : {}),
              }
            : { end_message: selectedEndMessage }),
        }),
      });
      const data = await response.json();
      if (current.signal.aborted) return;
      if (mode === "preview") {
        const nextEndMessage = data.token_budget.end_message;
        setPreview(data as Preview);
        setEndMessage(nextEndMessage);
        setTokenBudget(data.token_budget);
      } else {
        setReport(data as AttentionReport);
        setSampleIndex(0);
        setLayerIndex(0);
        setPage(0);
      }
    } catch (cause) {
      if (!current.signal.aborted)
        setError(
          cause instanceof ApiError && cause.status === 404
            ? t("当前后端未提供注意力诊断接口；需部署支持版本。")
            : cause instanceof Error
              ? cause.message
              : String(cause),
        );
    } finally {
      if (controller.current === current) {
        controller.current = null;
        setBusy(null);
      }
    }
  }
  const sample = report?.samples[sampleIndex];
  const layer = sample?.layers[layerIndex];
  const weights = layer?.weights ?? [];
  const promptCharacters = useMemo(
    () => Array.from(report?.rendered_prompt ?? ""),
    [report],
  );
  const heatUnits = useMemo(() => {
    const units: {
      start: number;
      end: number;
      first: number;
      last: number;
      weight: number;
      sampled: boolean;
    }[] = [];
    for (const [index, token] of (report?.tokens ?? []).entries()) {
      const previous = units[units.length - 1];
      if (previous && token.start < previous.end && token.end > token.start) {
        previous.end = Math.max(previous.end, token.end);
        previous.last = index;
        previous.weight += weights[index] ?? 0;
        previous.sampled ||= weights[index] !== undefined;
      } else
        units.push({
          start: token.start,
          end: token.end,
          first: index,
          last: index,
          weight: weights[index] ?? 0,
          sampled: weights[index] !== undefined,
        });
    }
    return units;
  }, [report, weights]);
  const peak = useMemo(
    () =>
      heatUnits.reduce((maximum, unit) => Math.max(maximum, unit.weight), 0),
    [heatUnits],
  );
  const visibleUnits = heatUnits.filter(
    (unit) => unit.last >= page * 256 && unit.first < (page + 1) * 256,
  );
  const attribution = useMemo(
    () =>
      report ? classifyAttentionTokens(report.tokens, report.messages) : [],
    [report],
  );
  const distribution = useMemo(() => {
    const categories: AttributionCategory[] = [
      "source",
      "marker",
      "whitespace",
      "unassigned",
      "boundary",
      "unmapped",
    ];
    const rows = categories.map((category) => ({
      category,
      mass: 0,
      count: 0,
    }));
    const messages = (report?.messages ?? []).map((message) => ({
      message,
      mass: 0,
      count: 0,
    }));
    let total = 0;
    weights.forEach((weight, index) => {
      const owner = attribution[index];
      const row =
        rows.find((row) => row.category === owner?.category) ?? rows[5];
      row.mass += weight;
      row.count++;
      total += weight;
      if (owner?.category === "source" && owner.messageIndex !== null) {
        messages[owner.messageIndex].mass += weight;
        messages[owner.messageIndex].count++;
      }
    });
    return { rows, messages, total };
  }, [report, weights, attribution]);
  const categoryLabels: Record<AttributionCategory, string> = {
    source: t("原始消息正文"),
    marker: t("正文外的边界标记文本"),
    whitespace: t("正文外的空白字符"),
    unassigned: t("正文外的其他文本（来源未确认）"),
    boundary: t("跨正文边界或归属重叠"),
    unmapped: t("无有效字符范围"),
  };
  const topTokens = useMemo(
    () =>
      weights
        .map((weight, index) => ({ weight, index }))
        .sort((a, b) => b.weight - a.weight)
        .slice(0, 20),
    [weights],
  );
  const budget = tokenBudget?.end_message === endMessage ? tokenBudget : null;
  const tokenLimit = budget
    ? Math.min(budget.prompt_tokens, budget.max_prompt_tokens)
    : 0;
  const selectedTokens =
    endToken === null ? (budget?.prompt_tokens ?? 0) : Number(endToken);
  const validTokenPrefix =
    !!budget &&
    Number.isInteger(selectedTokens) &&
    selectedTokens >= 1 &&
    selectedTokens <= budget.prompt_tokens;
  const layerIds = selectedLayers ?? preview?.default_layer_ids ?? [];
  const validLayers =
    !!preview &&
    layerIds.length > 0 &&
    layerIds.length <= preview.max_layers &&
    layerIds.every((id) => preview.available_layer_ids.includes(id));
  const canRun =
    validTokenPrefix &&
    validLayers &&
    selectedTokens <= tokenLimit &&
    busy === null &&
    !importBusy;
  const selectClass = "h-9 rounded-md border bg-background px-2 text-sm";

  return (
    <div className="page-frame space-y-6">
      <PageHeader
        title={t("注意力诊断")}
        description={t(
          "上传对话，裁剪历史，观察指定输入位置的 Full Attention 分布。仅诊断，不执行工具。",
        )}
      />
      <div className="rounded-lg border bg-muted/30 p-4 text-sm text-muted-foreground">
        {t(
          "重新计算裁剪后的对话，不还原历史请求状态。不注入记忆、不训练、不修改注意力。采样不覆盖 GDN，也不代表理解程度或因果归因。",
        )}
      </div>
      <section className="space-y-3 rounded-lg border p-4 sm:p-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="font-semibold">{t("对话源")}</h2>
          <label className="flex cursor-pointer items-center gap-2 text-sm">
            <Upload className="h-4 w-4" />
            {t("上传文件")}
            <input
              type="file"
              className="sr-only"
              aria-label={t("上传文件")}
              accept=".json,.txt,.chatml"
              onChange={(event) => {
                void loadFile(event.target.files?.[0]);
                event.target.value = "";
              }}
            />
          </label>
        </div>
        <div className="space-y-3 rounded-md border bg-muted/20 p-3">
          <p className="text-xs text-muted-foreground">
            {t(
              "选择会话后自动导入并预览，不会运行模型。优先使用服务器保留的原始记录；记录可能不完整。",
            )}
          </p>
          <div className="flex flex-wrap items-end gap-3">
            <label className="flex min-w-0 flex-1 basis-64 flex-col gap-1 text-sm">
              <span>{t("保留的服务器会话")}</span>
              <select
                className={`${selectClass} w-full`}
                value={selectedServer}
                disabled={listBusy}
                onChange={(event) => {
                  setSelectedServer(event.target.value);
                  setSelectedLocal("");
                  void importConversation("server", event.target.value);
                }}
              >
                <option value="">
                  {listBusy ? t("正在刷新") : t("选择会话")}
                </option>
                {serverConversations.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.title || item.id} · {item.source} ·{" "}
                    {item.message_count} {t("条消息")} ·{" "}
                    {item.updated_at
                      ? new Date(item.updated_at * 1000).toLocaleString()
                      : "—"}
                    {item.partial ? ` (${t("不完整")})` : ""}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex min-w-0 flex-1 basis-64 flex-col gap-1 text-sm">
              <span>
                {t("浏览器会话")} · {t("不完整")}
              </span>
              <select
                className={`${selectClass} w-full`}
                value={selectedLocal}
                onChange={(event) => {
                  setSelectedLocal(event.target.value);
                  setSelectedServer("");
                  void importConversation("local", event.target.value);
                }}
              >
                <option value="">{t("选择本地会话")}</option>
                {localSessions.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.title || item.id} · {item.messages.length}{" "}
                    {t("条消息")} · {item.updatedAt}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <p className="text-xs text-muted-foreground">
            {t(
              "浏览器会话最多保留 160 条本地消息，不含系统提示或工具结果来源。推理与工具调用按记录投影为惰性文本，可能不完整。",
            )}
          </p>
          <div className="flex flex-wrap items-center gap-2">
            <Button
              variant="outline"
              disabled={listBusy}
              onClick={() => {
                setLocalSessions(loadSessions());
                void loadConversations(serverOffset);
              }}
            >
              <RefreshCw className="h-4 w-4" />
              {t("刷新")}
            </Button>
            <Button
              variant="outline"
              disabled={listBusy || serverOffset === 0}
              onClick={() =>
                void loadConversations(Math.max(0, serverOffset - 25))
              }
            >
              {t("上一页")}
            </Button>
            <Button
              variant="outline"
              disabled={listBusy || serverOffset + 25 >= serverTotal}
              onClick={() => void loadConversations(serverOffset + 25)}
            >
              {t("下一页")}
            </Button>
            <span className="text-xs text-muted-foreground">
              {serverTotal ? serverOffset + 1 : 0}–
              {serverOffset + serverConversations.length} / {serverTotal}
            </span>
            {importBusy && (
              <Button variant="outline" onClick={stopRequests}>
                <LoaderCircle className="h-4 w-4 animate-spin" />
                {t("取消")}
              </Button>
            )}
          </div>
          {listError && (
            <p role="alert" className="text-xs text-destructive">
              {listError}
            </p>
          )}
          {sourceWarnings.length > 0 && (
            <div className="space-y-1 text-xs text-muted-foreground">
              <p className="font-medium">{t("来源说明（不影响运行）")}</p>
              {sourceWarnings.map((warning, index) => (
                <p key={index}>{warning}</p>
              ))}
            </div>
          )}
        </div>
        <p className="text-xs text-muted-foreground">
          {t(
            "支持 ChatML 文本、messages JSON、Responses input 和 Completions prompt。仅文本，最大 2 MiB。上传与预览不会调用模型。",
          )}
        </p>
        {filename && (
          <div className="text-xs font-mono break-all">{filename}</div>
        )}
        <textarea
          aria-label={t("对话内容")}
          className="min-h-48 w-full rounded-md border bg-background p-3 font-mono text-xs"
          value={content}
          onChange={(event) => {
            fileVersion.current++;
            replaceContent(event.target.value);
          }}
          spellCheck={false}
        />
        <Button
          variant="outline"
          disabled={!content.trim() || busy !== null || importBusy}
          onClick={() =>
            void request("preview", preview ? endMessage : undefined)
          }
        >
          {busy === "preview" && (
            <LoaderCircle className="h-4 w-4 animate-spin" />
          )}
          {t("解析预览")}
        </Button>
      </section>
      {preview && (
        <section className="space-y-4 rounded-lg border p-4 sm:p-6">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 className="font-semibold">{t("裁剪与采样")}</h2>
            <span className="text-xs font-mono">
              {preview.format} · {preview.messages.length} {t("条消息")}
            </span>
          </div>
          <div
            className="space-y-3 rounded-md border bg-muted/30 p-4"
            aria-live="polite"
          >
            {budget ? (
              <>
                <p className="text-sm font-semibold">
                  {t("实际渲染：{count} tokens · 诊断上限：{limit} tokens", {
                    count: budget.prompt_tokens,
                    limit: budget.max_prompt_tokens,
                  })}
                </p>
                <p className="text-sm">
                  {validTokenPrefix
                    ? t(
                        "当前消息范围内：选中 {selected} tokens，排除后缀 {excluded} tokens。",
                        {
                          selected: selectedTokens,
                          excluded: budget.prompt_tokens - selectedTokens,
                        },
                      )
                    : t("请输入有效的 token 前缀长度以确定选中与排除的数量。")}
                </p>
                {budget.prompt_tokens > budget.max_prompt_tokens && (
                  <p className="text-sm text-amber-700 dark:text-amber-300">
                    {t(
                      "当前消息范围超出诊断上限。请明确选择 token 前缀，或减少消息范围；不会自动截断。",
                    )}
                  </p>
                )}
                <div className="flex flex-wrap items-end gap-3">
                  <label className="flex flex-col gap-2 text-sm">
                    {t("从开头保留的 token 数")}
                    <Input
                      type="number"
                      aria-label={t("从开头保留的 token 数")}
                      className="w-40"
                      min={1}
                      max={tokenLimit}
                      step={1}
                      value={endToken ?? budget.prompt_tokens}
                      disabled={!!busy || importBusy}
                      onChange={(event) => {
                        invalidate();
                        setEndToken(event.target.value);
                      }}
                    />
                  </label>
                  {budget.prompt_tokens > budget.max_prompt_tokens && (
                    <Button
                      variant="outline"
                      disabled={!!busy || importBusy}
                      onClick={() => {
                        invalidate();
                        setEndToken(String(tokenLimit));
                      }}
                    >
                      {t("使用前 {limit} tokens", { limit: tokenLimit })}
                    </Button>
                  )}
                  {endToken !== null && (
                    <Button
                      variant="outline"
                      disabled={!!busy || importBusy}
                      onClick={() => {
                        invalidate();
                        setEndToken(null);
                      }}
                    >
                      {t("使用整个消息范围")}
                    </Button>
                  )}
                </div>
                {endToken !== null &&
                  (!validTokenPrefix || selectedTokens > tokenLimit) && (
                    <p className="text-xs text-destructive">
                      {t("token 前缀长度必须是 1 到 {limit} 之间的整数。", {
                        limit: tokenLimit,
                      })}
                    </p>
                  )}
                {validTokenPrefix && selectedTokens < budget.prompt_tokens && (
                  <p className="text-sm text-amber-700 dark:text-amber-300">
                    {t(
                      "仅诊断选中的精确 token 前缀，并非完整对话；后缀不会发送，可能在消息内部结束，不补齐消息或添加结束标记。",
                    )}
                  </p>
                )}
                {endMessage < preview.messages.length && (
                  <p className="text-xs text-muted-foreground">
                    {t(
                      "另有 {count} 条后续源消息不在当前范围内；上述 token 数仅对应当前消息范围。",
                      { count: preview.messages.length - endMessage },
                    )}
                  </p>
                )}
              </>
            ) : (
              <p role="status" className="text-sm">
                {busy === "preview"
                  ? t("正在计算实际渲染 token 数；完成前不能运行。")
                  : t(
                      "当前消息范围尚无有效 token 计数。请重新解析预览后运行。",
                    )}
              </p>
            )}
          </div>
          <fieldset
            className="space-y-3 rounded-md border p-4"
            disabled={!!busy || importBusy}
          >
            <legend className="px-1 text-sm font-medium">
              {t("采样层（Full Attention）")}
            </legend>
            <p className="text-xs text-muted-foreground">
              {t(
                "默认均匀选择覆盖前段到最后层的最多 {count} 个层；可自选 1–{count} 层。层号从 0 开始，不包含 GDN 层。",
                { count: preview.max_layers },
              )}
            </p>
            <div className="flex flex-wrap gap-3">
              {preview.available_layer_ids.map((id) => {
                const checked = layerIds.includes(id);
                return (
                  <label
                    key={id}
                    className="flex items-center gap-2 rounded-md border px-3 py-2 text-sm"
                  >
                    <input
                      type="checkbox"
                      checked={checked}
                      disabled={
                        !checked && layerIds.length >= preview.max_layers
                      }
                      onChange={() => {
                        invalidate();
                        setSelectedLayers(
                          checked
                            ? layerIds.filter((layer) => layer !== id)
                            : [...layerIds, id].sort((a, b) => a - b),
                        );
                      }}
                    />
                    {t("第 {layer} 层", { layer: id })}
                  </label>
                );
              })}
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <Button
                type="button"
                variant="outline"
                onClick={() => {
                  invalidate();
                  setSelectedLayers(null);
                }}
              >
                {t("恢复代表层")}
              </Button>
              <span className="text-xs text-muted-foreground">
                {t("已选 {count} / {limit} 层", {
                  count: layerIds.length,
                  limit: preview.max_layers,
                })}
              </span>
              {!validLayers && (
                <span role="alert" className="text-xs text-destructive">
                  {t("请选择至少一个有效的 Full Attention 层。")}
                </span>
              )}
            </div>
            <p className="text-xs text-muted-foreground">
              {t(
                "更多层会增加采样计算与结果传输量。结果逐层查看，不跨层平均；切层后色阶按当前层计算，请用原始百分比对照。",
              )}
            </p>
          </fieldset>
          <div className="flex flex-wrap items-end gap-4">
            <label className="flex flex-col gap-2 text-sm">
              {t("保留至消息（包含）")}
              <select
                aria-label={t("保留至消息（包含）")}
                className={selectClass}
                value={endMessage}
                disabled={busy === "run" || importBusy}
                onChange={(event) =>
                  void request("preview", Number(event.target.value))
                }
              >
                {preview.messages.map((message, index) => (
                  <option key={index} value={index + 1}>
                    #{index + 1} {message.role}{" "}
                    {message.name ?? message.tool_call_id ?? ""}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex flex-col gap-2 text-sm">
              {t("采样位置数")}
              <Input
                type="number"
                aria-label={t("采样位置数")}
                className="w-24"
                min={1}
                max={4}
                value={sampleCount}
                disabled={!!busy || importBusy}
                onChange={(event) => {
                  invalidate();
                  setSampleCount(
                    Math.max(1, Math.min(4, Number(event.target.value) || 1)),
                  );
                }}
              />
            </label>
            <Button disabled={!canRun} onClick={() => void request("run")}>
              <Play className="h-4 w-4" />
              {t("运行诊断")}
            </Button>
            {busy && (
              <Button variant="outline" onClick={invalidate}>
                <Square className="h-4 w-4" />
                {t("取消")}
              </Button>
            )}
          </div>
          <p className="text-xs text-muted-foreground">
            {t(
              "从开头保留到选中消息，之后的消息不会发送。一次采样观察末尾输入位置；多个采样观察末段不同位置，不生成或执行工具。",
            )}
          </p>
          <p className="text-xs font-medium text-muted-foreground">
            {t("源消息预览（未按 token 裁剪；实际发送范围以上方计数为准）")}
          </p>
          <div className="max-h-80 space-y-2 overflow-auto">
            {preview.messages.map((message, index) => (
              <details
                key={index}
                className={`rounded-md border p-3 ${index >= endMessage ? "opacity-40" : ""}`}
              >
                <summary className="cursor-pointer text-sm">
                  #{index + 1} · {message.role} {message.name ?? ""}{" "}
                  {message.tool_call_id ?? ""}{" "}
                  {index >= endMessage ? t("（已裁剪）") : ""}
                </summary>
                <pre className="mt-3 max-h-64 overflow-auto whitespace-pre-wrap break-words font-mono text-xs">
                  {message.content}
                </pre>
              </details>
            ))}
          </div>
          {preview.warnings.map((warning, index) => (
            <p key={index} className="text-xs text-muted-foreground">
              {warning}
            </p>
          ))}
        </section>
      )}
      {busy === "run" && (
        <div role="status" className="flex items-center gap-2 text-sm">
          <LoaderCircle className="h-4 w-4 animate-spin" />
          {t("模型正在计算采样；可取消，不会保存对话。")}
        </div>
      )}
      {error && (
        <div
          role="alert"
          className="rounded-lg border border-destructive/40 p-4 text-sm text-destructive"
        >
          {error}
        </div>
      )}
      {report && sample && layer && (
        <section className="space-y-4 rounded-lg border p-4 sm:p-6">
          <h2 className="font-semibold">{t("文本注意力热图")}</h2>
          <p className="break-all text-xs text-muted-foreground">
            {report.model} · {report.prompt_tokens} tokens · {report.method}
          </p>
          <div className="flex flex-wrap gap-4">
            <label className="flex items-center gap-2 text-sm">
              {t("查询位置")}
              <select
                className={selectClass}
                value={sampleIndex}
                onChange={(event) => {
                  setSampleIndex(Number(event.target.value));
                  setPage(0);
                }}
              >
                {report.samples.map((item, index) => (
                  <option key={index} value={index}>
                    token {item.query_position} ·{" "}
                    {visibleTokenText(item.query_text)}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex items-center gap-2 text-sm">
              {t("观测层")}
              <select
                aria-label={t("观测层")}
                className={selectClass}
                value={layerIndex}
                onChange={(event) => setLayerIndex(Number(event.target.value))}
              >
                {sample.layers.map((item, index) => (
                  <option key={item.layer_id} value={index}>
                    {item.layer_id}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div
            className="space-y-2 rounded-md border bg-muted/30 p-4 text-xs"
            aria-label={t("观测范围与守恒检查")}
          >
            <p>
              {t(
                "当前仅观测第 {layer} 层、查询 token #{query} 的 Full Attention，所有查询头平均；不是模型整体关注度。",
                { layer: layer.layer_id, query: sample.query_position },
              )}
            </p>
            <p>
              {t(
                "显示的是注意力概率，不含 Value 向量、输出投影、其他层、残差和 GDN 的贡献；不能据此判定理解程度或因果重要性。",
              )}
            </p>
            <p className="font-mono">
              {t(
                "原始权重合计：{sum}% · 相对 100% 的差值：{delta} 个百分点 · 已采样 {count} tokens",
                {
                  sum: (distribution.total * 100).toFixed(5),
                  delta: ((distribution.total - 1) * 100).toFixed(5),
                  count: weights.length,
                },
              )}
            </p>
            <p>
              {t(
                "保留全部权重，不剔除模板、不重新归一化。来源分类只依据字符范围与可见文本，不推断 token 的语义作用。",
              )}
            </p>
          </div>
          <AttentionBlockHeatmap
            report={report}
            promptCharacters={promptCharacters}
            layerId={layer.layer_id}
            onSelectToken={(index, selectedSample) => {
              setSampleIndex(selectedSample);
              setLayerIndex(
                report.samples[selectedSample].layers.findIndex(
                  (item) => item.layer_id === layer.layer_id,
                ),
              );
              setPage(Math.floor(index / 256));
            }}
          />
          <p className="text-xs text-muted-foreground">
            {t(
              "下方文本视图单独按当前查询的最大字符组权重作平方根缩放；重叠 Unicode token 合并显示。悬停查看原始占比，未进入该查询前缀的文本为灰色。",
            )}
          </p>
          <div className="flex items-center gap-3 text-xs">
            <Button
              size="sm"
              variant="outline"
              disabled={page === 0}
              onClick={() => setPage(page - 1)}
            >
              {t("上一页")}
            </Button>
            <span>
              token {page * 256}–
              {Math.min((page + 1) * 256, report.tokens.length) - 1} /{" "}
              {report.tokens.length}
            </span>
            <Button
              size="sm"
              variant="outline"
              disabled={(page + 1) * 256 >= report.tokens.length}
              onClick={() => setPage(page + 1)}
            >
              {t("下一页")}
            </Button>
          </div>
          <div
            className="space-y-2"
            aria-label={t("按来源范围分组（全部权重）")}
          >
            <h3 className="text-sm font-medium">
              {t("按来源范围分组（全部权重）")}
            </h3>
            <p className="text-xs text-muted-foreground">
              {t(
                "边界标记仅按正文外的可见拼写识别，不保证是 tokenizer 特殊 token；其他文本可能包含模板新增指令，不能笼统称为格式。跨界 token 单独保留，不按字符比例分摊。",
              )}
            </p>
            {distribution.rows.map((row) => (
              <div
                key={row.category}
                className="grid grid-cols-[minmax(0,1fr)_8rem] gap-2 text-xs"
                data-attention-category={row.category}
                data-mass={row.mass}
              >
                <div className="min-w-0">
                  <div>{categoryLabels[row.category]}</div>
                  <div className="mt-1 h-1 rounded bg-muted">
                    <div
                      className="h-1 rounded bg-amber-500"
                      style={{ width: `${row.mass * 100}%` }}
                    />
                  </div>
                </div>
                <div className="text-right font-mono">
                  {(row.mass * 100).toFixed(3)}%
                  <div className="text-muted-foreground">
                    {row.count} tokens
                  </div>
                  {row.count > 0 && (
                    <div>{((row.mass / row.count) * 100).toFixed(5)}%/tok</div>
                  )}
                </div>
              </div>
            ))}
            <details open>
              <summary className="cursor-pointer text-xs">
                {t("原始消息正文的细分（不含上述正文外权重）")}
              </summary>
              <div className="mt-2 max-h-64 space-y-3 overflow-auto">
                {distribution.messages.map((row, index) => (
                  <div
                    key={index}
                    className="grid grid-cols-[minmax(0,1fr)_9rem] gap-3 text-xs"
                    data-attention-message-row={index}
                    data-mass={row.mass}
                  >
                    <div className="min-w-0">
                      <div className="truncate">
                        #{row.message.index + 1} · {row.message.role}{" "}
                        {row.message.name ?? ""}
                        {row.message.truncated ? ` · ${t("不完整")}` : ""}
                      </div>
                      <div
                        className="mt-1 h-1.5 rounded bg-muted"
                        aria-hidden="true"
                      >
                        <div
                          className="h-1.5 rounded bg-amber-500"
                          style={{ width: `${Math.min(100, row.mass * 100)}%` }}
                        />
                      </div>
                    </div>
                    <div className="text-right font-mono">
                      {(row.mass * 100).toFixed(3)}%
                      <div className="text-muted-foreground">
                        {row.count} tokens
                      </div>
                      {row.count > 0 && (
                        <div>
                          {((row.mass / row.count) * 100).toFixed(5)}%/tok
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </details>
          </div>
          <div
            className="min-h-40 whitespace-pre-wrap break-words rounded-md border p-4 font-mono text-sm leading-7"
            id="attention-token-text"
            aria-label={t("文本注意力热图")}
          >
            {visibleUnits.map((unit) => {
              const text =
                unit.end > unit.start
                  ? promptCharacters.slice(unit.start, unit.end).join("")
                  : `[token ${report.tokens[unit.first].id}]`;
              return (
                <span
                  key={unit.first}
                  title={`token ${unit.first}–${unit.last} · ${unit.sampled ? `${(unit.weight * 100).toFixed(5)}%` : t("未采样")}`}
                  style={{
                    backgroundColor: unit.sampled
                      ? `rgba(245, 158, 11, ${peak > 0 ? 0.65 * Math.sqrt(Math.min(1, unit.weight / peak)) : 0})`
                      : undefined,
                  }}
                  className={unit.sampled ? "" : "text-muted-foreground"}
                >
                  {text}
                </span>
              );
            })}
          </div>
          <details>
            <summary className="cursor-pointer text-sm">
              {t("权重最高的 20 个 token")}
            </summary>
            <div className="mt-3 space-y-1 font-mono text-xs">
              {topTokens.map(({ index, weight }) => (
                <button
                  key={index}
                  className="block max-w-full truncate text-left hover:underline"
                  onClick={() => setPage(Math.floor(index / 256))}
                >
                  #{index} · {(weight * 100).toFixed(5)}% ·{" "}
                  {visibleTokenText(report.tokens[index]?.text ?? "")} ·{" "}
                  {categoryLabels[attribution[index]?.category ?? "unmapped"]}
                </button>
              ))}
            </div>
          </details>
          {report.warnings.map((warning, index) => (
            <p
              key={index}
              className="text-xs text-amber-600 dark:text-amber-400"
            >
              {warning}
            </p>
          ))}
        </section>
      )}
    </div>
  );
}
