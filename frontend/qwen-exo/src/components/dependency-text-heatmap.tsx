import { useMemo, useState } from "react";
import { useI18n } from "@/lib/i18n";
import type { DependencyReport } from "@/lib/attention-diagnostic-view";

export function DependencyTextHeatmap({
  report,
}: {
  report: DependencyReport;
}) {
  const { t } = useI18n();
  const [selection, setSelection] = useState<{
    report: DependencyReport;
    indices: number[];
  } | null>(null);
  const selected = selection?.report === report ? selection.indices : [];
  const { segments, peak } = useMemo(() => {
    const characters = Array.from(report.rendered_prompt);
    const ranges = report.blocks.flatMap((block, index) => {
      const tokens = report.tokens.slice(block.token_start, block.token_end);
      if (!tokens.length || !Number.isFinite(block.delta)) return [];
      const start = Math.min(...tokens.map((token) => token.start));
      const end = Math.max(...tokens.map((token) => token.end));
      if (
        !Number.isSafeInteger(start) ||
        !Number.isSafeInteger(end) ||
        start < 0 ||
        end <= start ||
        end > characters.length
      )
        return [];
      return [{ start, end, index }];
    });
    const boundaries = [
      ...new Set([
        0,
        characters.length,
        ...ranges.flatMap(({ start, end }) => [start, end]),
      ]),
    ].sort((a, b) => a - b);
    return {
      peak: report.blocks.reduce(
        (max, block) =>
          Number.isFinite(block.delta)
            ? Math.max(max, Math.abs(block.delta))
            : max,
        0,
      ),
      segments: boundaries.slice(0, -1).map((start, i) => {
        const end = boundaries[i + 1];
        return {
          start,
          text: characters.slice(start, end).join(""),
          indices: ranges
            .filter((range) => range.start <= start && range.end >= end)
            .map((range) => range.index),
        };
      }),
    };
  }, [report]);
  const signed = (value: number) =>
    `${value > 0 ? "+" : ""}${value.toFixed(5)}`;

  return (
    <div className="min-w-0 overflow-hidden rounded-xl border bg-background">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b bg-muted/20 px-4 py-3 sm:px-6">
        <div>
          <h3 className="text-sm font-semibold">{t("连续文本热力图")}</h3>
          <p className="mt-1 text-xs text-muted-foreground">
            {t(
              "保留原文顺序与换行。点击着色文字查看 Δ；颜色表示整块影响，不是逐字分数。",
            )}
          </p>
        </div>
        <div
          className="flex items-center gap-2 text-xs text-muted-foreground"
          aria-label={t("色阶：负 Δ 到正 Δ")}
        >
          <span>−{peak.toFixed(2)}</span>
          <span
            className="h-2 w-28 rounded-full"
            style={{
              background:
                "linear-gradient(90deg,rgba(14,165,233,.55),rgba(148,163,184,.10),rgba(249,115,22,.55))",
            }}
          />
          <span>+{peak.toFixed(2)}</span>
        </div>
      </div>
      <div className="flex flex-wrap gap-x-5 gap-y-2 px-4 pt-4 text-xs text-muted-foreground sm:px-6">
        <span>{t("暖色：移除后概率下降")}</span>
        <span>{t("冷色：移除后概率上升")}</span>
        <span>{t("灰底：已测 Δ = 0；无底色：未测；斜线：跨块字符")}</span>
      </div>
      <div
        data-dependency-text
        className="max-h-[65vh] overflow-y-auto whitespace-pre-wrap break-words px-4 py-5 font-mono text-sm leading-8 [overflow-wrap:anywhere] sm:px-6 sm:py-6"
      >
        {segments.map(({ start, text, indices }) => {
          if (!indices.length)
            return (
              <span key={start} title={t("未测区域")}>
                {text}
              </span>
            );
          const block = report.blocks[indices[0]];
          const overlap = indices.length > 1;
          const alpha = peak
            ? 0.07 + 0.43 * Math.sqrt(Math.abs(block.delta) / peak)
            : 0.1;
          const color =
            block.delta > 0
              ? "249,115,22"
              : block.delta < 0
                ? "14,165,233"
                : "148,163,184";
          const active = indices.some((index) => selected.includes(index));
          const label = overlap
            ? t("跨块字符：点击分别查看，不合并分数")
            : `tokens [${block.token_start}, ${block.token_end}) · Δ ${signed(block.delta)}`;
          return (
            <span
              key={start}
              role="button"
              tabIndex={0}
              aria-label={label}
              aria-pressed={active}
              title={label}
              onClick={() => setSelection({ report, indices })}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  setSelection({ report, indices });
                }
              }}
              className="cursor-pointer rounded-[2px] outline-offset-2 transition-colors hover:underline focus-visible:outline focus-visible:outline-2 focus-visible:outline-ring"
              style={{
                backgroundColor: overlap
                  ? "rgba(148,163,184,.12)"
                  : `rgba(${color},${alpha})`,
                backgroundImage: overlap
                  ? "repeating-linear-gradient(135deg,transparent,transparent 3px,rgba(148,163,184,.25) 3px,rgba(148,163,184,.25) 4px)"
                  : undefined,
                boxShadow: active ? "inset 0 -2px 0 currentColor" : undefined,
                boxDecorationBreak: "clone",
                WebkitBoxDecorationBreak: "clone",
              }}
            >
              {text}
            </span>
          );
        })}
      </div>
      <div
        aria-live="polite"
        data-dependency-detail
        className="min-h-24 space-y-3 border-t bg-muted/20 px-4 py-4 sm:px-6"
      >
        {!selected.length ? (
          <p className="text-sm text-muted-foreground">
            {t("点击上方文字，在这里查看对应块的分数。")}
          </p>
        ) : (
          selected.map((index) => {
            const block = report.blocks[index];
            return (
              <div
                key={index}
                className="flex flex-wrap items-center gap-x-8 gap-y-3 text-xs"
              >
                <div>
                  <p className="text-muted-foreground">
                    {t("所选范围（右端不含）")}
                  </p>
                  <p className="mt-1 font-mono">
                    tokens [{block.token_start}, {block.token_end})
                  </p>
                </div>
                <div>
                  <p className="text-muted-foreground">{t("基线")}</p>
                  <p className="mt-1 font-mono">
                    {report.base_logprob.toFixed(5)}
                  </p>
                </div>
                <div>
                  <p className="text-muted-foreground">{t("移除后")}</p>
                  <p className="mt-1 font-mono">
                    {block.ablated_logprob.toFixed(5)}
                  </p>
                </div>
                <div>
                  <p className="text-muted-foreground">Δ log P</p>
                  <p className="mt-1 font-mono text-lg font-semibold">
                    {signed(block.delta)}
                  </p>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
