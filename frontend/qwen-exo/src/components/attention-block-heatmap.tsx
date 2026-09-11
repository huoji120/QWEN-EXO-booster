import { useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  buildAttentionBlocks,
  visibleTokenText,
  type AttentionBlock,
  type AttentionReport,
} from "@/lib/attention-diagnostic-view";
import { useI18n } from "@/lib/i18n";

const COLUMNS_PER_PAGE = 32;
const ROWS_PER_PAGE = 4;
const EXCERPT_LIMIT = 2048;

type Metric = "mean" | "mass";
type HeatmapRow = {
  sampleIndex: number;
  queryPosition: number;
  queryText: string;
  available: boolean;
  blocks: AttentionBlock[];
};

function percent(value: number) {
  return `${Number((value * 100).toPrecision(6))}%`;
}

export function AttentionBlockHeatmap({
  report,
  layerId,
  promptCharacters,
  onSelectToken,
}: {
  report: AttentionReport;
  layerId: number;
  promptCharacters: readonly string[];
  onSelectToken: (index: number, sampleIndex: number) => void;
}) {
  const { t } = useI18n();
  const [blockSize, setBlockSize] = useState(64);
  const [metric, setMetric] = useState<Metric>("mean");
  const [page, setPage] = useState(0);
  const [rowPage, setRowPage] = useState(0);
  const rows = useMemo<HeatmapRow[]>(
    () =>
      report.samples.map((sample, sampleIndex) => {
        const layer = sample.layers.find((item) => item.layer_id === layerId);
        return {
          sampleIndex,
          queryPosition: sample.query_position,
          queryText: sample.query_text,
          available: !!layer,
          blocks: buildAttentionBlocks(
            report.tokens,
            layer?.weights ?? [],
            blockSize,
          ),
        };
      }),
    [report, layerId, blockSize],
  );
  const [selection, setSelection] = useState<{
    rows: HeatmapRow[];
    sampleIndex: number;
    blockIndex: number;
  } | null>(null);

  useEffect(() => {
    setPage(0);
    setRowPage(0);
    setSelection(null);
  }, [rows]);

  const blockCount = Math.ceil(report.tokens.length / blockSize);
  const pageCount = Math.max(1, Math.ceil(blockCount / COLUMNS_PER_PAGE));
  const activePage = Math.max(0, Math.min(page, pageCount - 1));
  const rowPageCount = Math.max(1, Math.ceil(rows.length / ROWS_PER_PAGE));
  const activeRowPage = Math.max(0, Math.min(rowPage, rowPageCount - 1));
  const firstBlock = activePage * COLUMNS_PER_PAGE;
  const lastBlock = Math.min(firstBlock + COLUMNS_PER_PAGE, blockCount);
  const columns = Array.from(
    { length: lastBlock - firstBlock },
    (_, index) => firstBlock + index,
  );
  const visibleRows = rows.slice(
    activeRowPage * ROWS_PER_PAGE,
    (activeRowPage + 1) * ROWS_PER_PAGE,
  );
  const globalPeak = useMemo(() => {
    let peak = 0;
    for (const row of rows) {
      if (!row.available) continue;
      for (const block of row.blocks) {
        if (block.count > 0) peak = Math.max(peak, block[metric]);
      }
    }
    return peak;
  }, [rows, metric]);
  const selectedRow =
    selection?.rows === rows ? rows[selection.sampleIndex] : undefined;
  const selectedBlock =
    selectedRow && selection
      ? selectedRow.blocks[selection.blockIndex]
      : undefined;
  const excerpt = useMemo(() => {
    if (!selectedBlock) return null;
    let start = Infinity;
    let end = -Infinity;
    for (let index = selectedBlock.start; index < selectedBlock.end; index++) {
      const token = report.tokens[index];
      if (
        !Number.isInteger(token.start) ||
        !Number.isInteger(token.end) ||
        token.start < 0 ||
        token.end <= token.start ||
        token.end > promptCharacters.length
      )
        continue;
      start = Math.min(start, token.start);
      end = Math.max(end, token.end);
    }
    if (!Number.isFinite(start)) return null;
    return {
      text: visibleTokenText(
        promptCharacters
          .slice(start, Math.min(end, start + EXCERPT_LIMIT))
          .join(""),
      ),
      truncated: end - start > EXCERPT_LIMIT,
    };
  }, [report, promptCharacters, selectedBlock]);
  const metricLabel =
    metric === "mean" ? t("每个已采样 token 的平均注意力") : t("块注意力总量");
  const selectClass =
    "h-9 min-w-0 rounded-md border bg-background px-2 text-sm";

  return (
    <section
      className="min-w-0 max-w-full space-y-4"
      aria-label={t("查询 × token 块注意力")}
      data-attention-block-heatmap
      data-layer-id={layerId}
      data-block-size={blockSize}
      data-metric={metric}
      data-global-peak={globalPeak}
      data-block-page={activePage}
    >
      <div className="space-y-1">
        <h3 className="text-sm font-medium">{t("查询 × token 块注意力")}</h3>
        <p className="text-xs text-muted-foreground">
          {t(
            "仅显示已采样查询在所选层的注意力，不是完整 N×N 注意力矩阵，也不表示因果贡献。",
          )}
        </p>
      </div>
      <div className="flex flex-wrap items-end gap-3">
        <label className="flex min-w-0 flex-col gap-1 text-xs">
          {t("每块 token 数")}
          <select
            className={selectClass}
            value={blockSize}
            data-attention-block-size
            onChange={(event) => setBlockSize(Number(event.target.value))}
          >
            {[16, 64, 256].map((size) => (
              <option key={size} value={size}>
                {size}
              </option>
            ))}
          </select>
        </label>
        <label className="flex min-w-0 max-w-full flex-col gap-1 text-xs">
          {t("块显示指标")}
          <select
            className={selectClass}
            value={metric}
            data-attention-block-metric
            onChange={(event) => setMetric(event.target.value as Metric)}
          >
            <option value="mean">{t("每个已采样 token 的平均注意力")}</option>
            <option value="mass">{t("块注意力总量")}</option>
          </select>
        </label>
      </div>
      <div
        className="space-y-2 text-xs text-muted-foreground"
        data-attention-block-legend
      >
        <p>
          {t(
            "统一色阶：所有查询与所有块共享峰值 {peak}；颜色强度 = √(显示值 / 峰值)，翻页不改变色阶。",
            { peak: percent(globalPeak) },
          )}
        </p>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
          <span className="flex items-center gap-2">
            <span
              className="h-3 w-16 rounded-sm"
              style={{
                background:
                  "linear-gradient(to right, transparent, rgb(59 130 246 / 0.82))",
              }}
              aria-hidden="true"
            />
            0% → {percent(globalPeak)}
          </span>
          <span className="flex items-center gap-2">
            <span
              className="h-3 w-3 rounded-sm border bg-background"
              aria-hidden="true"
            />
            {t("已采样零值：0%")}
          </span>
          <span className="flex items-center gap-2">
            <span
              className="h-3 w-3 rounded-sm border border-dashed bg-muted"
              aria-hidden="true"
            />
            {t("未采样：—")}
          </span>
        </div>
        <p>
          {t(
            "数值保留原始概率；均值仅除以块内已采样 token 数，不重新归一化。token 范围为从 0 开始的 [起点, 终点)。",
          )}
        </p>
        <p>
          {t("列标题是块序号；悬停或选择色块查看精确 token 范围与百分比。")}
        </p>
      </div>
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <Button
          type="button"
          size="sm"
          variant="outline"
          disabled={activePage === 0}
          aria-label={t("上一页 token 块")}
          onClick={() => setPage(activePage - 1)}
        >
          {t("上一页")}
        </Button>
        <span data-attention-block-page-label>
          {t("块 {start}–{end} / {total}", {
            start: blockCount ? firstBlock + 1 : 0,
            end: lastBlock,
            total: blockCount,
          })}
        </span>
        <Button
          type="button"
          size="sm"
          variant="outline"
          disabled={activePage + 1 >= pageCount}
          aria-label={t("下一页 token 块")}
          onClick={() => setPage(activePage + 1)}
        >
          {t("下一页")}
        </Button>
        {rowPageCount > 1 && (
          <>
            <Button
              type="button"
              size="sm"
              variant="outline"
              disabled={activeRowPage === 0}
              onClick={() => setRowPage(activeRowPage - 1)}
            >
              {t("上一组查询")}
            </Button>
            <span>
              {t("查询 {start}–{end} / {total}", {
                start: activeRowPage * ROWS_PER_PAGE + 1,
                end: Math.min((activeRowPage + 1) * ROWS_PER_PAGE, rows.length),
                total: rows.length,
              })}
            </span>
            <Button
              type="button"
              size="sm"
              variant="outline"
              disabled={activeRowPage + 1 >= rowPageCount}
              onClick={() => setRowPage(activeRowPage + 1)}
            >
              {t("下一组查询")}
            </Button>
          </>
        )}
      </div>
      {rows.length === 0 || blockCount === 0 ? (
        <p className="text-sm text-muted-foreground">
          {t("没有可显示的查询或 token。")}
        </p>
      ) : (
        <div
          className="min-w-0 max-w-full overflow-x-auto rounded-md border"
          data-attention-block-scroll
          role="region"
          aria-label={t("可横向滚动的注意力块矩阵")}
          tabIndex={0}
        >
          <table
            className="border-collapse text-xs"
            style={{ tableLayout: "fixed", width: 112 + columns.length * 26 }}
          >
            <caption className="sr-only">
              {t("查询 × token 块注意力")} · {metricLabel} · {t("观测层")}{" "}
              {layerId}
            </caption>
            <colgroup>
              <col style={{ width: 112 }} />
              {columns.map((index) => (
                <col key={index} style={{ width: 26 }} />
              ))}
            </colgroup>
            <thead>
              <tr>
                <th
                  scope="col"
                  className="sticky left-0 z-10 border-b bg-card p-2 text-left"
                >
                  {t("查询位置 / 块序号")}
                </th>
                {columns.map((blockIndex) => (
                  <th
                    key={blockIndex}
                    scope="col"
                    className="border-b py-2 text-center font-mono text-[9px] font-normal text-muted-foreground"
                  >
                    {blockIndex + 1}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {visibleRows.map((row) => (
                <tr
                  key={row.sampleIndex}
                  data-attention-query-row={row.sampleIndex}
                  data-query-position={row.queryPosition}
                >
                  <th
                    scope="row"
                    className="sticky left-0 z-10 border-b bg-card p-2 text-left font-normal"
                  >
                    <div className="font-mono">
                      {t("查询位置")} {row.queryPosition}
                    </div>
                    <div
                      className="mt-1 truncate font-mono text-muted-foreground"
                      title={visibleTokenText(row.queryText)}
                    >
                      {visibleTokenText(row.queryText)}
                    </div>
                    {!row.available && (
                      <div className="mt-1 text-muted-foreground">
                        {t("此查询无所选层数据")}
                      </div>
                    )}
                  </th>
                  {columns.map((blockIndex) => {
                    const block = row.blocks[blockIndex];
                    const sampled = row.available && block.count > 0;
                    const value = block[metric];
                    const selected =
                      selectedRow === row && selectedBlock === block;
                    const status = !row.available
                      ? t("此查询无所选层数据")
                      : sampled
                        ? percent(value)
                        : t("未采样");
                    const label = t(
                      "查询 {query}，token [{start}, {end})，{metric}：{value}，已采样 {count} tokens",
                      {
                        query: row.queryPosition,
                        start: block.start,
                        end: block.end,
                        metric: metricLabel,
                        value: status,
                        count: block.count,
                      },
                    );
                    return (
                      <td key={blockIndex} className="border-b p-0.5">
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          className={`relative h-7 w-full rounded-sm border p-0 font-mono text-[9px] text-foreground ${sampled ? "border-border" : "border-dashed border-border bg-muted text-muted-foreground"} ${selected ? "ring-2 ring-ring ring-offset-1 ring-offset-background" : ""}`}
                          style={
                            sampled
                              ? {
                                  backgroundColor: `rgb(59 130 246 / ${globalPeak > 0 ? 0.82 * Math.sqrt(value / globalPeak) : 0})`,
                                }
                              : undefined
                          }
                          aria-label={label}
                          aria-pressed={selected}
                          title={label}
                          data-attention-block-cell
                          data-sample-index={row.sampleIndex}
                          data-block-index={blockIndex}
                          data-token-start={block.start}
                          data-token-end={block.end}
                          data-sampled={sampled}
                          data-count={block.count}
                          data-value={sampled ? value : undefined}
                          onClick={() => {
                            setSelection({
                              rows,
                              sampleIndex: row.sampleIndex,
                              blockIndex,
                            });
                            if (row.available)
                              onSelectToken(block.start, row.sampleIndex);
                          }}
                        >
                          {sampled ? (
                            value === 0 ? (
                              "0"
                            ) : (
                              <span className="sr-only">{percent(value)}</span>
                            )
                          ) : (
                            "—"
                          )}
                        </Button>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {selectedRow && selectedBlock ? (
        <div
          className="min-w-0 space-y-3 rounded-md border p-3"
          data-attention-block-detail
          aria-live="polite"
        >
          <h4 className="text-sm font-medium">
            {t("选中块：查询 {query} · token [{start}, {end})", {
              query: selectedRow.queryPosition,
              start: selectedBlock.start,
              end: selectedBlock.end,
            })}
          </h4>
          <p className="break-all font-mono text-xs">
            {t("查询文本")}：{visibleTokenText(selectedRow.queryText)}
          </p>
          <dl className="flex flex-wrap gap-x-6 gap-y-2 text-xs">
            <div>
              <dt className="text-muted-foreground">
                {t("已采样 / 块内 token")}
              </dt>
              <dd>
                {selectedBlock.count} /{" "}
                {selectedBlock.end - selectedBlock.start}
              </dd>
            </div>
            {(["mass", "mean", "peak"] as const).map((key) => (
              <div key={key}>
                <dt className="text-muted-foreground">
                  {key === "mass"
                    ? t("块注意力总量")
                    : key === "mean"
                      ? t("每个已采样 token 的平均注意力")
                      : t("块内 token 峰值")}
                </dt>
                <dd data-attention-block-stat={key}>
                  {selectedRow.available && selectedBlock.count > 0
                    ? percent(selectedBlock[key])
                    : "—"}
                </dd>
              </div>
            ))}
          </dl>
          {!selectedRow.available && (
            <p className="text-xs text-muted-foreground">
              {t("此查询无所选层数据")}
            </p>
          )}
          {selectedRow.available && selectedBlock.count === 0 && (
            <p className="text-xs text-muted-foreground">{t("未采样")}</p>
          )}
          <p className="text-xs text-muted-foreground">
            {t("块内原文（空白可见）")}
          </p>
          <pre
            className="max-h-48 overflow-y-auto whitespace-pre-wrap break-all rounded bg-muted p-3 font-mono text-xs"
            data-attention-block-excerpt
          >
            {excerpt?.text ?? t("此块没有有效原文范围")}
          </pre>
          {excerpt?.truncated && (
            <p className="text-xs text-muted-foreground">
              {t("原文预览仅显示此块的前 {count} 个字符。", {
                count: EXCERPT_LIMIT,
              })}
            </p>
          )}
        </div>
      ) : (
        <p className="text-xs text-muted-foreground">
          {t("选择一个块查看原文与原始统计，并定位到 token 文本。")}
        </p>
      )}
    </section>
  );
}
