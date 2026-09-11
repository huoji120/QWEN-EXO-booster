export type AttentionToken = {
  id: number;
  text: string;
  start: number;
  end: number;
};

export type AttentionMessage = {
  index: number;
  role: string;
  content: string;
  name?: string;
  tool_call_id?: string;
  start?: number;
  end?: number;
  truncated?: boolean;
};

export type AttentionReport = {
  schema: string;
  method: string;
  model: string;
  prompt_tokens: number;
  rendered_prompt: string;
  tokens: AttentionToken[];
  samples: {
    query_position: number;
    query_text: string;
    layers: { layer_id: number; weights: number[] }[];
  }[];
  messages: AttentionMessage[];
  warnings: string[];
};

export type AttributionCategory =
  | "source"
  | "boundary"
  | "marker"
  | "whitespace"
  | "unassigned"
  | "unmapped";

// Text resemblance only: token IDs are not checked against a tokenizer vocabulary.
const MARKER_LOOKING_TEXT =
  /^(?:<\|[A-Za-z0-9_]+\|>|<think>|<\/think>)(?![\s\S])/u;
const WHITESPACE_TEXT = /^\s+$/u;

function validSpan(
  start: number | undefined,
  end: number | undefined,
): boolean {
  return (
    start !== undefined &&
    end !== undefined &&
    Number.isSafeInteger(start) &&
    Number.isSafeInteger(end) &&
    start >= 0 &&
    end > start
  );
}

export function classifyAttentionTokens(
  tokens: AttentionToken[],
  messages: AttentionMessage[],
): { category: AttributionCategory; messageIndex: number | null }[] {
  const spans: {
    start: number;
    end: number;
    messageIndex: number;
    maxEnd: number;
  }[] = [];
  for (let index = 0; index < messages.length; index++) {
    const { start, end } = messages[index];
    if (start !== undefined && end !== undefined && validSpan(start, end)) {
      spans.push({ start, end, messageIndex: index, maxEnd: end });
    }
  }
  spans.sort((a, b) => a.start - b.start);
  for (let index = 1; index < spans.length; index++) {
    spans[index].maxEnd = Math.max(spans[index - 1].maxEnd, spans[index].end);
  }

  return tokens.map((token) => {
    if (!validSpan(token.start, token.end)) {
      return { category: "unmapped", messageIndex: null };
    }

    // Prefix maxima safely skip expired spans, including nested/overlapping messages.
    // Binary search also handles token offsets that overlap or arrive out of order.
    let low = 0;
    let high = spans.length;
    while (low < high) {
      const middle = low + Math.floor((high - low) / 2);
      if (spans[middle].maxEnd <= token.start) low = middle + 1;
      else high = middle;
    }

    let owner: number | null = null;
    for (
      let index = low;
      index < spans.length && spans[index].start < token.end;
      index++
    ) {
      const span = spans[index];
      if (span.end <= token.start) continue;
      // Any partial overlap or multiple owners is ambiguous, never first-owner wins.
      if (span.start > token.start || span.end < token.end || owner !== null) {
        return { category: "boundary", messageIndex: null };
      }
      owner = span.messageIndex;
    }
    if (owner !== null) return { category: "source", messageIndex: owner };

    // Source-body literals retain source attribution even when they resemble markers.
    if (MARKER_LOOKING_TEXT.test(token.text)) {
      return { category: "marker", messageIndex: null };
    }
    if (WHITESPACE_TEXT.test(token.text)) {
      return { category: "whitespace", messageIndex: null };
    }
    return { category: "unassigned", messageIndex: null };
  });
}

export type AttentionBlock = {
  start: number;
  end: number;
  count: number;
  mass: number;
  mean: number;
  peak: number;
};

export function buildAttentionBlocks(
  tokens: AttentionToken[],
  weights: number[],
  blockSize: number,
): AttentionBlock[] {
  if (!Number.isSafeInteger(blockSize) || blockSize <= 0) {
    throw new RangeError(
      "Attention block size must be a positive safe integer",
    );
  }
  const blocks: AttentionBlock[] = [];
  for (let start = 0; start < tokens.length; start += blockSize) {
    const end = Math.min(start + blockSize, tokens.length);
    let count = 0;
    let mass = 0;
    let peak = 0;
    for (let index = start; index < end && index < weights.length; index++) {
      const weight = weights[index];
      if (weight === undefined) continue;
      count++;
      mass += weight;
      peak = Math.max(peak, weight);
    }
    // Count distinguishes an unsampled block from a sampled block with zero mass.
    blocks.push({
      start,
      end,
      count,
      mass,
      mean: count ? mass / count : 0,
      peak,
    });
  }
  return blocks;
}

export function visibleTokenText(text: string): string {
  if (text === "") return "∅";
  return text.replace(/[\n\t \r]/g, (character) => {
    switch (character) {
      case "\n":
        return "\\n";
      case "\t":
        return "\\t";
      case " ":
        return "·";
      default:
        return "\\r";
    }
  });
}
