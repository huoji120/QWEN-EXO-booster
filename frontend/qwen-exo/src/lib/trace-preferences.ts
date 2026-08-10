export type TraceDefaultScope = "activity" | "actual" | "all";

export function isTraceDefaultScope(value: string): value is TraceDefaultScope {
  return value === "activity" || value === "actual" || value === "all";
}
