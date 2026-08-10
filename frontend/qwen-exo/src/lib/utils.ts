import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import { currentLanguage } from "@/lib/i18n";

const NUMBER_FORMATTERS = {
  "zh-CN": new Intl.NumberFormat("zh-CN"),
  "en-US": new Intl.NumberFormat("en-US"),
};

const FIXED_NUMBER_FORMATTERS = {
  "zh-CN": {
    one: new Intl.NumberFormat("zh-CN", {
      minimumFractionDigits: 1,
      maximumFractionDigits: 1,
    }),
    two: new Intl.NumberFormat("zh-CN", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }),
  },
  "en-US": {
    one: new Intl.NumberFormat("en-US", {
      minimumFractionDigits: 1,
      maximumFractionDigits: 1,
    }),
    two: new Intl.NumberFormat("en-US", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }),
  },
};

const DATE_TIME_FORMATTERS = {
  "zh-CN": new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }),
  "en-US": new Intl.DateTimeFormat("en-US", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }),
};

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function shortHash(value?: string | null, length = 10) {
  if (!value) return "—";
  return value.length <= length ? value : `${value.slice(0, length)}…`;
}

export function formatNumber(value?: number | null) {
  return typeof value === "number"
    ? NUMBER_FORMATTERS[currentLanguage()].format(value)
    : "—";
}

export function formatDuration(seconds?: number | null) {
  if (typeof seconds !== "number") return "—";
  if (seconds < 1) {
    return `${NUMBER_FORMATTERS[currentLanguage()].format(Math.round(seconds * 1000))} ms`;
  }
  const precision = seconds < 10 ? "two" : "one";
  return `${FIXED_NUMBER_FORMATTERS[currentLanguage()][precision].format(seconds)} s`;
}

export function formatBytes(bytes?: number | null) {
  if (typeof bytes !== "number") return "—";
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1024;
  let unit = units[0];
  for (let index = 1; index < units.length && value >= 1024; index += 1) {
    value /= 1024;
    unit = units[index];
  }
  const precision = value >= 10 ? "one" : "two";
  return `${FIXED_NUMBER_FORMATTERS[currentLanguage()][precision].format(value)} ${unit}`;
}

export function formatTime(value?: string | number | null) {
  if (value === null || value === undefined) return "—";
  const date =
    typeof value === "number" ? new Date(value * 1000) : new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return DATE_TIME_FORMATTERS[currentLanguage()].format(date);
}

export function makeId(prefix: string) {
  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}
