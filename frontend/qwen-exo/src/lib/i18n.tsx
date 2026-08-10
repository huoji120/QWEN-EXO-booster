import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import english from "@/i18n/en-US.json";

export const LANGUAGE_STORAGE_KEY = "qwen-exo-language";

export type Language = "zh-CN" | "en-US";
export type LanguagePreference = "browser" | Language;
export type TranslationValues = Record<string, string | number>;

const ENGLISH_TRANSLATIONS = english as Record<string, string>;
const SUPPORTED_PREFERENCES = new Set<LanguagePreference>([
  "browser",
  "zh-CN",
  "en-US",
]);

const RUNTIME_STATE_SOURCES: Record<string, string> = {
  created: "已创建",
  starting: "启动中",
  ready: "就绪",
  stopping: "停止中",
  stopped: "已停止",
  failed: "失败",
  disabled: "已停用",
  unavailable: "不可用",
  offline: "离线",
  connecting: "连接中",
};

export function runtimeStateSource(
  state: string | null | undefined,
  fallback = "离线",
) {
  const value = String(state || "").trim();
  return RUNTIME_STATE_SOURCES[value] || value || fallback;
}

let activeLanguage: Language = "zh-CN";

function interpolate(template: string, values?: TranslationValues) {
  if (!values) return template;
  return template.replace(/\{([a-zA-Z0-9_]+)\}/g, (match, key: string) =>
    Object.prototype.hasOwnProperty.call(values, key)
      ? String(values[key])
      : match,
  );
}

export function detectBrowserLanguage(language = navigator.language): Language {
  const normalized = String(language || "")
    .trim()
    .toLowerCase();
  if (normalized.startsWith("en")) return "en-US";
  if (normalized.startsWith("zh")) return "zh-CN";
  return "zh-CN";
}

export function resolveLanguage(
  preference: LanguagePreference,
  browserLanguage: Language,
): Language {
  return preference === "browser" ? browserLanguage : preference;
}

export function translateFor(
  language: Language,
  source: string,
  values?: TranslationValues,
) {
  const template =
    language === "en-US" ? ENGLISH_TRANSLATIONS[source] || source : source;
  return interpolate(template, values);
}

export function currentLanguage() {
  return activeLanguage;
}

export function translate(source: string, values?: TranslationValues) {
  return translateFor(activeLanguage, source, values);
}

function storedPreference(): LanguagePreference {
  try {
    const value = window.localStorage.getItem(LANGUAGE_STORAGE_KEY);
    return value && SUPPORTED_PREFERENCES.has(value as LanguagePreference)
      ? (value as LanguagePreference)
      : "browser";
  } catch {
    return "browser";
  }
}

type I18nContextValue = {
  language: Language;
  locale: Language;
  preference: LanguagePreference;
  setPreference: (preference: LanguagePreference) => void;
  t: typeof translate;
};

const I18nContext = createContext<I18nContextValue | null>(null);

export function I18nProvider({ children }: { children: ReactNode }) {
  const [preference, setStoredPreference] =
    useState<LanguagePreference>(storedPreference);
  const [browserLanguage, setBrowserLanguage] = useState<Language>(() =>
    detectBrowserLanguage(),
  );
  const language = resolveLanguage(preference, browserLanguage);
  activeLanguage = language;

  const setPreference = useCallback((next: LanguagePreference) => {
    setStoredPreference(next);
    try {
      window.localStorage.setItem(LANGUAGE_STORAGE_KEY, next);
    } catch {
      // Local storage is optional; the in-memory selection still applies.
    }
  }, []);

  useEffect(() => {
    const onLanguageChange = () => setBrowserLanguage(detectBrowserLanguage());
    const onStorage = (event: StorageEvent) => {
      if (event.key === LANGUAGE_STORAGE_KEY) {
        setStoredPreference(storedPreference());
      }
    };
    window.addEventListener("languagechange", onLanguageChange);
    window.addEventListener("storage", onStorage);
    return () => {
      window.removeEventListener("languagechange", onLanguageChange);
      window.removeEventListener("storage", onStorage);
    };
  }, []);

  useEffect(() => {
    activeLanguage = language;
    document.documentElement.lang = language;
    document.title = translateFor(language, "QWEN EXO 控制台");
    document
      .querySelector('meta[name="description"]')
      ?.setAttribute(
        "content",
        translateFor(language, "QWEN EXO 模型推理、原生记忆与运维控制台"),
      );
  }, [language]);

  const t = useCallback(
    (source: string, values?: TranslationValues) =>
      translateFor(language, source, values),
    [language],
  );
  const value = useMemo<I18nContextValue>(
    () => ({ language, locale: language, preference, setPreference, t }),
    [language, preference, setPreference, t],
  );

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n() {
  const value = useContext(I18nContext);
  if (!value) throw new Error("useI18n must be used within I18nProvider");
  return value;
}
