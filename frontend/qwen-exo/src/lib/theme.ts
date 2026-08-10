import { useEffect, useState } from "react";

export type ThemeMode = "light" | "dark" | "system";

const STORAGE_KEY = "qwen-exo-theme";

function initialMode(): ThemeMode {
  const saved = window.localStorage.getItem(STORAGE_KEY);
  return saved === "light" || saved === "dark" || saved === "system"
    ? saved
    : "system";
}

export function useTheme() {
  const [mode, setMode] = useState<ThemeMode>(initialMode);
  const [dark, setDark] = useState(
    () =>
      mode === "dark" ||
      (mode === "system" &&
        window.matchMedia("(prefers-color-scheme: dark)").matches),
  );

  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const update = () => {
      const next = mode === "dark" || (mode === "system" && media.matches);
      setDark(next);
      document.documentElement.classList.toggle("dark", next);
      document.documentElement.style.colorScheme = next ? "dark" : "light";
    };
    update();
    window.localStorage.setItem(STORAGE_KEY, mode);
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, [mode]);

  return { mode, setMode, dark };
}
