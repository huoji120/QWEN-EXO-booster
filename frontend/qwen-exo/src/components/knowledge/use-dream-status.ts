import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { getDreamStatus, runDream } from "@/lib/api";
import { translate as t } from "@/lib/i18n";
import type { DreamStatus } from "@/lib/types";
import { errorText } from "./knowledge-utils";

const RUNNING_POLL_MS = 5_000;
const IDLE_POLL_MS = 60_000;

export type DreamState = {
  status: DreamStatus | null;
  /** ``Date.now()`` when ``status`` was fetched; anchors the countdown. */
  fetchedAt: number;
  error: string | null;
  starting: boolean;
  refresh: () => Promise<void>;
  start: () => Promise<void>;
};

/**
 * Poll the dream agent: every 5 s while a run is active, every 60 s otherwise.
 * ``onFinished`` fires once each time a run stops (work notes may have changed).
 */
export function useDreamStatus(onFinished: () => void): DreamState {
  const [status, setStatus] = useState<DreamStatus | null>(null);
  const [fetchedAt, setFetchedAt] = useState(() => Date.now());
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const wasRunning = useRef(false);
  const finished = useRef(onFinished);
  finished.current = onFinished;

  const apply = useCallback((next: DreamStatus) => {
    setStatus(next);
    setFetchedAt(Date.now());
    setError(null);
    const running = Boolean(next.running);
    if (wasRunning.current && !running) finished.current();
    wasRunning.current = running;
  }, []);

  const refresh = useCallback(async () => {
    try {
      apply(await getDreamStatus());
    } catch (cause) {
      setError(errorText(cause, t("未知错误")));
    }
  }, [apply]);

  const start = useCallback(async () => {
    setStarting(true);
    try {
      apply(await runDream());
      toast.success(t("已开始整理工作记忆"));
    } catch (cause) {
      toast.error(t("无法开始整理"), { description: errorText(cause, t("未知错误")) });
      void refresh();
    } finally {
      setStarting(false);
    }
  }, [apply, refresh]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const running = Boolean(status?.running);
  useEffect(() => {
    const timer = window.setInterval(
      () => void refresh(),
      running ? RUNNING_POLL_MS : IDLE_POLL_MS,
    );
    return () => window.clearInterval(timer);
  }, [refresh, running]);

  return { status, fetchedAt, error, starting, refresh, start };
}
