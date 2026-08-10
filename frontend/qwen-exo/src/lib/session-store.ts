import type { ChatSession } from "@/lib/types";
import { makeId } from "@/lib/utils";

const STORAGE_KEY = "qwen-exo-console-v2";
const MAX_SESSIONS = 24;
const MAX_MESSAGES = 160;

export function createSession(): ChatSession {
  const now = new Date().toISOString();
  return {
    id: makeId("session"),
    title: "新对话",
    createdAt: now,
    updatedAt: now,
    lastResponseId: null,
    messages: [],
  };
}

export function loadSessions(): ChatSession[] {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter(
        (item) =>
          item && typeof item.id === "string" && Array.isArray(item.messages),
      )
      .slice(0, MAX_SESSIONS);
  } catch {
    return [];
  }
}

export function saveSessions(sessions: ChatSession[]) {
  const bounded = sessions.slice(0, MAX_SESSIONS).map((session) => ({
    ...session,
    messages: session.messages.slice(-MAX_MESSAGES),
  }));
  localStorage.setItem(STORAGE_KEY, JSON.stringify(bounded));
}
