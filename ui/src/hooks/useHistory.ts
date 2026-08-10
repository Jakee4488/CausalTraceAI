// Sidebar conversation list with cursor pagination.
// Ported from causal-agent.js:742-791.

import { useCallback, useState } from "react";
import { fetchHistory } from "../lib/api";
import type { Conversation } from "../types";

export function useHistory() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [cursor, setCursor] = useState<string | null>(null);

  const reload = useCallback(async () => {
    try {
      const page = await fetchHistory();
      if (!page) return;
      setConversations(page.conversations || []);
      setCursor(page.next_cursor || null);
    } catch (e) {
      console.error("Failed to load history:", e);
    }
  }, []);

  const loadMore = useCallback(async () => {
    if (!cursor) return;
    try {
      const page = await fetchHistory(cursor);
      if (!page) return;
      setConversations((prev) => [...prev, ...(page.conversations || [])]);
      setCursor(page.next_cursor || null);
    } catch (e) {
      console.error("Failed to load history:", e);
    }
  }, [cursor]);

  const reset = useCallback(() => {
    setConversations([]);
    setCursor(null);
  }, []);

  return { conversations, hasMore: !!cursor, reload, loadMore, reset };
}
