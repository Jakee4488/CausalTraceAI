// Live pipeline state for the in-flight turn.
//
// Progress frames arrive on every ADK event. State drives the render; a ref
// mirrors it so the caller can read the final stage list the moment the run
// ends, without racing the last setState.

import { useCallback, useRef, useState } from "react";
import type { ProgressFrame } from "../lib/api";
import { applyProgress, initialStages, type Stage } from "../lib/stages";
import type { CausalGraph } from "../types";

export function useRunProgress() {
  const [stages, setStages] = useState<Stage[]>(initialStages);
  const [graph, setGraph] = useState<CausalGraph | null>(null);
  const stagesRef = useRef<Stage[]>(stages);
  const elapsedRef = useRef(0);

  const reset = useCallback(() => {
    const fresh = initialStages();
    stagesRef.current = fresh;
    elapsedRef.current = 0;
    setStages(fresh);
    setGraph(null);
  }, []);

  const onProgress = useCallback((frame: ProgressFrame) => {
    elapsedRef.current = frame.elapsed_ms;
    setStages((prev) => {
      const next = applyProgress(prev, frame);
      stagesRef.current = next;
      return next;
    });
  }, []);

  const onGraph = useCallback((next: CausalGraph | null) => {
    if (next) setGraph(next);
  }, []);

  return { stages, graph, stagesRef, elapsedRef, reset, onProgress, onGraph };
}
