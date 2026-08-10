import { useEffect, useState } from "react";

/**
 * Live media-query state.
 *
 * The layout used to sample its breakpoint exactly once, in a `useState`
 * initialiser, so nothing ever re-read it: resizing to 700px left the sidebar
 * open and squeezed the chat column to 249px, and there was no path back.
 */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() => window.matchMedia(query).matches);

  useEffect(() => {
    const list = window.matchMedia(query);
    const onChange = (event: MediaQueryListEvent) => setMatches(event.matches);
    // Re-sync on subscribe: the query may have flipped between the initial
    // render and this effect running.
    setMatches(list.matches);
    list.addEventListener("change", onChange);
    return () => list.removeEventListener("change", onChange);
  }, [query]);

  return matches;
}
