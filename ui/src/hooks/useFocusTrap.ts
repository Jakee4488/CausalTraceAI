import { useEffect, type RefObject } from "react";

const FOCUSABLE = [
  "a[href]",
  "button:not([disabled])",
  "input:not([disabled])",
  "select:not([disabled])",
  "textarea:not([disabled])",
  '[tabindex]:not([tabindex="-1"])',
].join(",");

/**
 * Contains keyboard focus inside a dialog and restores it on close.
 *
 * Declaring `aria-modal="true"` without this is worse than not declaring it:
 * assistive tech stops announcing the background, but Tab still walks straight
 * into it — which is exactly what the drawers used to do, landing on the
 * sidebar behind their own backdrop.
 */
export function useFocusTrap(ref: RefObject<HTMLElement | null>, onClose: () => void): void {
  useEffect(() => {
    const panel = ref.current;
    if (!panel) return;

    const previous = document.activeElement as HTMLElement | null;
    panel.focus();

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.stopPropagation();
        onClose();
        return;
      }
      if (event.key !== "Tab") return;

      const targets = Array.from(panel.querySelectorAll<HTMLElement>(FOCUSABLE))
        .filter((el) => el.offsetParent !== null || el === document.activeElement);
      if (targets.length === 0) {
        // Nothing to move to, so the panel itself keeps focus.
        event.preventDefault();
        panel.focus();
        return;
      }

      const first = targets[0];
      const last = targets[targets.length - 1];
      const active = document.activeElement;

      if (event.shiftKey && (active === first || active === panel)) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && active === last) {
        event.preventDefault();
        first.focus();
      }
    };

    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      // Only take focus back if it is still inside the panel being unmounted;
      // otherwise the user has already moved on and we would yank them back.
      if (previous && panel.contains(document.activeElement)) previous.focus();
    };
  }, [ref, onClose]);
}
