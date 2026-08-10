export type Theme = "light" | "dark";

const STORAGE_KEY = "causaltrace-theme";

function resolveTheme(value: string | null): Theme {
  return value === "light" ? "light" : "dark";
}

function readStoredTheme(): Theme {
  try {
    return resolveTheme(localStorage.getItem(STORAGE_KEY));
  } catch {
    return "dark";
  }
}

export const currentTheme: Theme = readStoredTheme();

export function applyTheme(theme: Theme): void {
  document.documentElement.setAttribute("data-theme", theme);
  try {
    localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    // Ignore blocked storage.
  }
}
