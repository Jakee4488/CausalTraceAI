import DOMPurify from "dompurify";
import { marked } from "marked";

// hljs core plus an explicit language set, rather than the default entry point.
// That entry registers ~190 grammars (roughly 900KB raw) and every one of them
// shipped in the main chunk to highlight the handful of languages this agent
// actually emits. Auto-detection now only considers what is registered here, so
// add a language when the model starts producing it.
import hljs from "highlight.js/lib/core";
import bash from "highlight.js/lib/languages/bash";
import c from "highlight.js/lib/languages/c";
import cpp from "highlight.js/lib/languages/cpp";
import css from "highlight.js/lib/languages/css";
import go from "highlight.js/lib/languages/go";
import java from "highlight.js/lib/languages/java";
import javascript from "highlight.js/lib/languages/javascript";
import json from "highlight.js/lib/languages/json";
import markdown from "highlight.js/lib/languages/markdown";
import python from "highlight.js/lib/languages/python";
import r from "highlight.js/lib/languages/r";
import rust from "highlight.js/lib/languages/rust";
import sql from "highlight.js/lib/languages/sql";
import typescript from "highlight.js/lib/languages/typescript";
import xml from "highlight.js/lib/languages/xml";
import yaml from "highlight.js/lib/languages/yaml";

// Python and R lead deliberately: the code executor runs Python, and causal
// analysis answers quote R far more often than this stack otherwise would.
for (const [name, language] of Object.entries({
  python, r, sql, json, yaml, bash, javascript, typescript,
  markdown, xml, css, java, go, rust, c, cpp,
})) {
  hljs.registerLanguage(name, language);
}

// `breaks` stays off deliberately. Models hard-wrap prose at ~80 columns, and
// with breaks:true every one of those newlines became a <br> — the answer was
// laid out at the model's column width instead of the bubble's, and never
// reflowed on resize. GFM still honours a blank line as a paragraph and a
// trailing double-space as an explicit break, which is all the model needs.
marked.setOptions({ gfm: true, breaks: false });

const CITATION_RE = /\[Node:\s*([^\]]+)\]/gi;

/** Nodes whose text is verbatim source — a citation there is a false positive. */
const VERBATIM = new Set(["CODE", "PRE", "KBD", "SAMP"]);

/**
 * Rewrites `[Node: X]` into a real button, in place, over the sanitized DOM.
 *
 * Walking text nodes rather than running a regex over the serialized HTML is
 * what keeps the transform out of code samples, and it means the markup we
 * inject is built with createElement instead of being spliced in after
 * DOMPurify has already run.
 */
function linkCitations(root: HTMLElement): void {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      for (let el = node.parentElement; el && el !== root; el = el.parentElement) {
        if (VERBATIM.has(el.tagName)) return NodeFilter.FILTER_REJECT;
      }
      // Reset first: CITATION_RE is /g, and .test() on a global regex resumes
      // from lastIndex and leaves it advanced. Across sibling text nodes that
      // made the filter skip roughly every other citation-bearing paragraph —
      // those citations rendered as literal "[Node: X]" text and could not be
      // clicked to highlight the graph.
      CITATION_RE.lastIndex = 0;
      return CITATION_RE.test(node.nodeValue ?? "")
        ? NodeFilter.FILTER_ACCEPT
        : NodeFilter.FILTER_REJECT;
    },
  });

  const targets: Text[] = [];
  for (let n = walker.nextNode(); n; n = walker.nextNode()) targets.push(n as Text);

  for (const text of targets) {
    const source = text.nodeValue ?? "";
    const frag = document.createDocumentFragment();
    let cursor = 0;

    CITATION_RE.lastIndex = 0;
    for (let m = CITATION_RE.exec(source); m; m = CITATION_RE.exec(source)) {
      if (m.index > cursor) {
        frag.append(source.slice(cursor, m.index));
      }
      const label = m[1].trim();
      const button = document.createElement("button");
      button.type = "button";
      button.className = "node-citation";
      button.dataset.node = label;
      button.title = `Causal node: ${label} — show in the graph`;
      button.textContent = label;
      frag.append(button);
      cursor = m.index + m[0].length;
    }

    if (cursor < source.length) frag.append(source.slice(cursor));
    text.replaceWith(frag);
  }
}

/** A cell whose content is a number, optionally with a unit or a %. */
const NUMERIC_CELL = /^[-+]?[$£€]?\s*\d[\d,.\s]*\s*(%|[a-zA-Z°µ/²³]{1,6})?$/;

/**
 * Right-aligns table columns whose body cells are all numeric.
 *
 * Models almost never emit GFM alignment markers, so effect tables arrived
 * left-aligned and their magnitudes could not be read down the column.
 */
function alignNumericColumns(root: HTMLElement): void {
  for (const table of root.querySelectorAll("table")) {
    const rows = Array.from(table.tBodies).flatMap((b) => Array.from(b.rows));
    if (rows.length === 0) continue;
    const columns = Math.max(...rows.map((r) => r.cells.length));

    for (let col = 0; col < columns; col += 1) {
      const cells = rows.map((r) => r.cells[col]).filter(Boolean);
      if (cells.length === 0) continue;
      const numeric = cells.every((c) => NUMERIC_CELL.test((c.textContent || "").trim()));
      if (!numeric) continue;

      for (const cell of cells) cell.classList.add("md-num");
      for (const head of Array.from(table.tHead?.rows ?? [])) {
        head.cells[col]?.classList.add("md-num");
      }
    }
  }
}

export function renderMarkdown(text: string): string {
  const raw = marked.parse(text ?? "");
  return DOMPurify.sanitize(typeof raw === "string" ? raw : "");
}

/**
 * Post-render passes over the mounted output: syntax highlighting, then the
 * citation buttons. Both need the real DOM, so both live here rather than in
 * the string-producing path above.
 */
export function enhanceMarkdown(root: HTMLElement | null): void {
  if (!root) return;
  for (const node of root.querySelectorAll("pre code")) {
    hljs.highlightElement(node as HTMLElement);
  }
  alignNumericColumns(root);
  linkCitations(root);
}

export function fmtNum(value: number | null | undefined, digits = 3): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
}
