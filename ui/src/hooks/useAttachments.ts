// Upload + chip state. Ported from causal-agent.js:508-604.

import { useCallback, useRef, useState } from "react";
import { uploadFile } from "../lib/api";
import type { Attachment } from "../types";

// Mirrors ALLOWED_UPLOAD_EXTENSIONS / MAX_UPLOAD_BYTES in proxy/main.py.
export const ALLOWED_EXTENSIONS = [
  ".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".yaml", ".yml",
  ".toml", ".xml", ".html", ".css", ".js", ".ts", ".py", ".java", ".go",
  ".rs", ".c", ".cpp", ".h", ".sh", ".sql", ".log",
];
export const MAX_UPLOAD_BYTES = 5 * 1024 * 1024;

export function formatSize(bytes: number): string {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(bytes < 10 * 1024 ? 1 : 0) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

export function useAttachments() {
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const counter = useRef(0);

  const patch = useCallback((localId: string, changes: Partial<Attachment>) => {
    setAttachments((prev) =>
      prev.map((a) => (a.localId === localId ? { ...a, ...changes } : a)),
    );
  }, []);

  const handleFiles = useCallback(
    (fileList: FileList | File[] | null) => {
      Array.from(fileList || []).forEach((file) => {
        counter.current += 1;
        const localId = `att-${Date.now()}-${counter.current}`;
        const dot = file.name.lastIndexOf(".");
        const ext = dot >= 0 ? file.name.slice(dot).toLowerCase() : "";

        setAttachments((prev) => [
          ...prev,
          { localId, id: null, name: file.name, size: file.size, status: "uploading" },
        ]);

        // Client-side pre-checks mirror the backend's 415/413 responses.
        if (!ALLOWED_EXTENSIONS.includes(ext)) {
          patch(localId, {
            status: "error",
            error: `Unsupported file type '${ext || file.name}'`,
          });
          return;
        }
        if (file.size > MAX_UPLOAD_BYTES) {
          patch(localId, { status: "error", error: "File exceeds 5 MB limit" });
          return;
        }

        void uploadFile(file)
          .then((data) => patch(localId, { id: data.file_id, status: "done" }))
          .catch((error: Error) => {
            console.error("Upload failed:", error);
            patch(localId, { status: "error", error: error.message });
          });
      });
    },
    [patch],
  );

  const remove = useCallback((localId: string) => {
    setAttachments((prev) => prev.filter((a) => a.localId !== localId));
  }, []);

  const clear = useCallback(() => setAttachments([]), []);

  return { attachments, handleFiles, remove, clear };
}
