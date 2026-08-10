import { useEffect, useRef, useState } from "react";

function dragHasFiles(event: DragEvent): boolean {
  const types = event.dataTransfer?.types;
  return !!types && Array.from(types).includes("Files");
}

/**
 * Page-wide drag & drop. The depth counter avoids flicker from child
 * enter/leave events. Ported from causal-agent.js:606-635.
 */
export function DropOverlay({ onFiles }: { onFiles: (files: FileList | null) => void }) {
  const [visible, setVisible] = useState(false);
  const depth = useRef(0);

  useEffect(() => {
    const onEnter = (event: DragEvent) => {
      if (!dragHasFiles(event)) return;
      event.preventDefault();
      depth.current += 1;
      setVisible(true);
    };
    const onOver = (event: DragEvent) => {
      if (dragHasFiles(event)) event.preventDefault();
    };
    const onLeave = (event: DragEvent) => {
      if (!dragHasFiles(event)) return;
      depth.current = Math.max(0, depth.current - 1);
      if (depth.current === 0) setVisible(false);
    };
    const onDrop = (event: DragEvent) => {
      if (!dragHasFiles(event)) return;
      event.preventDefault();
      depth.current = 0;
      setVisible(false);
      onFiles(event.dataTransfer?.files ?? null);
    };

    document.addEventListener("dragenter", onEnter);
    document.addEventListener("dragover", onOver);
    document.addEventListener("dragleave", onLeave);
    document.addEventListener("drop", onDrop);
    return () => {
      document.removeEventListener("dragenter", onEnter);
      document.removeEventListener("dragover", onOver);
      document.removeEventListener("dragleave", onLeave);
      document.removeEventListener("drop", onDrop);
    };
  }, [onFiles]);

  return (
    <div className={visible ? "drop-overlay show" : "drop-overlay"} id="drop-overlay">
      <div className="drop-inner">
        <div className="drop-icon">⤓</div>
        <h2>Drop files to attach</h2>
        <p>.txt · .md · .csv · .json · code — up to 5 MB</p>
      </div>
    </div>
  );
}
