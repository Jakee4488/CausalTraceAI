export interface SseMessage {
  event: string;
  data: string;
}

export async function* readSse(response: Response): AsyncGenerator<SseMessage> {
  if (!response.body) throw new Error("Missing response body");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let event = "message";
  let data = "";

  const emit = (): SseMessage | null => {
    if (!data) return null;
    const message = { event, data: data.endsWith("\n") ? data.slice(0, -1) : data };
    event = "message";
    data = "";
    return message;
  };

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });

    let split = buffer.indexOf("\n");
    while (split >= 0) {
      const line = buffer.slice(0, split).replace(/\r$/, "");
      buffer = buffer.slice(split + 1);

      if (line === "") {
        const message = emit();
        if (message) yield message;
      } else if (line.startsWith("event:")) {
        event = line.slice(6).trim() || "message";
      } else if (line.startsWith("data:")) {
        data += `${line.slice(5).trimStart()}\n`;
      }

      split = buffer.indexOf("\n");
    }

    if (done) {
      const message = emit();
      if (message) yield message;
      break;
    }
  }
}
