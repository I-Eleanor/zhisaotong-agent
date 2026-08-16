import type { AgentEvent, ChatMessage, HealthInfo } from "./types";

const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";

/**
 * 消费后端 SSE 流（POST + JSON body）。使用 XHR，避免浏览器扩展拦截 fetch。
 */
export async function streamAgent(
  endpoint: string,
  body: unknown,
  onEvent: (ev: AgentEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  console.log("[streamAgent] 开始请求", endpoint);

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}${endpoint}`);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.responseType = "text";
    xhr.timeout = 120000;

    if (signal) {
      signal.addEventListener("abort", () => xhr.abort());
    }

    let lastIndex = 0;
    let buffer = "";

    xhr.onprogress = () => {
      const newText = xhr.responseText.substring(lastIndex);
      lastIndex = xhr.responseText.length;
      if (!newText) return;

      buffer += newText;
      console.log("[XHR] 收到数据, 长度:", newText.length, "buffer:", buffer.length);

      // SSE 事件分隔符可能是 \r\n\r\n 或 \n\n，先统一为 \n\n
      buffer = buffer.replace(/\r\n/g, "\n");

      // 解析完整的 SSE 事件（以 \n\n 分隔）
      let sep: number;
      while ((sep = buffer.indexOf("\n\n")) !== -1) {
        const raw = buffer.substring(0, sep);
        buffer = buffer.substring(sep + 2);

        // 提取 data: 行
        let dataLine = "";
        for (const line of raw.split("\n")) {
          const trimmed = line.trim();
          if (trimmed.startsWith("data:")) {
            dataLine += trimmed.substring(5).trim();
          }
        }
        if (!dataLine) continue;

        try {
          const ev = JSON.parse(dataLine) as AgentEvent;
          console.log("[SSE] 事件:", ev.type, ev.content?.slice(0, 40));
          onEvent(ev);
        } catch (e) {
          console.warn("[SSE] JSON 解析失败:", dataLine.slice(0, 80), e);
        }
      }
    };

    xhr.onload = () => {
      console.log("[XHR] 完成, status:", xhr.status, "总长度:", xhr.responseText.length);
      if (xhr.status >= 200 && xhr.status < 300) {
        // 处理剩余 buffer
        if (buffer.trim()) {
          let dataLine = "";
          for (const line of buffer.split("\n")) {
            const trimmed = line.trim();
            if (trimmed.startsWith("data:")) {
              dataLine += trimmed.substring(5).trim();
            }
          }
          if (dataLine) {
            try {
              const ev = JSON.parse(dataLine) as AgentEvent;
              onEvent(ev);
            } catch {}
          }
        }
        resolve();
      } else {
        reject(new Error(`请求失败 (${xhr.status}): ${xhr.statusText}`));
      }
    };

    xhr.onerror = () => reject(new Error("网络请求失败"));
    xhr.ontimeout = () => reject(new Error("请求超时 (120s)"));
    xhr.send(JSON.stringify(body));
  });
}

export async function syncChat(query: string, history: ChatMessage[], mode?: string): Promise<{ answer: string }> {
  const resp = await fetch(`${API_BASE}/api/chat/sync`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, history, mode }),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err.error || `同步请求失败 (${resp.status})`);
  }
  return resp.json();
}

export async function uploadKnowledge(files: File[]): Promise<{ success: boolean; file_count: number }> {
  const form = new FormData();
  for (const f of files) form.append("files", f, f.name);
  const resp = await fetch(`${API_BASE}/api/knowledge/upload`, { method: "POST", body: form });
  if (!resp.ok) throw new Error(`上传失败 (${resp.status})`);
  return resp.json();
}

export async function rebuildKnowledge(): Promise<{ success: boolean; chunk_count: number }> {
  const resp = await fetch(`${API_BASE}/api/knowledge/rebuild`, { method: "POST" });
  if (!resp.ok) throw new Error(`重建失败 (${resp.status})`);
  return resp.json();
}

export async function checkHealth(): Promise<HealthInfo> {
  const resp = await fetch(`${API_BASE}/api/health`, { method: "GET" });
  if (!resp.ok) throw new Error(`健康检查失败 (${resp.status})`);
  return resp.json();
}

export type { ChatMessage };