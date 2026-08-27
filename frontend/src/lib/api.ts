import type {
  AdminHandoffTicket,
  AdminHandoffsResponse,
  AdminReplyPayload,
  AgentEvent,
  ChatMessage,
  HandoffTicket,
  HealthInfo,
} from "./types";
import { AdminApiError } from "./types";

const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";

/** 解析管理接口错误：401/403 抛出带状态码的 AdminApiError。 */
async function adminError(resp: Response): Promise<Error> {
  const err = await resp.json().catch(() => ({}));
  const message =
    (err as { safe_message?: string }).safe_message ||
    (err as { detail?: string }).detail ||
    (err as { error_code?: string }).error_code ||
    (err as { message?: string }).message ||
    `请求失败 (${resp.status})`;
  if (resp.status === 401 || resp.status === 403) {
    return new AdminApiError(resp.status, message);
  }
  return new Error(message);
}

// 管理端接口鉴权基于服务端会话（HttpOnly Cookie）。
// 请求须携带凭据（credentials: include）以自动带上 Cookie，不再发送任何 Token 头。
function adminFetchInit(): RequestInit {
  return { credentials: "include", headers: { "Content-Type": "application/json" } };
}

export async function adminLogin(adminToken: string): Promise<{ admin_enabled: boolean; cookie_host?: string }> {
  const resp = await fetch(`${API_BASE}/api/admin/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ admin_token: adminToken }),
  });
  if (!resp.ok) throw await adminError(resp);
  return resp.json();
}

export interface AdminListOptions {
  status?: string;
  limit?: number;
  offset?: number;
}

export async function adminListHandoffs(options: AdminListOptions = {}): Promise<AdminHandoffsResponse> {
  const params = new URLSearchParams();
  if (options.status) params.set("status", options.status);
  if (options.limit != null) params.set("limit", String(options.limit));
  if (options.offset != null) params.set("offset", String(options.offset));
  const qs = params.toString();
  const resp = await fetch(`${API_BASE}/api/admin/handoffs${qs ? `?${qs}` : ""}`, {
    method: "GET",
    ...adminFetchInit(),
  });
  if (!resp.ok) throw await adminError(resp);
  return resp.json();
}

export async function adminGetHandoff(ticketId: string): Promise<AdminHandoffTicket> {
  const resp = await fetch(`${API_BASE}/api/admin/handoffs/${encodeURIComponent(ticketId)}`, {
    method: "GET",
    ...adminFetchInit(),
  });
  if (!resp.ok) throw await adminError(resp);
  return resp.json();
}

export async function adminReplyHandoff(
  ticketId: string,
  payload: AdminReplyPayload
): Promise<AdminHandoffTicket> {
  const resp = await fetch(`${API_BASE}/api/admin/handoffs/${encodeURIComponent(ticketId)}/reply`, {
    method: "POST",
    ...adminFetchInit(),
    body: JSON.stringify(payload),
  });
  if (!resp.ok) throw await adminError(resp);
  return resp.json();
}

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
            } catch (e) {
              console.warn("[SSE] 剩余数据解析失败:", dataLine.slice(0, 80), e);
            }
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

export interface SyncChatResult {
  answer: string;
  need_handoff?: boolean;
}

export async function syncChat(
  query: string,
  history: ChatMessage[],
  mode?: string,
  conversationId?: string
): Promise<SyncChatResult> {
  const resp = await fetch(`${API_BASE}/api/chat/sync`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, history, mode, conversation_id: conversationId }),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err.error || `同步请求失败 (${resp.status})`);
  }
  return resp.json();
}

interface CreateHandoffPayload {
  user_question: string;
  conversation_id?: string;
  conversation_summary?: string;
  recent_conversations?: ChatMessage[] | string;
  retrieval_sources?: unknown;
  diagnostic_result?: string;
  handoff_reason?: string;
}

/**
 * 创建人工工单。创建成功返回工单（含一次性 access_key）。
 * access_key 仅在此处获得，调用方应保持在当前前端会话，且不得写入聊天消息内容。
 */
export async function createHandoffTicket(payload: CreateHandoffPayload): Promise<HandoffTicket> {
  const resp = await fetch(`${API_BASE}/api/handoff`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err.safe_message || `工单创建失败 (${resp.status})`);
  }
  return resp.json();
}

/** 用 access_key 查询工单最新状态（不涉及凭证外泄，仅当前会话内存中使用）。 */
export async function fetchHandoffTicket(
  ticketId: string,
  accessKey: string
): Promise<HandoffTicket> {
  const resp = await fetch(
    `${API_BASE}/api/handoff/${encodeURIComponent(ticketId)}?access_key=${encodeURIComponent(accessKey)}`,
    { method: "GET" }
  );
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err.safe_message || `工单查询失败 (${resp.status})`);
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