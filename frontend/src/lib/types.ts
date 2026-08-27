// 与后端 agent/events.py 的 AgentEvent 对齐
export type AgentEventType =
  | "route"
  | "thinking"
  | "message"
  | "tool_start"
  | "tool_end"
  | "plan"
  | "step"
  | "replan"
  | "report"
  | "error"
  | "done"
  | "handoff_suggested"
  | "handoff_created"
  | "handoff_human_service"
  | "human_replied"
  | "handoff_resolved";

export interface AgentEvent {
  type: AgentEventType;
  agent?: string;
  content?: string;
  data?: Record<string, any>;
}

// 人工工单状态：与后端 api/handoff_store.py 的 VALID_STATUSES 对齐
export type HandoffStatus = "pending" | "processing" | "resolved" | "closed";

// 一条人工客服回复（多轮累积，按时间正序）
export interface HandoffReply {
  content: string;
  created_at?: string | null;
}

// 人工工单（POST /api/handoff 创建返回；access_key 仅回传一次）
export interface HandoffTicket {
  ticket_id: string;
  conversation_id?: string;
  status: HandoffStatus;
  user_question: string;
  handoff_reason?: string;
  human_reply?: string | null;
  replies?: HandoffReply[];
  created_at?: string;
  updated_at?: string;
  access_key?: string; // 仅创建响应中一次性返回，前端会话内保存
}

// 管理端工单（管理接口返回；绝不包含 access_key / access_key_hash）
export interface AdminHandoffTicket {
  ticket_id: string;
  conversation_id?: string | null;
  user_question: string;
  conversation_summary?: string;
  recent_conversations?: unknown;
  retrieval_sources?: unknown;
  diagnostic_result?: string;
  handoff_reason?: string;
  status: HandoffStatus;
  human_reply?: string | null;
  replies?: HandoffReply[];
  live_messages?: { role: "user" | "human"; content: string; created_at?: string | null }[];
  created_at?: string;
  updated_at?: string;
}

export interface AdminHandoffsResponse {
  total: number;
  limit: number;
  offset: number;
  items: AdminHandoffTicket[];
}

export interface AdminReplyPayload {
  human_reply: string;
  status?: HandoffStatus;
}

// 管理接口鉴权失败（401/403）时抛出的错误
export class AdminApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = "AdminApiError";
    this.status = status;
  }
}

// 多轮对话历史（发给后端 /api/chat 的 history）
export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

// 后端 /api/health 返回
export interface HealthInfo {
  status: string;
  model: string;
  embedding: string;
  reranker_enabled?: boolean;
}
