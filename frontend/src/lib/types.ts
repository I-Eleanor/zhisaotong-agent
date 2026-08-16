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
  | "done";

export interface AgentEvent {
  type: AgentEventType;
  agent?: string;
  content?: string;
  data?: Record<string, any>;
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
