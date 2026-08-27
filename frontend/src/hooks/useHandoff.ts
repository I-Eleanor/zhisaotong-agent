import { useCallback, useEffect, useRef, useState } from "react";
import { createHandoffTicket, fetchHandoffTicket } from "@/lib/api";
import type { AgentEvent, ChatMessage, HandoffReply, HandoffStatus, HandoffTicket } from "@/lib/types";

export interface HandoffContext {
  user_question: string;
  conversation_id?: string;
  conversation_summary?: string;
  recent_conversations?: ChatMessage[] | string;
  retrieval_sources?: unknown;
  diagnostic_result?: string;
  handoff_reason?: string;
}

export interface HandoffState {
  status: HandoffStatus | null;
  ticketId: string | null;
  creating: boolean;
  /** 上一份工单成功后给用户看到的提示（可含 ticket_id）。 */
  notice: string | null;
  error: string | null;
  /** 是否已持有 access_key（当前前端会话内，仅用于刷新状态，绝不渲染）。 */
  hasAccessKey: boolean;
  /** 人工客服的最新回复（如有）。 */
  reply: string | null;
  /** 多轮人工回复历史（按时间正序，用于渲染成对话消息）。 */
  replies: HandoffReply[];
  /** 会话能否发起新工单（防重复）。 */
  canRequest: boolean;
}

interface UseHandoffOptions {
  /** 确认对话框；便于测试注入。默认 window.confirm。 */
  confirm?: (message: string) => boolean;
}

/**
 * 用户端「转人工」hook：负责捕获 access_key、建单、防重复、SSE 事件与状态刷新。
 *
 * - access_key 仅在建单成功时保存在本 hook 的 state 中（当前前端会话），
 *   不放入聊天消息内容，也不提供给渲染层展示；
 * - 同一会话内已有进行中的工单时禁止重复请求（防重复提交）。
 */
export function useHandoff(options: UseHandoffOptions = {}) {
  const confirmFn = options.confirm ?? ((msg: string) => window.confirm(msg));
  const [ticketId, setTicketId] = useState<string | null>(null);
  const [status, setStatus] = useState<HandoffStatus | null>(null);
  const [creating, setCreating] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reply, setReply] = useState<string | null>(null);
  const [replies, setReplies] = useState<HandoffReply[]>([]);
  // access_key 只在内存态：字段存在即代表已持有，不用于展示
  const [, setAccessKey] = useState<string | null>(null);
  const accessKeyRef = useRef<string | null>(null);

  const canRequest = !creating && ticketId === null;
  const hasAccessKey = accessKeyRef.current !== null;

  /** 处理来自 /api/chat、/api/diagnose SSE 流的转人工相关事件。 */
  const handleEvent = useCallback((ev: AgentEvent) => {
    if (ev.type === "handoff_suggested") {
      setError(null);
      setNotice(null);
    } else if (ev.type === "handoff_created") {
      const data = ev.data ?? {};
      const tid = typeof data.ticket_id === "string" ? data.ticket_id : null;
      // 自动建单路径：事件一次性携带明文 access_key → 仅保存在当前会话内存状态
      if (typeof data.access_key === "string" && data.access_key) {
        accessKeyRef.current = data.access_key;
      }
      if (tid) {
        setTicketId(tid);
        setNotice(`人工工单创建成功：工单号 ${tid}。`);
      }
      if (data.status) setStatus(data.status as HandoffStatus);
      setReply(null);
    } else if (ev.type === "human_replied") {
      const data = ev.data ?? {};
      const content = ev.content || "";
      if (content) {
        setReply(content);
        setReplies((prev) => {
          const last = prev[prev.length - 1];
          if (last && last.content === content) return prev; // 与轮询去重
          return [...prev, { content, created_at: null }];
        });
      }
      if (data.status) setStatus(data.status as HandoffStatus);
    } else if (ev.type === "handoff_resolved") {
      setStatus((ev.data?.status as HandoffStatus) || "closed");
    }
  }, []);

  /** 用户点击「转人工客服」：确认后建单。返回是否创建成功。 */
  const requestHandoff = useCallback(
    async (ctx: HandoffContext): Promise<boolean> => {
      // 防重复：正在请求或已存在工单时直接拒绝
      if (creating || ticketId !== null) {
        setError("已存在进行中的人工工单，请勿重复提交。");
        return false;
      }
      if (!confirmFn("确认转接人工客服吗？将为您创建一个人工工单。")) {
        return false;
      }
      setCreating(true);
      setError(null);
      setNotice(null);
      try {
        const ticket: HandoffTicket = await createHandoffTicket({
          user_question: ctx.user_question,
          conversation_id: ctx.conversation_id,
          conversation_summary: ctx.conversation_summary,
          recent_conversations: ctx.recent_conversations,
          retrieval_sources: ctx.retrieval_sources,
          diagnostic_result: ctx.diagnostic_result,
          handoff_reason: ctx.handoff_reason ?? "user_request",
        });
        // access_key 只进内存态，绝不渲染、不写入消息内容
        setAccessKey(ticket.access_key ?? null);
        accessKeyRef.current = ticket.access_key ?? null;
        setTicketId(ticket.ticket_id);
        setStatus(ticket.status ?? "pending");
        setReply(ticket.human_reply ?? null);
        setReplies(ticket.replies ?? []);
        setNotice(`人工工单创建成功，工单号：${ticket.ticket_id}。客服将尽快与您联系。`);
        return true;
      } catch (e) {
        const msg = e instanceof Error ? e.message : "未知错误";
        setError(`人工转接失败：${msg}，请稍后重试。`);
        return false;
      } finally {
        setCreating(false);
      }
    },
    [confirmFn, creating, ticketId]
  );

  /** 展示一条转人工提示（用于 sync 兜底场景：后端已自动建单但 access_key 不可得）。 */
  const suggestHandoff = useCallback((message: string) => {
    setNotice(message);
    setError(null);
  }, []);

  /** 用会话内 access_key 刷新工单最新状态（仅当前会话内存使用）。 */
  const refreshStatus = useCallback(async () => {
    const tid = ticketId;
    const key = accessKeyRef.current;
    if (!tid || !key) return;
    try {
      const ticket = await fetchHandoffTicket(tid, key);
      setStatus(ticket.status ?? status);
      setReply(ticket.human_reply ?? null);
      setReplies(ticket.replies ?? []);
      setError(null);
    } catch {
      // 刷新失败不打断用户，静默保留当前状态
    }
  }, [ticketId, status]);

  // 实时同步：存在未结束工单时轮询刷新，让用户免手动刷新即可看到“人工处理中/已解决/人工回复”
  useEffect(() => {
    if (!ticketId || !accessKeyRef.current) return;
    if (status === "resolved" || status === "closed") return;
    const id = window.setInterval(() => {
      void refreshStatus();
    }, 3000);
    return () => window.clearInterval(id);
  }, [ticketId, status, refreshStatus]);

  return {
    state: {
      status,
      ticketId,
      creating,
      notice,
      error,
      hasAccessKey,
      reply,
      replies,
      canRequest,
    } satisfies HandoffState,
    requestHandoff,
    handleEvent,
    refreshStatus,
    suggestHandoff,
    getAccessKey: () => accessKeyRef.current,
  };
}

export type UseHandoffReturn = ReturnType<typeof useHandoff>;