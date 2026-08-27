import { useEffect, useRef, useState } from "react";
import { Headset, Send } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Textarea } from "@/components/ui/textarea";
import { MessageBubble } from "./MessageBubble";
import { HandoffTicketCard } from "./HandoffTicketCard";
import { streamAgent, syncChat } from "@/lib/api";
import { useHandoff } from "@/hooks/useHandoff";
import type { ChatMessage } from "@/lib/types";

interface UIMessage extends ChatMessage {
  pending?: boolean;
  status?: string;
  label?: string;
}

// 会话内稳定的会话标识（与后端人工工单的 conversation_id 关联）。
// 同一页面/组件生命周期内保持唯一，转人工与后续提问都携带它，
// 确保人工接管期间后端能识别“同一次会话”并让模型停手。
function createConversationId(): string {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return `conv-${crypto.randomUUID()}`;
  }
  return `conv-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

export function ChatPanel() {
  const [messages, setMessages] = useState<UIMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const handoff = useHandoff();
  const { state: handoffState, requestHandoff, handleEvent, refreshStatus, suggestHandoff } = handoff;
  // 组件实例内稳定的会话 ID：所有 /api/chat 请求与人工工单共用
  const conversationIdRef = useRef<string>(createConversationId());
  // 记录已把多少条人工回复注入过消息列表（按工单重置）
  const appliedReplies = useRef<{ ticketId: string | null; count: number }>({
    ticketId: null,
    count: 0,
  });

  // 人工回复以普通对话气泡形式插入消息列表（位于机器回复区域），无需用户手动刷新
  useEffect(() => {
    const replies = handoffState.replies ?? [];
    const tid = handoffState.ticketId;
    if (appliedReplies.current.ticketId !== tid) {
      appliedReplies.current = { ticketId: tid, count: 0 };
    }
    if (replies.length > appliedReplies.current.count) {
      const fresh = replies.slice(appliedReplies.current.count);
      appliedReplies.current.count = replies.length;
      setMessages((prev) => [
        ...prev,
        ...fresh.map((r) => ({
          role: "assistant" as const,
          content: r.content,
          label: "人工客服",
        })),
      ]);
    }
  }, [handoffState.replies, handoffState.ticketId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const onHandoffClick = async () => {
    // 取最近一条用户问题作为工单主题
    const lastUser = [...messages].reverse().find((m) => m.role === "user");
    const question = lastUser?.content || input.trim() || "用户请求转人工";
    const history = messages
      .filter((m) => !m.pending)
      .map((m) => ({ role: m.role, content: m.content }));
    // 使用会话内稳定的 conversation_id 定位人工工单，与后续 /api/chat 保持一致
    const conversationId = conversationIdRef.current;
    const ok = await requestHandoff({
      user_question: question,
      conversation_id: conversationId,
      conversation_summary: history
        .map((m) => `${m.role}: ${m.content}`)
        .join("\n")
        .slice(-1000),
      recent_conversations: history,
    });
    if (ok) {
      // 追加一条系统提示消息；工单号由上方「人工工单卡片」展示（不用闭包陈旧值）
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "已为您转接人工客服，请查看上方工单卡片。" },
      ]);
    }
  };

  const send = async () => {
    const q = input.trim();
    if (!q || loading) return;
    setInput("");

    const history = messages
      .filter((m) => !m.pending)
      .map((m) => ({ role: m.role, content: m.content }));

    setMessages((prev) => [
      ...prev,
      { role: "user", content: q },
      { role: "assistant", content: "", pending: true, status: "" },
    ]);
    setLoading(true);

    let streamFailed = false;

    try {
      await streamAgent(
        "/api/chat",
        { query: q, history, conversation_id: conversationIdRef.current },
        (ev) => {
          console.log("[ChatPanel onEvent]", ev.type, ev.content?.slice(0, 50));
          // 更新转人工状态：human_replied 会写入 replies，由下方 effect 渲染成对话气泡（与轮询去重）
          handleEvent(ev);
          setMessages((prev) => {
            const arr = [...prev];
            const last = arr[arr.length - 1];
            if (!last || last.role !== "assistant") {
              console.warn("[ChatPanel] last is not assistant, skip", last);
              return prev;
            }
            const updated = { ...last };
            if (ev.type === "message" || ev.type === "report") {
              updated.content = (updated.content || "") + (ev.content || "");
            } else if (ev.type === "tool_start") {
              updated.status = `调用工具：${ev.data?.tool || "未知"}`;
            } else if (ev.type === "route") {
              updated.status = `路由：${ev.data?.mode_label || ev.content || ""}`;
            } else if (ev.type === "handoff_suggested") {
              // 接收转人工时立刻给出可见反馈“正在转人工...”
              updated.content = "正在转人工...";
              updated.status = `转人工：${ev.content || ""}`;
            } else if (ev.type === "handoff_created") {
              updated.content = (updated.content || "") + (ev.content || "");
              updated.status = `转人工：${ev.content || ""}`;
            } else if (ev.type === "handoff_human_service") {
              // 人工接管期间模型停手：占位气泡显示提示，不产出模型回答
              updated.content = ev.content || "您当前处于人工服务中，消息将由人工客服回复。";
              updated.status = "";
            } else if (ev.type === "error") {
              updated.content = (updated.content || "") + `\n[错误] ${ev.content}`;
            }
            arr[arr.length - 1] = updated;
            return arr;
          });
        }
      );
    } catch (e: any) {
      streamFailed = true;
      console.warn("[ChatPanel] 流式请求失败，尝试同步兜底", e?.message);

      // 同步兜底：一次性获取完整回答
      try {
        const sync = await syncChat(q, history, undefined, conversationIdRef.current);
        setMessages((prev) => {
          const arr = [...prev];
          const last = arr[arr.length - 1];
          if (last) {
            arr[arr.length - 1] = {
              ...last,
              content: sync.answer,
              pending: false,
              status: "",
            };
          }
          return arr;
        });
        // 后端判定需人工介入（need_handoff）时给出提示
        if (sync.need_handoff) {
          suggestHandoff("检测到您可能需要人工协助，可点击“转人工客服”创建工单。");
        }
      } catch (syncErr: any) {
        setMessages((prev) => {
          const arr = [...prev];
          const last = arr[arr.length - 1];
          if (last)
            arr[arr.length - 1] = {
              ...last,
              content: `[连接失败] ${e?.message || e}；同步兜底也失败：${syncErr?.message || syncErr}`,
              pending: false,
            };
          return arr;
        });
      }
    } finally {
      if (!streamFailed) {
        setMessages((prev) =>
          prev.map((m, i) => (i === prev.length - 1 ? { ...m, pending: false, status: "" } : m))
        );
      }
      setLoading(false);
    }
  };

  return (
    <div className="flex h-full flex-col">
      <ScrollArea className="flex-1 px-1">
        <div className="space-y-4 py-4 pr-2">
          {(handoffState.notice || handoffState.error || handoffState.ticketId) ? (
            <HandoffTicketCard
              state={handoffState}
              onRefresh={() => refreshStatus()}
            />
          ) : null}
          {messages.length === 0 ? (
            <div className="mt-10 text-center text-sm text-muted-foreground">
              👋 你好，我是智扫通客服。有什么可以帮你？
            </div>
          ) : (
            messages.map((m, i) => (
              <MessageBubble
                key={i}
                role={m.role}
                content={m.content}
                status={m.status}
                pending={m.pending}
                label={m.label}
              />
            ))
          )}
          <div ref={bottomRef} />
        </div>
      </ScrollArea>

      <div className="mt-3 flex items-end gap-2 border-t pt-3">
        <Textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
          placeholder="输入您的问题，Enter 发送 / Shift+Enter 换行"
          className="max-h-32 min-h-[44px] flex-1 resize-none"
        />
        <Button
          onClick={onHandoffClick}
          disabled={!!handoffState.creating || !handoffState.canRequest}
          variant="outline"
          size="icon"
          aria-label="转人工客服"
          title="转人工客服"
          className="h-11 w-11 shrink-0"
        >
          <Headset className="h-5 w-5" />
        </Button>
        <Button
          onClick={send}
          disabled={loading || !input.trim()}
          size="icon"
          className="h-11 w-11 shrink-0"
        >
          <Send className="h-5 w-5" />
        </Button>
      </div>
    </div>
  );
}
