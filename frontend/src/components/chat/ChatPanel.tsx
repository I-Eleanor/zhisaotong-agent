import { useEffect, useRef, useState } from "react";
import { Send } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Textarea } from "@/components/ui/textarea";
import { MessageBubble } from "./MessageBubble";
import { streamAgent, syncChat } from "@/lib/api";
import type { ChatMessage } from "@/lib/types";

interface UIMessage extends ChatMessage {
  pending?: boolean;
  status?: string;
}

export function ChatPanel() {
  const [messages, setMessages] = useState<UIMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

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
        { query: q, history },
        (ev) => {
          console.log("[ChatPanel onEvent]", ev.type, ev.content?.slice(0, 50));
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
        const { answer } = await syncChat(q, history);
        setMessages((prev) => {
          const arr = [...prev];
          const last = arr[arr.length - 1];
          if (last) {
            arr[arr.length - 1] = {
              ...last,
              content: answer,
              pending: false,
              status: "",
            };
          }
          return arr;
        });
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
