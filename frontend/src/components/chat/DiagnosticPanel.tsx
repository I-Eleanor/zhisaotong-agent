import { useEffect, useRef, useState } from "react";
import { Send, Stethoscope } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Textarea } from "@/components/ui/textarea";
import { EventRenderer } from "./EventRenderer";
import { streamAgent } from "@/lib/api";
import type { AgentEvent } from "@/lib/types";

interface Turn {
  query: string;
  events: AgentEvent[];
  pending: boolean;
}

export function DiagnosticPanel() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns]);

  const run = async () => {
    const q = input.trim();
    if (!q || loading) return;
    setInput("");

    setTurns((t) => [...t, { query: q, events: [], pending: true }]);
    setLoading(true);

    const append = (ev: AgentEvent) =>
      setTurns((prev) => {
        const arr = prev.map((t, i) => (i === prev.length - 1 ? { ...t } : t));
        const last = arr[arr.length - 1];
        last.events = [...last.events, ev];
        return arr;
      });

    try {
      await streamAgent("/api/diagnose", { query: q }, append);
    } catch (e: any) {
      append({ type: "error", content: String(e?.message || e) });
    } finally {
      setTurns((prev) =>
        prev.map((t, i) => (i === prev.length - 1 ? { ...t, pending: false } : t))
      );
      setLoading(false);
    }
  };

  return (
    <div className="flex h-full flex-col">
      <ScrollArea className="flex-1 px-1">
        <div className="space-y-4 py-4 pr-2">
          {turns.length === 0 ? (
            <div className="mt-10 text-center text-sm text-muted-foreground">
              🛠 描述设备故障，例如「最近清洁效率很低」，我会逐步排查并给出报告。
            </div>
          ) : (
            turns.map((turn, ti) => (
              <div key={ti} className="space-y-2">
                <div className="flex justify-end">
                  <div className="max-w-[80%] rounded-2xl rounded-tr-sm bg-primary px-4 py-2.5 text-sm text-primary-foreground">
                    {turn.query}
                  </div>
                </div>
                <div className="space-y-2 pl-2">
                  {turn.events.map((ev, ei) => (
                    <EventRenderer key={ei} event={ev} />
                  ))}
                  {turn.pending ? (
                    <div className="text-xs text-muted-foreground">排查进行中…</div>
                  ) : null}
                </div>
              </div>
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
              run();
            }
          }}
          placeholder="描述您的设备故障，例如「最近清洁效率很低」"
          className="max-h-32 min-h-[44px] flex-1 resize-none"
        />
        <Button
          onClick={run}
          disabled={loading || !input.trim()}
          size="icon"
          className="h-11 w-11 shrink-0"
        >
          <Stethoscope className="h-5 w-5" />
        </Button>
      </div>
    </div>
  );
}
