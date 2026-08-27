import { Bot, Loader2, User } from "lucide-react";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { cn } from "@/lib/utils";

interface Props {
  role: "user" | "assistant";
  content: string;
  status?: string;
  pending?: boolean;
  /** 气泡上方的小标签（如“人工客服”），用于区分人工回复与机器人回复。 */
  label?: string;
}

export function MessageBubble({ role, content, status, pending, label }: Props) {
  const isUser = role === "user";
  return (
    <div className={cn("flex gap-3 animate-fade-in", isUser ? "flex-row-reverse" : "flex-row")}>
      <Avatar className={cn("h-8 w-8 shrink-0", isUser ? "bg-secondary" : "bg-primary/10")}>
        <AvatarFallback>
          {isUser ? (
            <User className="h-4 w-4" />
          ) : (
            <Bot className="h-4 w-4 text-primary" />
          )}
        </AvatarFallback>
      </Avatar>
      <div
        className={cn(
          "max-w-[80%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed",
          isUser
            ? "rounded-tr-sm bg-primary text-primary-foreground"
            : "rounded-tl-sm bg-muted"
        )}
      >
        {status ? (
          <div className="mb-1 flex items-center gap-1.5 text-xs text-muted-foreground">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            {status}
          </div>
        ) : null}
        {label && content ? (
          <div className="mb-1 text-xs font-medium text-primary">{label}</div>
        ) : null}
        {content ? (
          <div className="whitespace-pre-wrap">{content}</div>
        ) : pending ? (
          <span className="text-muted-foreground">正在思考…</span>
        ) : null}
      </div>
    </div>
  );
}
