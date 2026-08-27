import { CheckCircle2, Layers, RefreshCw, XCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { HandoffState } from "@/hooks/useHandoff";
import { cn } from "@/lib/utils";

const STATUS_LABEL: Record<string, string> = {
  pending: "排队中",
  processing: "人工处理中",
  resolved: "已解决",
  closed: "已关闭",
};

/**
 * 人工工单状态卡片：展示工单创建成功提示、工单号与实时状态。
 * 绝不渲染 access_key（该凭证仅存在于前端会话内部）。
 */
export function HandoffTicketCard({
  state,
  onRefresh,
}: {
  state: HandoffState;
  onRefresh?: () => void;
}) {
  const { status, ticketId, notice, error, creating, hasAccessKey } = state;

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-border bg-card p-3 text-sm">
      {notice ? (
        <div className="flex items-start gap-2 text-foreground">
          <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
          <span>{notice}</span>
        </div>
      ) : null}

      {ticketId ? (
        <div className="flex items-center gap-2 text-muted-foreground">
          <Layers className="h-4 w-4 shrink-0" />
          <span className="font-mono">{ticketId}</span>
          {status ? (
            <span
              className={cn(
                "rounded-full px-2 py-0.5 text-xs font-medium",
                status === "resolved" || status === "closed"
                  ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-300"
                  : "bg-amber-500/10 text-amber-600 dark:text-amber-300"
              )}
            >
              {STATUS_LABEL[status] ?? status}
            </span>
          ) : null}
        </div>
      ) : null}

      {error ? (
        <div className="flex items-start gap-2 text-destructive">
          <XCircle className="mt-0.5 h-4 w-4 shrink-0" />
          <span>{error}</span>
        </div>
      ) : null}

      {hasAccessKey && ticketId ? (
        <Button
          variant="ghost"
          size="sm"
          className="w-fit gap-1 self-start"
          onClick={onRefresh}
          disabled={!!creating}
        >
          <RefreshCw className="h-3.5 w-3.5" /> 刷新状态
        </Button>
      ) : null}
    </div>
  );
}