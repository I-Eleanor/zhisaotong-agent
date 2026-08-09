import {
  AlertTriangle,
  Cpu,
  FileText,
  ListChecks,
  RefreshCw,
  Wrench,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { AgentEvent } from "@/lib/types";

// 把一个 AgentEvent 渲染成适合「设备诊断」时间线的视觉块
export function EventRenderer({ event }: { event: AgentEvent }) {
  const { type, content, data } = event;

  switch (type) {
    case "route":
      return (
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Cpu className="h-4 w-4 text-primary" />
          <span>
            已路由至 <Badge variant="accent">{data?.mode_label || content || "未知"}</Badge>
          </span>
        </div>
      );

    case "tool_start":
      return (
        <Badge variant="secondary" className="gap-1">
          <Wrench className="h-3.5 w-3.5" /> 调用工具：{data?.tool || "未知"}
        </Badge>
      );

    case "plan": {
      const steps: string[] = data?.steps || [];
      return (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-base">
              <ListChecks className="h-4 w-4 text-primary" /> 排查计划
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ol className="list-decimal space-y-1 pl-5 text-sm">
              {steps.map((s, i) => (
                <li key={i}>{s}</li>
              ))}
            </ol>
          </CardContent>
        </Card>
      );
    }

    case "step":
      return (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">
              步骤 {data?.index ?? "?"}：{data?.description || ""}
            </CardTitle>
          </CardHeader>
          {content ? <CardContent className="md">{content}</CardContent> : null}
        </Card>
      );

    case "replan":
      return (
        <div className="flex items-start gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-sm text-amber-700 dark:text-amber-300">
          <RefreshCw className="mt-0.5 h-4 w-4 shrink-0" />
          <span>
            <b>重规划：</b>
            {content}
          </span>
        </div>
      );

    case "report":
      return (
        <Card className="border-primary/30">
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-base">
              <FileText className="h-4 w-4 text-primary" /> 诊断报告
            </CardTitle>
          </CardHeader>
          <CardContent className="md">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{content || ""}</ReactMarkdown>
          </CardContent>
        </Card>
      );

    case "message":
      return content ? <div className="text-sm text-muted-foreground">{content}</div> : null;

    case "error":
      return (
        <div className="flex items-start gap-2 rounded-lg border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" /> {content}
        </div>
      );

    default:
      return content ? <div className="whitespace-pre-wrap text-sm">{content}</div> : null;
  }
}
