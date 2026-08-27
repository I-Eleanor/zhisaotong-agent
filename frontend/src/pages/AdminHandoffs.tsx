import { useCallback, useEffect, useState } from "react";
import { ArrowLeft, CheckCircle2, RefreshCw, Reply, SearchX, ShieldAlert } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import { adminGetHandoff, adminListHandoffs, adminLogin, adminReplyHandoff } from "@/lib/api";
import { AdminApiError } from "@/lib/types";
import type { AdminHandoffTicket } from "@/lib/types";

const PAGE_SIZE = 10;

const STATUS_LABEL: Record<string, string> = {
  pending: "排队中",
  processing: "处理中",
  resolved: "已解决",
  closed: "已关闭",
};

const STATUS_OPTIONS = [
  { value: "", label: "全部" },
  { value: "pending", label: "排队中" },
  { value: "processing", label: "处理中" },
  { value: "resolved", label: "已解决" },
  { value: "closed", label: "已关闭" },
];

function statusClass(status?: string): string {
  if (status === "resolved" || status === "closed") {
    return "bg-emerald-500/10 text-emerald-600 dark:text-emerald-300";
  }
  if (status === "processing") return "bg-sky-500/10 text-sky-600 dark:text-sky-300";
  return "bg-amber-500/10 text-amber-600 dark:text-amber-300";
}

function formatTime(iso?: string): string {
  if (!iso) return "-";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString();
}

function renderSources(sources: unknown): string {
  if (Array.isArray(sources)) {
    return sources
      .map((s: any) => {
        if (typeof s === "string") return s;
        if (s && typeof s === "object") {
          return [s.content, s.title, s.source].filter(Boolean).join(" — ");
        }
        return String(s);
      })
      .join("\n");
  }
  if (typeof sources === "string") return sources;
  if (sources == null) return "";
  try {
    return JSON.stringify(sources, null, 2);
  } catch {
    return String(sources);
  }
}

function DetailField({ label, value }: { label: string; value?: string }) {
  return (
    <div>
      <div className="text-xs font-medium text-muted-foreground">{label}</div>
      <div className="whitespace-pre-wrap text-sm">{value || "-"}</div>
    </div>
  );
}

/**
 * 管理端人工工单页面（/admin/handoffs）。
 * 仅通过管理接口交互；本页面绝不读取/渲染 access_key / access_key_hash（接口也不返回）。
 * 管理 Token 由环境变量提供，不硬编码。
 */
export function AdminHandoffs() {
  const [items, setItems] = useState<AdminHandoffTicket[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [statusFilter, setStatusFilter] = useState("");
  const [loading, setLoading] = useState(true);
  const [listError, setListError] = useState<string | null>(null);

  const [detail, setDetail] = useState<AdminHandoffTicket | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [replyText, setReplyText] = useState("");
  const [replying, setReplying] = useState(false);
  const [replyError, setReplyError] = useState<string | null>(null);

  // 登录态（会话 cookie 由后端下发，前端只负责调用登录接口）
  const [authRequired, setAuthRequired] = useState(false);
  const [authError, setAuthError] = useState<string | null>(null);
  const [tokenInput, setTokenInput] = useState("");
  const [loggingIn, setLoggingIn] = useState(false);

  const describeError = (e: unknown): string => {
    if (e instanceof AdminApiError) {
      if (e.status === 401) return "权限校验失败（401）：请检查管理员 Token 是否正确。";
      if (e.status === 403) return "管理员功能未启用（403）。";
    }
    return `加载失败：${e instanceof Error ? e.message : "网络异常，请稍后重试。"}`;
  };

  const loadList = useCallback(
    async (nextOffset: number, nextStatus: string) => {
      setLoading(true);
      setListError(null);
      try {
        const data = await adminListHandoffs({
          status: nextStatus || undefined,
          limit: PAGE_SIZE,
          offset: nextOffset,
        });
        setItems(data.items);
        setTotal(data.total);
        setOffset(data.offset);
      } catch (e) {
        // 未登录/会话失效（401）→ 切到登录视图；403=功能未启用，登录无效，走错误提示
        if (e instanceof AdminApiError && e.status === 401) {
          setAuthRequired(true);
        } else {
          setListError(describeError(e));
        }
        setItems([]);
        setTotal(0);
        setDetail(null);
      } finally {
        setLoading(false);
      }
    },
    []
  );

  useEffect(() => {
    loadList(0, statusFilter);
  }, [loadList, statusFilter]);

  // 详情打开期间轮询刷新，实时同步用户新消息与人工回复
  const detailId = detail?.ticket_id;
  useEffect(() => {
    if (!detailId) return;
    const id = window.setInterval(async () => {
      try {
        const fresh = await adminGetHandoff(detailId);
        setDetail((prev) => (prev && prev.ticket_id === detailId ? fresh : prev));
      } catch {
        // 轮询失败静默（会话过期等由显式操作触发登录）
      }
    }, 2000);
    return () => window.clearInterval(id);
  }, [detailId]);

  const changeStatus = (value: string) => {
    setStatusFilter(value);
    setOffset(0);
  };

  const doLogin = async () => {
    if (!tokenInput.trim()) return;
    setLoggingIn(true);
    setAuthError(null);
    try {
      await adminLogin(tokenInput.trim());
      setAuthRequired(false);
      setTokenInput("");
      await loadList(0, statusFilter);
    } catch (e) {
      setAuthError(describeError(e));
    } finally {
      setLoggingIn(false);
    }
  };

  const selectTicket = async (t: AdminHandoffTicket) => {
    setDetail(t);
    setReplyError(null);
    setReplyText("");
    setLoadingDetail(true);
    try {
      const full = await adminGetHandoff(t.ticket_id);
      setDetail(full);
      setReplyText("");
    } catch {
      // 详情拉取失败时保留列表里已有的行级数据
    } finally {
      setLoadingDetail(false);
    }
  };

  const saveReply = async () => {
    if (!detail || !replyText.trim()) return;
    setReplying(true);
    setReplyError(null);
    try {
      const updated = await adminReplyHandoff(detail.ticket_id, {
        human_reply: replyText.trim(),
        status: "processing",
      });
      setDetail(updated);
      setReplyText("");
      await loadList(offset, statusFilter);
    } catch (e) {
      setReplyError(describeError(e));
    } finally {
      setReplying(false);
    }
  };

  const completeHandoff = async () => {
    if (!detail) return;
    setReplying(true);
    setReplyError(null);
    try {
      // 有输入则作为最后一条回复一并保存；无输入则仅结束工单（后端不会追加空回复）
      const updated = await adminReplyHandoff(detail.ticket_id, {
        human_reply: replyText.trim(),
        status: "resolved",
      });
      setDetail(updated);
      setReplyText("");
      await loadList(offset, statusFilter);
    } catch (e) {
      setReplyError(describeError(e));
    } finally {
      setReplying(false);
    }
  };

  const hasPrev = offset > 0;
  const hasNext = offset + PAGE_SIZE < total;

  return (
    <div className="flex h-screen flex-col bg-background">
      {authRequired ? (
        <div className="flex flex-1 items-center justify-center p-4">
          <Card className="w-full max-w-sm">
            <CardContent className="p-6">
              <div className="mb-2 flex items-center gap-2 text-sm font-semibold">
                <ShieldAlert className="h-4 w-4" /> 需要管理员登录
              </div>
              <p className="mb-3 text-xs text-muted-foreground">
                请输入管理员 Token 完成登录（会话由系统保存，退出登录或重启后需重新登录）。
              </p>
              <input
                aria-label="管理员 Token"
                type="password"
                className="h-9 w-full rounded-md border bg-transparent px-3 text-sm"
                placeholder="管理员 Token"
                value={tokenInput}
                onChange={(e) => setTokenInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") void doLogin();
                }}
                autoFocus
              />
              {authError ? (
                <div className="mt-2 text-xs text-destructive">{authError}</div>
              ) : null}
              <Button className="mt-3 w-full" onClick={() => void doLogin()} disabled={loggingIn || !tokenInput.trim()}>
                {loggingIn ? "登录中…" : "登录"}
              </Button>
            </CardContent>
          </Card>
        </div>
      ) : (
        <>
      <header className="flex shrink-0 items-center justify-between border-b px-6 py-3">
        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            size="icon"
            title="返回客户页面"
            onClick={() => (window.location.hash = "#/")}
          >
            <ArrowLeft className="h-5 w-5" />
          </Button>
          <div>
            <h1 className="text-lg font-bold leading-tight">人工工单管理</h1>
            <p className="text-xs text-muted-foreground">共 {total} 条工单</p>
          </div>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => loadList(offset, statusFilter)}
          disabled={loading}
        >
          <RefreshCw className="h-4 w-4" /> 刷新
        </Button>
      </header>

      <div className="flex h-full min-h-0 flex-col gap-4 p-4 lg:flex-row">
        {/* 列表 */}
        <Card className="flex min-h-0 flex-1 flex-col">
          <CardContent className="flex min-h-0 flex-1 flex-col p-4">
            <div className="mb-3 flex items-center gap-2">
              <span className="text-sm text-muted-foreground">状态筛选</span>
              <select
                aria-label="状态筛选"
                className="h-9 rounded-md border bg-transparent px-3 text-sm"
                value={statusFilter}
                onChange={(e) => changeStatus(e.target.value)}
              >
                {STATUS_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
            </div>

            {listError ? (
              <div className="flex items-center gap-2 rounded-md border border-destructive/40 p-3 text-sm text-destructive">
                <ShieldAlert className="h-4 w-4 shrink-0" />
                {listError}
              </div>
            ) : null}

            {!listError && loading ? (
              <div className="flex flex-1 items-center justify-center text-sm text-muted-foreground">
                加载中…
              </div>
            ) : null}

            {!listError && !loading && items.length === 0 ? (
              <div className="flex flex-1 flex-col items-center justify-center gap-2 text-muted-foreground">
                <SearchX className="h-8 w-8" />
                <span className="text-sm">暂无工单</span>
              </div>
            ) : null}

            {!listError && !loading && items.length > 0 ? (
              <div className="min-h-0 flex-1 space-y-2 overflow-y-auto">
                {items.map((t) => (
                  <button
                    key={t.ticket_id}
                    onClick={() => selectTicket(t)}
                    className={cn(
                      "flex w-full items-center justify-between gap-3 rounded-lg border p-3 text-left transition-colors",
                      detail?.ticket_id === t.ticket_id
                        ? "border-primary bg-muted/60"
                        : "border-border hover:bg-muted/40"
                    )}
                  >
                    <div className="min-w-0">
                      <div className="font-mono text-sm">{t.ticket_id}</div>
                      <div className="truncate text-xs text-muted-foreground">{t.user_question}</div>
                      <div className="text-xs text-muted-foreground">{formatTime(t.created_at)}</div>
                    </div>
                    <div className="flex flex-col items-end gap-1">
                      <Badge className={cn("shrink-0", statusClass(t.status))}>
                        {STATUS_LABEL[t.status] ?? t.status}
                      </Badge>
                      <div className="text-xs text-muted-foreground">
                        {t.handoff_reason ? `转人工：${t.handoff_reason}` : "-"}
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            ) : null}

            {/* 分页 */}
            <div className="mt-3 flex shrink-0 items-center justify-between">
              <span className="text-xs text-muted-foreground">
                第 {total === 0 ? 0 : offset + 1}–{Math.min(offset + PAGE_SIZE, total)} 条 / 共 {total} 条
              </span>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  disabled={!hasPrev}
                  onClick={() => loadList(Math.max(0, offset - PAGE_SIZE), statusFilter)}
                >
                  上一页
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  disabled={!hasNext}
                  onClick={() => loadList(offset + PAGE_SIZE, statusFilter)}
                >
                  下一页
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* 详情 */}
        <Card className="flex min-h-0 flex-1 flex-col">
          <CardContent className="min-h-0 flex-1 overflow-y-auto p-4">
            {!detail ? (
              <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
                点击左侧工单查看详情
              </div>
            ) : (
              <div className="flex min-h-0 flex-col gap-4">
                <div className="flex items-center justify-between">
                  <div className="font-mono text-sm">{detail.ticket_id}</div>
                  <Badge className={cn("shrink-0", statusClass(detail.status))}>
                    {STATUS_LABEL[detail.status] ?? detail.status}
                  </Badge>
                </div>

                {/* 实时对话：用户/人工按时间正序展示，打开详情期间每 2 秒自动同步 */}
                <div>
                  <div className="mb-1 text-xs font-medium text-muted-foreground">
                    实时对话
                  </div>
                  <div className="flex flex-col gap-2 rounded-lg border p-3">
                    {(detail.live_messages ?? []).length === 0 ? (
                      <div className="text-sm text-muted-foreground">暂无对话</div>
                    ) : (
                      (detail.live_messages ?? []).map((m, i) => {
                        const isUser = m.role === "user";
                        return (
                          <div
                            key={i}
                            className={cn(
                              "max-w-[90%] rounded-lg px-3 py-2 text-sm",
                              isUser
                                ? "self-end bg-primary text-primary-foreground"
                                : "self-start bg-muted"
                            )}
                          >
                            {!isUser ? (
                              <div className="mb-0.5 text-xs text-muted-foreground">人工客服</div>
                            ) : null}
                            <div className="whitespace-pre-wrap">{m.content}</div>
                          </div>
                        );
                      })
                    )}
                  </div>
                </div>

                <div className="grid gap-3 sm:grid-cols-2">
                  <DetailField label="创建时间" value={formatTime(detail.created_at)} />
                  <DetailField label="转人工原因" value={detail.handoff_reason} />
                </div>
                <DetailField label="用户问题" value={detail.user_question} />
                <DetailField label="对话摘要" value={detail.conversation_summary} />
                <DetailField label="RAG 来源" value={renderSources(detail.retrieval_sources)} />
                <DetailField label="诊断结果" value={detail.diagnostic_result} />

                <div>
                  <div className="mb-1 text-xs font-medium text-muted-foreground">
                    人工回复记录（{detail.replies?.length ?? 0}）
                  </div>
                  {detail.replies && detail.replies.length > 0 ? (
                    <div className="space-y-2">
                      {detail.replies.map((r, i) => (
                        <div
                          key={i}
                          className="whitespace-pre-wrap rounded bg-muted p-2 text-sm"
                        >
                          {r.content}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-sm text-muted-foreground">暂无回复</div>
                  )}
                </div>

                <div className="mt-2">
                  <div className="mb-1 text-xs font-medium text-muted-foreground">
                    回复用户（保存回复后状态变为“处理中”，可多次回复；全部解答后点“完成工单”结束）
                  </div>
                  <Textarea
                    aria-label="人工回复"
                    value={replyText}
                    onChange={(e) => setReplyText(e.target.value)}
                    placeholder="请输入给用户的回复…"
                    rows={4}
                    disabled={loadingDetail || replying}
                  />
                  {replyError ? (
                    <div className="mt-1 text-xs text-destructive">{replyError}</div>
                  ) : null}
                  <div className="mt-2 flex flex-wrap items-center gap-2">
                    <Button
                      onClick={saveReply}
                      disabled={loadingDetail || replying || !replyText.trim()}
                    >
                      <Reply className="h-4 w-4" /> {replying ? "提交中…" : "保存回复"}
                    </Button>
                    <Button
                      variant="secondary"
                      onClick={completeHandoff}
                      disabled={loadingDetail || replying || detail.status === "resolved" || detail.status === "closed"}
                    >
                      <CheckCircle2 className="h-4 w-4" /> 完成工单
                    </Button>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
        </>
      )}
    </div>
  );
}