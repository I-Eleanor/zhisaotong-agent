import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { AdminHandoffs } from "./AdminHandoffs";
import { AdminApiError } from "@/lib/types";
import type { AdminHandoffTicket, AdminHandoffsResponse } from "@/lib/types";

vi.mock("@/lib/api", () => ({
  adminListHandoffs: vi.fn(),
  adminGetHandoff: vi.fn(),
  adminReplyHandoff: vi.fn(),
  adminLogin: vi.fn(),
}));

import { adminListHandoffs, adminGetHandoff, adminReplyHandoff, adminLogin } from "@/lib/api";

const mockList = vi.mocked(adminListHandoffs);
const mockGet = vi.mocked(adminGetHandoff);
const mockReply = vi.mocked(adminReplyHandoff);
const mockLogin = vi.mocked(adminLogin);

function ticket(overrides: Partial<AdminHandoffTicket> = {}): AdminHandoffTicket {
  return {
    ticket_id: "tk-1",
    conversation_id: "conv-1",
    user_question: "扫地机不工作了",
    conversation_summary: "用户：扫地机不工作了\n客服：请描述更多现象",
    retrieval_sources: [],
    diagnostic_result: "可能是滚刷卡滞",
    handoff_reason: "user_request",
    status: "pending",
    human_reply: null,
    created_at: "2026-08-27T10:00:00Z",
    updated_at: "2026-08-27T10:00:00Z",
    ...overrides,
  };
}

function listResp(items: AdminHandoffTicket[], total = items.length, offset = 0): AdminHandoffsResponse {
  return { total, limit: 10, offset, items };
}

beforeEach(() => {
  vi.clearAllMocks();
  mockList.mockReset();
  mockGet.mockReset();
  mockReply.mockReset();
});

describe("AdminHandoffs 管理端", () => {
  it("工单列表加载并展示工单号与总数", async () => {
    mockList.mockResolvedValue(listResp([ticket({ ticket_id: "tk-1" }), ticket({ ticket_id: "tk-2" })]));
    render(<AdminHandoffs />);
    expect(await screen.findByText("tk-1")).toBeInTheDocument();
    expect(screen.getByText("tk-2")).toBeInTheDocument();
    expect(screen.getByText(/共 2 条工单/)).toBeInTheDocument();
  });

  it("状态筛选：切换后以 status 参数重新请求", async () => {
    mockList.mockResolvedValue(listResp([ticket({ ticket_id: "tk-1" }), ticket({ ticket_id: "tk-2" })]));
    const user = userEvent.setup();
    render(<AdminHandoffs />);
    await screen.findByText("tk-1");
    await user.selectOptions(screen.getByLabelText("状态筛选"), "pending");
    await waitFor(() => {
      expect(mockList).toHaveBeenLastCalledWith({ status: "pending", limit: 10, offset: 0 });
    });
  });

  it("分页：下一页以 offset 递增、上一页禁用", async () => {
    const items = Array.from({ length: 10 }, (_, i) => ticket({ ticket_id: `tk-${i}` }));
    mockList.mockResolvedValue(listResp(items, 25, 0));
    const user = userEvent.setup();
    render(<AdminHandoffs />);
    const next = (await screen.findAllByRole("button", { name: /下一页/ })).find((b) => !b.hasAttribute("disabled"))!;
    const prev = screen.getByRole("button", { name: /上一页/ });
    expect(prev).toBeDisabled();
    await user.click(next);
    await waitFor(() => {
      expect(mockList).toHaveBeenCalledWith({ status: undefined, limit: 10, offset: 10 });
    });
  });

  it("点击工单加载详情：展示原因、状态与回复记录", async () => {
    mockList.mockResolvedValue(listResp([ticket({ ticket_id: "tk-1" })]));
    mockGet.mockResolvedValue(
      ticket({ ticket_id: "tk-1", replies: [{ content: "我已经在处理了" }], status: "processing" })
    );
    const user = userEvent.setup();
    render(<AdminHandoffs />);
    await user.click(await screen.findByText("tk-1"));
    expect(screen.getAllByText("我已经在处理了").length).toBeGreaterThan(0);
    expect(screen.getAllByText("user_request").length).toBeGreaterThan(0);
    expect(screen.getAllByText("处理中").length).toBeGreaterThan(0);
  });

  it("上下文展示：RAG 来源、诊断结果、对话摘要渲染", async () => {
    mockList.mockResolvedValue(listResp([ticket({ ticket_id: "tk-1" })]));
    mockGet.mockResolvedValue(
      ticket({
        ticket_id: "tk-1",
        conversation_summary: "摘要行一\n摘要行二",
        retrieval_sources: [{ content: "知识库片段A" }, "来源B"],
        diagnostic_result: "滚刷卡滞",
      })
    );
    const user = userEvent.setup();
    render(<AdminHandoffs />);
    await user.click(await screen.findByText("tk-1"));
    expect(await screen.findByText(/摘要行一/)).toBeInTheDocument();
    expect(screen.getByText(/知识库片段A/)).toBeInTheDocument();
    expect(screen.getByText("滚刷卡滞")).toBeInTheDocument();
  });

  it("保存回复：调用 reply 接口(processing)并刷新详情与列表", async () => {
    mockList.mockResolvedValue(listResp([ticket({ ticket_id: "tk-1" })]));
    mockGet.mockResolvedValue(ticket({ ticket_id: "tk-1" }));
    mockReply.mockResolvedValue(ticket({ ticket_id: "tk-1", status: "processing", human_reply: "已为您解决" }));
    const user = userEvent.setup();
    render(<AdminHandoffs />);
    await user.click(await screen.findByText("tk-1"));
    await user.type(screen.getByLabelText("人工回复"), "已为您解决");
    await user.click(screen.getByRole("button", { name: /保存回复/ }));
    await waitFor(() => {
      expect(mockReply).toHaveBeenCalledWith("tk-1", { human_reply: "已为您解决", status: "processing" });
    });
  });

  it("完成工单：保存末条回复并置为已解决", async () => {
    mockList.mockResolvedValue(listResp([ticket({ ticket_id: "tk-1" })]));
    mockGet.mockResolvedValue(ticket({ ticket_id: "tk-1" }));
    mockReply.mockResolvedValue(ticket({ ticket_id: "tk-1", status: "resolved", human_reply: "最终回复" }));
    const user = userEvent.setup();
    render(<AdminHandoffs />);
    await user.click(await screen.findByText("tk-1"));
    await user.type(screen.getByLabelText("人工回复"), "最终回复");
    await user.click(screen.getByRole("button", { name: /完成工单/ }));
    await waitFor(() => {
      expect(mockReply).toHaveBeenCalledWith("tk-1", { human_reply: "最终回复", status: "resolved" });
    });
    expect((await screen.findAllByText("已解决")).length).toBeGreaterThan(0);
  });

  it("详情展示实时对话（用户提问与人工回复）", async () => {
    mockList.mockResolvedValue(listResp([ticket({ ticket_id: "tk-1" })]));
    mockGet.mockResolvedValue(
      ticket({
        ticket_id: "tk-1",
        status: "processing",
        live_messages: [
          { role: "user", content: "有没有优惠券" },
          { role: "human", content: "有的，新用户首单 9 折" },
        ],
      })
    );
    const user = userEvent.setup();
    render(<AdminHandoffs />);
    await user.click(await screen.findByText("tk-1"));
    expect(await screen.findByText("实时对话")).toBeInTheDocument();
    expect(screen.getByText("有没有优惠券")).toBeInTheDocument();
    expect(screen.getByText("有的，新用户首单 9 折")).toBeInTheDocument();
    expect(screen.getByText("人工客服")).toBeInTheDocument();
  });

  it("多轮回复记录完整展示在详情中", async () => {
    mockList.mockResolvedValue(listResp([ticket({ ticket_id: "tk-1" })]));
    mockGet.mockResolvedValue(
      ticket({
        ticket_id: "tk-1",
        status: "processing",
        replies: [
          { content: "第一条回复内容", created_at: "2026-01-01T00:00:00" },
          { content: "第二条回复内容", created_at: "2026-01-01T00:01:00" },
        ],
      })
    );
    const user = userEvent.setup();
    render(<AdminHandoffs />);
    await user.click(await screen.findByText("tk-1"));
    expect(await screen.findByText("第一条回复内容")).toBeInTheDocument();
    expect(screen.getByText("第二条回复内容")).toBeInTheDocument();
    expect(screen.getByText("人工回复记录（2）")).toBeInTheDocument();
  });

  it("401 未登录展示登录视图，登录成功后加载列表", async () => {
    mockList
      .mockRejectedValueOnce(new AdminApiError(401, "admin required"))
      .mockResolvedValueOnce(listResp([ticket({ ticket_id: "tk-1" })]));
    mockLogin.mockResolvedValueOnce({ admin_enabled: true });
    const user = userEvent.setup();
    render(<AdminHandoffs />);
    expect(await screen.findByText(/需要管理员登录/)).toBeInTheDocument();
    await user.type(screen.getByLabelText("管理员 Token"), "secret-token");
    await user.click(screen.getByRole("button", { name: /登录/ }));
    await waitFor(() => {
      expect(mockLogin).toHaveBeenCalledWith("secret-token");
    });
    expect(await screen.findByText("tk-1")).toBeInTheDocument();
  });

  it("403 未启用统一提示", async () => {
    mockList.mockRejectedValue(new AdminApiError(403, "disabled"));
    render(<AdminHandoffs />);
    expect(await screen.findByText(/403/)).toBeInTheDocument();
  });

  it("空列表展示“暂无工单”", async () => {
    mockList.mockResolvedValue(listResp([]));
    render(<AdminHandoffs />);
    expect(await screen.findByText("暂无工单")).toBeInTheDocument();
  });

  it("即使响应含 access_key 也绝不渲染", async () => {
    mockList.mockResolvedValue(listResp([ticket({ ticket_id: "tk-1", access_key: "ADMIN-SHOULD-NEVER-SHOW" } as any)]));
    mockGet.mockResolvedValue(ticket({ ticket_id: "tk-1", access_key: "ADMIN-SHOULD-NEVER-SHOW" } as any));
    const user = userEvent.setup();
    render(<AdminHandoffs />);
    await user.click(await screen.findByText("tk-1"));
    expect((await screen.findAllByText("排队中")).length).toBeGreaterThan(0);
    expect(document.body.textContent).not.toContain("ADMIN-SHOULD-NEVER-SHOW");
    expect(document.body.textContent).not.toContain("access_key");
    expect(document.body.textContent).not.toContain("access_key_hash");
  });
});