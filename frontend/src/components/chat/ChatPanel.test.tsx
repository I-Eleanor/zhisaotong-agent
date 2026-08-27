import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { ChatPanel } from "./ChatPanel";
import type { AgentEvent } from "@/lib/types";

vi.mock("@/lib/api", () => ({
  streamAgent: vi.fn(),
  syncChat: vi.fn(),
  createHandoffTicket: vi.fn(),
  fetchHandoffTicket: vi.fn(),
}));

import { streamAgent, syncChat, createHandoffTicket } from "@/lib/api";

const mockedStream = vi.mocked(streamAgent);
const mockedSync = vi.mocked(syncChat);
const mockedCreate = vi.mocked(createHandoffTicket);

beforeEach(() => {
  vi.clearAllMocks();
  mockedStream.mockReset();
  mockedSync.mockReset();
  mockedCreate.mockReset();
  vi.spyOn(window, "confirm").mockReturnValue(true);
});

async function sendViaEnter(user: ReturnType<typeof userEvent.setup>, question: string) {
  await user.type(screen.getByPlaceholderText(/输入您的问题/), `${question}{enter}`);
}

describe("ChatPanel 转人工", () => {
  it("普通问答回归：发送后渲染回答并清空输入", async () => {
    mockedStream.mockImplementation(async (_ep, _body, onEvent) => {
      onEvent({ type: "message", agent: "conversation", content: "这是普通回答。" });
      onEvent({ type: "done", agent: "conversation" });
    });
    const user = userEvent.setup();
    render(<ChatPanel />);
    await sendViaEnter(user, "今天天气如何");
    await screen.findByText("这是普通回答。");
    expect(screen.getByPlaceholderText(/输入您的问题/)).toHaveValue("");
  });

  it("人工接管期间模型停手：仅提示由人工服务，不产出模型回答", async () => {
    mockedStream.mockImplementation(async (_ep, _body, onEvent) => {
      onEvent({ type: "handoff_human_service", content: "您当前已转人工服务…" });
      onEvent({ type: "done", agent: "orchestrator" });
    });
    const user = userEvent.setup();
    render(<ChatPanel />);
    await sendViaEnter(user, "有没有优惠券");
    expect(await screen.findByText(/您当前已转人工服务/)).toBeInTheDocument();
    // 模型不得参与：不应渲染任何机器人正常回答
    expect(screen.queryByText("这是普通回答。")).not.toBeInTheDocument();
    expect(screen.getByPlaceholderText(/输入您的问题/)).toHaveValue("");
  });

  it("用户确认后创建工单，展示工单号，且 access_key 不进入消息", async () => {
    mockedStream.mockResolvedValue(undefined);
    mockedCreate.mockResolvedValue({
      ticket_id: "tk-c1",
      status: "pending",
      user_question: "设备故障",
      access_key: "SUPER-SECRET-KEY",
    } as never);
    const user = userEvent.setup();
    render(<ChatPanel />);
    // 先有一条用户消息作为工单主题
    await sendViaEnter(user, "设备故障");
    await waitFor(() => expect(screen.getByPlaceholderText(/输入您的问题/)).toHaveValue(""));

    await user.click(screen.getByLabelText(/转人工客服/));
    await screen.findByText(/工单号：tk-c1/);
    expect(mockedCreate).toHaveBeenCalledTimes(1);
    expect(mockedCreate).toHaveBeenCalledWith(
      expect.objectContaining({ user_question: "设备故障" })
    );
    // access_key 绝不出现在任何可读文本中
    expect(screen.queryByText(/SUPER-SECRET-KEY/)).not.toBeInTheDocument();
    expect(document.body.textContent).not.toContain("SUPER-SECRET-KEY");
  });

  it("重复点击拦截：建单成功后按钮禁用，不再二次建单", async () => {
    mockedStream.mockResolvedValue(undefined);
    mockedCreate.mockResolvedValue({
      ticket_id: "tk-c2",
      status: "pending",
      user_question: "x",
      access_key: "K2",
    } as never);
    const user = userEvent.setup();
    render(<ChatPanel />);
    const btn = screen.getByLabelText(/转人工客服/);
    await user.click(btn);
    await screen.findByText(/工单号：tk-c2/);
    // 建单成功 → canRequest=false → 按钮禁用，且再次触发不建单
    expect(btn).toBeDisabled();
    await user.click(btn).catch(() => undefined);
    expect(mockedCreate).toHaveBeenCalledTimes(1);
  });

  it("创建失败展示错误提示且可重试", async () => {
    mockedStream.mockResolvedValue(undefined);
    mockedCreate.mockRejectedValueOnce(new Error("服务不可用"));
    const user = userEvent.setup();
    render(<ChatPanel />);
    await user.click(screen.getByLabelText(/转人工客服/));
    await screen.findByText(/人工转接失败/);
    expect(screen.getByText(/人工转接失败/)).toBeInTheDocument();
    // 失败后仍可重试（按钮未因禁用被永久锁死）
    expect(screen.getByLabelText(/转人工客服/)).not.toBeDisabled();
  });

  it("handoff_created 事件展示工单与排队中状态，且 access_key 不进入页面文本", async () => {
    mockedStream.mockImplementation(async (_ep, _body, onEvent) => {
      onEvent({
        type: "handoff_created",
        agent: "orchestrator",
        content: "人工工单已创建",
        data: { ticket_id: "tk-sse-9", access_key: "ssetopsecret99", status: "pending", message: "人工工单已创建" },
      } as AgentEvent);
    });
    const user = userEvent.setup();
    render(<ChatPanel />);
    await sendViaEnter(user, "我要人工");
    expect(await screen.findByText("tk-sse-9")).toBeInTheDocument();
    expect(await screen.findByText(/排队中/)).toBeInTheDocument();
    // 自动建单拿到凭证 → 显示「刷新状态」按钮；且凭证绝不渲染到页面
    expect(await screen.findByText(/刷新状态/)).toBeInTheDocument();
    expect(document.body.textContent).not.toContain("ssetopsecret99");
  });

  it("human_replied 事件展示人工回复", async () => {
    mockedStream.mockImplementation(async (_ep, _body, onEvent) => {
      onEvent({
        type: "handoff_created",
        agent: "orchestrator",
        content: "已创建",
        data: { ticket_id: "tk-h", status: "pending", message: "已创建" },
      } as AgentEvent);
      onEvent({
        type: "human_replied",
        content: "您好，已为您处理中。",
        data: { status: "processing" },
      } as AgentEvent);
    });
    const user = userEvent.setup();
    render(<ChatPanel />);
    await sendViaEnter(user, "等待人工");
    expect(await screen.findByText("您好，已为您处理中。")).toBeInTheDocument();
    expect(await screen.findByText(/人工处理中/)).toBeInTheDocument();
  });

  it("handoff_resolved 事件更新状态为已解决", async () => {
    mockedStream.mockImplementation(async (_ep, _body, onEvent) => {
      onEvent({
        type: "handoff_created",
        agent: "orchestrator",
        content: "已创建",
        data: { ticket_id: "tk-r", status: "pending", message: "已创建" },
      } as AgentEvent);
      onEvent({
        type: "handoff_resolved",
        content: "已解决",
        data: { status: "resolved" },
      } as AgentEvent);
    });
    const user = userEvent.setup();
    render(<ChatPanel />);
    await sendViaEnter(user, "问题已解决");
    expect(await screen.findByText("已解决")).toBeInTheDocument();
  });

  it("need_handoff 展示：同步兜底返回 need_handoff 时给出人工提示", async () => {
    mockedStream.mockRejectedValue(new Error("stream down"));
    mockedSync.mockResolvedValue({ answer: "同步回答", need_handoff: true } as never);
    const user = userEvent.setup();
    render(<ChatPanel />);
    await sendViaEnter(user, "很复杂的问题");
    expect(await screen.findByText(/可能需要人工协助/)).toBeInTheDocument();
  });
});