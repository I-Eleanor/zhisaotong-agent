import { renderHook, waitFor, act } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { useHandoff } from "./useHandoff";
import type { HandoffTicket } from "@/lib/types";

vi.mock("@/lib/api", () => ({
  createHandoffTicket: vi.fn(),
  fetchHandoffTicket: vi.fn(),
}));

import { createHandoffTicket, fetchHandoffTicket } from "@/lib/api";

const mockedCreate = vi.mocked(createHandoffTicket);
const mockedFetch = vi.mocked(fetchHandoffTicket);

function makeTicket(overrides: Partial<HandoffTicket> = {}): HandoffTicket {
  return {
    ticket_id: "tk-123",
    status: "pending",
    user_question: "设备故障",
    access_key: "secret-key-abc",
    ...overrides,
  };
}

function render(confirmValue: boolean) {
  return renderHook(() =>
    useHandoff({
      confirm: () => confirmValue,
    })
  );
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe("useHandoff", () => {
  it("确认后创建工单，保存 ticket_id 与 access_key（会话内），并置为不可再申请", async () => {
    mockedCreate.mockResolvedValue(makeTicket());
    const { result } = render(true);

    let ok = false;
    await act(async () => {
      ok = await result.current.requestHandoff({ user_question: "设备故障" });
    });

    expect(ok).toBe(true);
    expect(mockedCreate).toHaveBeenCalledTimes(1);
    expect(result.current.state.ticketId).toBe("tk-123");
    expect(result.current.state.status).toBe("pending");
    expect(result.current.state.hasAccessKey).toBe(true);
    // access_key 只通过会话内存暴露，绝不进入 notice/状态展示
    expect(result.current.state.notice).not.toContain("secret-key-abc");
    expect(result.current.getAccessKey()).toBe("secret-key-abc");
    expect(result.current.state.canRequest).toBe(false);
  });

  it("重复点击拦截：已有工单时再次请求不重复建单", async () => {
    mockedCreate.mockResolvedValue(makeTicket());
    const { result } = render(true);
    await act(async () => {
      await result.current.requestHandoff({ user_question: "q1" });
    });
    expect(mockedCreate).toHaveBeenCalledTimes(1);

    let ok = true;
    await act(async () => {
      ok = await result.current.requestHandoff({ user_question: "q2" });
    });
    expect(ok).toBe(false);
    expect(mockedCreate).toHaveBeenCalledTimes(1); // 未二次建单
    expect(result.current.state.error).toContain("请勿重复提交");
  });

  it("用户取消确认则不创建工单", async () => {
    const { result } = render(false);
    let ok = true;
    await act(async () => {
      ok = await result.current.requestHandoff({ user_question: "设备故障" });
    });
    expect(ok).toBe(false);
    expect(mockedCreate).not.toHaveBeenCalled();
    expect(result.current.state.canRequest).toBe(true);
  });

  it("创建失败返回统一错误且可重试", async () => {
    mockedCreate.mockRejectedValue(new Error("数据库繁忙"));
    const { result } = render(true);
    let ok = true;
    await act(async () => {
      ok = await result.current.requestHandoff({ user_question: "设备故障" });
    });
    expect(ok).toBe(false);
    expect(result.current.state.error).toContain("人工转接失败");
    expect(result.current.state.ticketId).toBeNull();
    // 失败后可再次尝试
    expect(result.current.state.canRequest).toBe(true);
  });

  it("handoff_created 事件保存 ticket_id/access_key/status（仅会话内存，不进入状态展示）", () => {
    const { result } = render(true);
    act(() => {
      result.current.handleEvent({
        type: "handoff_created",
        agent: "orchestrator",
        content: "人工工单已创建",
        data: { ticket_id: "tk-sse", access_key: "k-sse-323232", status: "pending", message: "人工工单已创建" },
      });
    });
    expect(result.current.state.ticketId).toBe("tk-sse");
    expect(result.current.state.status).toBe("pending");
    expect(result.current.state.notice).toBe("人工工单创建成功：工单号 tk-sse。");
    // access_key 保存于会话内存（可刷新用），但不进入可渲染状态
    expect(result.current.state.hasAccessKey).toBe(true);
    expect(result.current.getAccessKey()).toBe("k-sse-323232");
    expect(JSON.stringify(result.current.state)).not.toContain("k-sse-323232");
    expect(JSON.stringify(result.current.state)).not.toContain("access_key");
  });

  it("SSE 自动建单凭证可用于刷新状态", async () => {
    const { result } = render(true);
    act(() => {
      result.current.handleEvent({
        type: "handoff_created",
        data: { ticket_id: "tk-sse", access_key: "k-sse-refresh", status: "pending", message: "x" },
      });
    });
    mockedFetch.mockResolvedValue(makeTicket({ status: "processing", human_reply: "在处理" }));
    await act(async () => {
      await result.current.refreshStatus();
    });
    await waitFor(() => {
      expect(mockedFetch).toHaveBeenCalledWith("tk-sse", "k-sse-refresh");
      expect(result.current.state.status).toBe("processing");
      expect(result.current.state.reply).toBe("在处理");
    });
  });

  it("刷新页面后凭证清除：新会话（重挂载）初始无 access_key", () => {
    const { result } = render(true);
    expect(result.current.getAccessKey()).toBeNull();
    expect(result.current.state.hasAccessKey).toBe(false);
    expect(result.current.state.ticketId).toBeNull();
  });

  it("human_replied 事件展示人工回复并更新处理中状态", () => {
    const { result } = render(true);
    act(() => {
      result.current.handleEvent({
        type: "human_replied",
        content: "请稍等，我已接手处理。",
        data: { status: "processing" },
      });
    });
    expect(result.current.state.reply).toBe("请稍等，我已接手处理。");
    expect(result.current.state.status).toBe("processing");
  });

  it("handoff_resolved 事件将状态更新为已解决", () => {
    const { result } = render(true);
    act(() => {
      result.current.handleEvent({
        type: "handoff_resolved",
        content: "问题已解决",
        data: { status: "resolved" },
      });
    });
    expect(result.current.state.status).toBe("resolved");
  });

  it("access_key 不进入任何可渲染状态（notion/error/reply）", async () => {
    mockedCreate.mockResolvedValue(makeTicket({ access_key: "TOP-SECRET-KEY" }));
    const { result } = render(true);
    await act(async () => {
      await result.current.requestHandoff({ user_question: "设备故障" });
    });
    const json = JSON.stringify(result.current.state);
    expect(json).not.toContain("TOP-SECRET-KEY");
  });

  it("refreshStatus 用会话 access_key 拉取最新状态", async () => {
    mockedCreate.mockResolvedValue(makeTicket());
    const { result } = render(true);
    await act(async () => {
      await result.current.requestHandoff({ user_question: "设备故障" });
    });

    mockedFetch.mockResolvedValue(makeTicket({ status: "processing", human_reply: "我在处理" }));
    await act(async () => {
      await result.current.refreshStatus();
    });
    await waitFor(() => {
      expect(mockedFetch).toHaveBeenCalledWith("tk-123", "secret-key-abc");
      expect(result.current.state.status).toBe("processing");
      expect(result.current.state.reply).toBe("我在处理");
    });
  });

  it("suggestHandoff 可设置提示（sync need_handoff 场景）", () => {
    const { result } = render(true);
    act(() => {
      result.current.suggestHandoff("您可能需要人工协助");
    });
    expect(result.current.state.notice).toBe("您可能需要人工协助");
  });
});