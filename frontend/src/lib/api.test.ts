import { afterEach, describe, expect, it, vi } from "vitest";
import { adminGetHandoff, adminListHandoffs, adminReplyHandoff } from "./api";

function jsonResponse(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(body),
  } as unknown as Response;
}

describe("管理端 api 鉴权（服务端会话 Cookie，不再发送 Token 头）", () => {
  const fetchMock = vi.fn();

  afterEach(() => {
    fetchMock.mockReset();
    vi.unstubAllGlobals();
  });

  it("列表请求携带 Cookie 凭据，且不发送任何令牌 / 鉴权请求头", async () => {
    vi.stubGlobal(
      "fetch",
      fetchMock.mockResolvedValue(
        jsonResponse(200, { total: 0, limit: 10, offset: 0, items: [] })
      )
    );

    await adminListHandoffs({ status: "pending" });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(init.credentials).toBe("include");
    const headers = (init.headers ?? {}) as Record<string, string>;
    // 仅允许 Content-Type，不得携带任何令牌或鉴权头
    expect(Object.keys(headers)).toEqual(["Content-Type"]);
  });

  it("详情/回复请求同样只靠 Cookie 凭据，不带任何令牌", async () => {
    vi.stubGlobal("fetch", fetchMock);
    fetchMock.mockResolvedValue(jsonResponse(200, {}));
    await adminGetHandoff("tk-1");

    fetchMock.mockResolvedValue(jsonResponse(200, {}));
    await adminReplyHandoff("tk-1", { human_reply: "ok", status: "resolved" });

    expect(fetchMock).toHaveBeenCalledTimes(2);
    for (const [, init] of fetchMock.mock.calls as [string, RequestInit][]) {
      expect(init.credentials).toBe("include");
      const headers = (init.headers ?? {}) as Record<string, string>;
      // 仅允许 Content-Type，不得携带任何令牌或鉴权头
      expect(Object.keys(headers)).toEqual(["Content-Type"]);
    }
  });

  it("401 抛出带状态码的 AdminApiError", async () => {
    vi.stubGlobal(
      "fetch",
      fetchMock.mockResolvedValue(jsonResponse(401, { safe_message: "需要有效的管理会话。" }))
    );

    await expect(adminListHandoffs()).rejects.toMatchObject({
      name: "AdminApiError",
      status: 401,
    });
  });
});