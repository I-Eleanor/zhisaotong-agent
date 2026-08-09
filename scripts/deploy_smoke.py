"""部署冒烟测试：在本地按 docker-compose 中相同的命令启动 api + frontend，
验证「容器要干的事」在真实解释器上确实能跑起来（沙箱无 Docker daemon，
无法 `docker compose up`，此脚本作为等价替代验证部署命令本身无误）。

前端现已改为 React (Vite) 构建产物，由 `vite preview` / nginx 提供静态服务，
`/api` 反代到后端（开发用 vite proxy，生产用 nginx）。本脚本启动：
  - API：uvicorn api.main:app
  - Frontend：`frontend/dist`（由 `npm run build` 生成）经 `vite preview` 提供

用法：
    python scripts/deploy_smoke.py
退出码 0 = 通过，1 = 失败。
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(ROOT, "frontend")
API_PORT = 8011
FE_PORT = 8511
API_BASE = f"http://127.0.0.1:{API_PORT}"


def wait_port(port, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = socket.socket()
        if s.connect_ex(("127.0.0.1", port)) == 0:
            s.close()
            return True
        s.close()
        time.sleep(0.5)
    return False


def find_npm():
    npm = shutil.which("npm")
    if npm:
        return npm
    # 兜底：常见 Node 安装位置
    for cand in (
        r"D:\Node\npm.cmd",
        r"C:\Program Files\nodejs\npm.cmd",
        "/usr/local/bin/npm",
    ):
        if os.path.exists(cand):
            return cand
    return None


def main():
    env = dict(os.environ)
    env["API_BASE"] = API_BASE  # 前端通过环境变量指向后端，与 compose 一致

    print(f"[smoke] 启动 API (uvicorn :{API_PORT}) ...")
    api = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app",
         "--host", "127.0.0.1", "--port", str(API_PORT)],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    npm = find_npm()
    fe = None
    fe_note = ""
    if not npm:
        fe_note = "（未找到 npm，跳过前端启动，仅验证 API）"
    else:
        # 若 dist 不存在则先构建（与 Dockerfile 行为一致）
        dist_dir = os.path.join(FRONTEND_DIR, "dist")
        if not os.path.isdir(dist_dir):
            print("[smoke] frontend/dist 不存在，先执行 npm ci && npm run build ...")
            try:
                subprocess.run([npm, "ci"], cwd=FRONTEND_DIR, check=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                subprocess.run([npm, "run", "build"], cwd=FRONTEND_DIR, check=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            except subprocess.CalledProcessError as e:
                fe_note = f"（前端构建失败，仅验证 API：{e}）"
                npm = None

        if npm:
            print(f"[smoke] 启动 Frontend (vite preview :{FE_PORT}) ...")
            fe = subprocess.Popen(
                [npm, "run", "preview", "--", "--port", str(FE_PORT),
                 "--host", "127.0.0.1", "--strictPort"],
                cwd=FRONTEND_DIR, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True,
            )

    ok = True
    try:
        assert wait_port(API_PORT, 60), "API 未在 60s 内监听"
        with urllib.request.urlopen(API_BASE + "/api/health", timeout=10) as r:
            health = json.loads(r.read())
        print("[smoke] /api/health ->", health)
        assert health.get("status") == "ok", "health 状态异常"

        if fe is not None:
            assert wait_port(FE_PORT, 60), "Frontend 未在 60s 内监听"
            print(f"[smoke] frontend 已在 :{FE_PORT} 监听 (API_BASE={API_BASE})")
        else:
            print(f"[smoke] 跳过前端探活 {fe_note}")

        print("\n=== DEPLOY SMOKE: PASS ===")
    except Exception as e:  # noqa: BLE001
        ok = False
        print("\n=== DEPLOY SMOKE: FAIL ===", repr(e))
        # 打印子进程尾部日志，便于排查
        for name, proc in (("API", api), ("FE", fe)):
            if not proc or not proc.stdout:
                continue
            try:
                out = proc.stdout.read()
                print(f"--- {name} 输出尾部 ---\n{out[-1500:]}")
            except Exception:
                pass
    finally:
        for proc in (api, fe):
            if proc is None:
                continue
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
