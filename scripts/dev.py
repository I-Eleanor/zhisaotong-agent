"""统一开发命令入口。

用法：
    python scripts/dev.py lint            # 代码检查（ruff）
    python scripts/dev.py test            # 运行测试
    python scripts/dev.py coverage        # 运行测试并生成覆盖率报告
    python scripts/dev.py frontend-build  # 构建前端
    python scripts/dev.py smoke           # 冒烟测试（MCP + API health）
    python scripts/dev.py check-all       # lint + test + coverage
"""
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")


def _run(cmd, cwd=None, check=True):
    print(f"\n> {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd or PROJECT_ROOT, check=check)


def cmd_lint():
    _run([sys.executable, "-m", "ruff", "check", "."])


def cmd_test():
    _run([sys.executable, "-m", "pytest", "tests/", "-v"])


def cmd_coverage():
    _run([sys.executable, "-m", "pytest", "tests/", "-v", "--cov=.", "--cov-report=term-missing"])


def cmd_frontend_build():
    npm = "npm.cmd" if sys.platform == "win32" else "npm"
    _run([npm, "install"], cwd=FRONTEND_DIR)
    _run([npm, "run", "build"], cwd=FRONTEND_DIR)


def cmd_smoke():
    _run([sys.executable, os.path.join(PROJECT_ROOT, "scripts", "mcp_smoke.py")], check=False)


def cmd_check_all():
    cmd_lint()
    cmd_test()
    cmd_coverage()


COMMANDS = {
    "lint": cmd_lint,
    "test": cmd_test,
    "coverage": cmd_coverage,
    "frontend-build": cmd_frontend_build,
    "smoke": cmd_smoke,
    "check-all": cmd_check_all,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("用法: python scripts/dev.py <command>")
        print(f"可用命令: {', '.join(COMMANDS)}")
        sys.exit(1)

    COMMANDS[sys.argv[1]]()


if __name__ == "__main__":
    main()
