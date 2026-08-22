"""真实密钥扫描审计测试（P1-12.1 安全阻断项，P1-12.2 调整）。

背景：.env 曾包含真实 DEEPSEEK_API_KEY（已清理并轮换）。现策略调整为：
本地 .env / .env.docker 允许存放真实密钥（本地运行文件，被 .gitignore 排除），
防线是「密钥不进仓库」——扫描范围以 Git 跟踪文件为准。

扫描范围：
- 所有 Git 跟踪文件（``git ls-files``），新增源码文件自动纳入；
- 额外强制纳入 ``.env.example``（模板必须占位符，即使被 Git 忽略）。

显式跳过（本地运行文件，允许真实密钥）：
- ``.env`` / ``.env.docker``：由 .gitignore 排除闸门测试（见
  ``test_env_local_files_are_gitignored``）保证它们不会进入仓库。

真实密钥形态：
- ``sk-`` + 32 位小写十六进制（DeepSeek / DashScope 实际格式）；
- ``sk-`` + 40 位以上连续字母数字（OpenAI 长密钥格式）。

测试中标记的伪造密钥（``sk-FACTORY-998877665544`` 等）含大写标记词与
连字符、数字段不足 32/40 位，天然不匹配真实密钥正则——豁免靠形态
不匹配而非按文件名跳过，避免豁免变成盲区。
"""
import re
import subprocess
from pathlib import Path

# 真实密钥形态：sk- 前缀 + 高熵连续串（无连字符/大写标记词）
_REAL_KEY_PATTERNS = re.compile(
    r"sk-[0-9a-f]{32}"       # DeepSeek / DashScope：sk- + 32 位小写十六进制
    r"|sk-[0-9a-zA-Z]{40,}"  # OpenAI 长密钥：sk- + 40+ 位连续字母数字
)

# 允许存放真实密钥的本地运行文件（不入库，由 .gitignore 闸门保证）
_LOCAL_ENV_FILES = {".env", ".env.docker"}

# 强制纳入扫描的模板文件（必须占位符）
_TEMPLATE_ENV_FILES = {".env.example"}


def _git_tracked_files(project_root: Path) -> list[Path]:
    """返回所有 Git 跟踪文件路径（.env 等被忽略文件天然不在其中）。"""
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=project_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return [
        (project_root / line.strip()).resolve()
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def _iter_scan_targets(project_root: Path):
    """扫描目标 = Git 跟踪文件 + .env.example（模板）。"""
    seen: set[Path] = set()
    for path in _git_tracked_files(project_root):
        if not path.is_file():
            continue
        seen.add(path)
        try:
            yield path, path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError):
            continue
    for name in _TEMPLATE_ENV_FILES:
        path = (project_root / name).resolve()
        if path.is_file() and path not in seen:
            yield path, path.read_text(encoding="utf-8")


def test_no_real_api_keys_in_git_tracked_files():
    """所有 Git 跟踪文件及 .env.example 中不得出现真实密钥形态。"""
    project_root = Path(__file__).resolve().parent.parent

    violations: list[str] = []
    for path, text in _iter_scan_targets(project_root):
        for match in _REAL_KEY_PATTERNS.finditer(text):
            line_no = text[:match.start()].count("\n") + 1
            violations.append(f"{path.relative_to(project_root).as_posix()}:{line_no}")

    assert not violations, (
        "发现疑似真实 API 密钥（sk- 前缀高熵串）进入仓库文件，安全阻断：\n"
        f"{violations}\n"
        "请立即轮换该密钥；仓库文件（含 .env.example）只允许占位符，"
        "真实密钥只能放在被 .gitignore 排除的本地 .env / .env.docker 中。"
    )


def test_env_example_contains_only_placeholders():
    """.env.example 模板必须全部为占位符（防止复制模板即泄漏）。"""
    project_root = Path(__file__).resolve().parent.parent

    for name in sorted(_TEMPLATE_ENV_FILES):
        env_path = project_root / name
        if not env_path.exists():
            continue
        for line_no, line in enumerate(env_path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                continue
            value = stripped.split("=", 1)[1].strip()
            if not value:
                continue  # 空值（如 API_TOKEN=）合法
            assert _REAL_KEY_PATTERNS.search(value) is None, (
                f"{name}:{line_no} 的值疑似真实密钥，模板必须使用占位符"
            )
            assert not value.startswith("sk-"), (
                f"{name}:{line_no} 的值以 sk- 开头（密钥形态），"
                "模板文件必须使用占位符（如 your_deepseek_api_key_here）"
            )


def test_env_local_files_are_gitignored():
    """本地 env 文件（允许存真实密钥）必须被 .gitignore 排除——防泄漏主闸门。"""
    project_root = Path(__file__).resolve().parent.parent

    for name in sorted(_LOCAL_ENV_FILES):
        if not (project_root / name).exists():
            continue  # 文件不存在时无需检查
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", name],
            cwd=project_root,
            capture_output=True,
        )
        assert result.returncode == 0, (
            f"{name} 未被 .gitignore 排除——该文件允许存放真实密钥，"
            "一旦被跟踪就会泄漏进仓库，必须加入 .gitignore"
        )
        # 双重确认：Git 确实未跟踪该文件
        tracked = subprocess.run(
            ["git", "ls-files", name],
            cwd=project_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        ).stdout.strip()
        assert not tracked, (
            f"{name} 已被 Git 跟踪（{tracked}）——"
            "请先从索引移除（git rm --cached）并确认历史中无密钥后轮换密钥"
        )
