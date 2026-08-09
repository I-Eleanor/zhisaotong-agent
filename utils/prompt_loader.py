from utils.config_handler import prompts_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger

# 进程内提示词缓存：key -> 文件内容，避免同一运行期反复读磁盘
_cache: dict[str, str] = {}


def _load_prompt(key: str, label: str) -> str:
    """读取并缓存一个提示词文件的内容。

    key:   prompts.yml 中的配置项名（如 "main_prompt_path"）
    label: 仅用于日志的中文名（如 "系统提示词"）
    """
    if key in _cache:
        return _cache[key]
    try:
        path = get_abs_path(prompts_conf[key])
    except KeyError as e:
        logger.error(f"[{label}] 在 yaml 配置项中没有 {key} 配置项")
        raise e
    try:
        content = open(path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[{label}] 解析提示词出错，{str(e)}")
        raise e
    _cache[key] = content
    return content


def load_system_prompts():
    return _load_prompt("main_prompt_path", "系统提示词")


def load_rag_prompts():
    return _load_prompt("rag_summarize_prompt_path", "RAG总结提示词")


def load_report_prompts():
    return _load_prompt("report_prompt_path", "报告生成提示词")


def load_diagnostic_plan_prompt():
    return _load_prompt("diagnostic_plan_prompt_path", "诊断计划提示词")


def load_diagnostic_replan_prompt():
    return _load_prompt("diagnostic_replan_prompt_path", "重规划提示词")


def load_diagnostic_report_prompt():
    return _load_prompt("diagnostic_report_prompt_path", "诊断报告提示词")


def load_orchestrator_prompt():
    return _load_prompt("orchestrator_prompt_path", "编排路由提示词")


if __name__ == '__main__':
    print(load_report_prompts())
