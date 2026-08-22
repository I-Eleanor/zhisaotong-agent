# 架构验收说明（P1-17）

最终验收锁定的跨模块契约。测试入口：`tests/test_architecture_acceptance.py`（复用 conftest 桩与夹具，零真实模型 / 网络 / 本地向量库依赖）。

## 1. 启动契约

- `lifespan` 在容器创建前执行 `validate_startup()`：纯配置检查（环境变量 / 路径 / 模型配置 / CORS 格式），零重型资源加载；
- 校验失败 → 记 `startup_config_validation_failed` 安全日志（`safe_exception_fields` 形态）→ 异常原样抛出阻止启动，不构造、不挂载容器；
- 校验通过 → 新建 `AppContainer` 挂载 `app.state`（构造零成本，资源全懒加载）；
- `/api/health/live`：无任何依赖（无 Depends / 容器 / 模型），进程存活即 200；
- `/api/health/ready`：容器 OPEN 且依赖检查通过 → 200；未挂载 / CLOSING / CLOSED / 检查失败 / 超时 → 安全 503。

## 2. 依赖边界契约

- API 路由只经 `Depends(get_app_container)` / `Depends(get_mounted_container)` 取容器，**不调用**模块级 `get_orchestrator` / `get_chat_model` 等全局单例 getter（旧入口仅限非 API 路径）；
- `get_app_container`（业务路由）：挂载 + OPEN 双检查——非 OPEN 容器上的业务请求统一 503 并记 `container_not_ready` 日志，**绝不经懒加载隐式重建资源**（容器关闭后重建只能显式发生）；
- `get_mounted_container`（就绪探针）：只检查挂载，状态语义由探针自身输出（`status` / `checks` 安全结构）；
- 容器生命周期状态机 `OPEN → CLOSING → CLOSED`：CLOSING 期间资源访问抛 `ContainerStateError`；CLOSED 后访问显式重建回到 OPEN。

## 3. 错误与日志契约

- API 错误响应恒为三字段安全结构 `{error_code, safe_message, request_id}`；异常原文、内部类型、路径、密钥不出现在任何响应中；
- SSE 协议：生产线程异常 → 安全 `error` 事件（`STREAM_FAILED` + 固定文案）→ 流必然终止（done 走独立控制队列，永不丢失、不重复）；
- 日志全量结构化 dict（无 f-string 拼接，静态扫描锁定）；异常摘要统一 `safe_exception_fields`；密钥 / 绝对路径 / 用户原始输入 / traceback 不进日志；
- 错误码只增不改（`utils/error_codes.py`），未知码经 `normalize_error_code` 回退 `INTERNAL_ERROR`。

## 4. 审计结论（P1-17）

- 修复一个真实问题：`get_app_container` 原先只检查挂载不检查状态，CLOSED 容器上的业务请求会经懒加载隐式重建全部资源（游离于生命周期之外，泄漏）→ 加 OPEN 守卫修复；
- 其余审计项（路由 getter 边界、三字段结构、SSE 协议、日志脱敏、启动零成本）均无问题，由验收测试锁定。
