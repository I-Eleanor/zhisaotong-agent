# 使用 Python 3.11 作为基础镜像（与升级计划保持一致，Python 3.10+）
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖（部分 Python 包编译需要 build-essential）
RUN sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件（优先利用镜像层缓存）
COPY requirements.txt .

# 设置 pip 镜像源并安装 Python 依赖（含 FastAPI / SSE / MCP / 测试；前端已迁至 React）
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir -r requirements.txt

# 复制项目文件（venv/、chroma_db/、logs/、前端遗留文件等由 .dockerignore 排除）
COPY . .

# 暴露 API(8000)；前端已迁移至 frontend/（React + nginx，独立镜像）
EXPOSE 8000

# 默认启动 API 服务；docker-compose 会按服务覆盖 command
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000"]
