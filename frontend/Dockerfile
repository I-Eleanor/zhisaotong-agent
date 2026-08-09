# syntax=docker/dockerfile:1

# ---------- 构建阶段：用 Node 构建 React 静态产物 ----------
FROM node:20-alpine AS builder

WORKDIR /app

# 优先利用层缓存：先装依赖（需 package-lock.json，由 npm ci 精确还原）
COPY package.json package-lock.json ./
RUN npm ci

# 再复制源码并构建（产出 dist/）
COPY . .
RUN npm run build

# ---------- 运行阶段：nginx 提供静态文件并反代 /api ----------
FROM nginx:1.27-alpine

# 自定义配置：静态服务 + 将 /api 反代到 compose 中的 api 服务（8000）
COPY nginx.conf /etc/nginx/conf.d/default.conf

# 拷贝构建产物到 nginx 默认根目录
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
