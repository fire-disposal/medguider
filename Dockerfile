# 使用轻量级 Python 基础镜像
FROM python:3.13.4-slim-bookworm

# 从官方 uv 镜像中复制 uv 可执行文件
COPY --from=ghcr.io/astral-sh/uv:0.5.21 /uv /uvx /bin/

# 设置工作目录
WORKDIR /app

# 复制项目配置文件
COPY pyproject.toml uv.lock ./

# 安装项目依赖（使用 --frozen 确保版本一致，--no-install-project 优化缓存）
RUN uv sync --frozen --no-install-project

# 复制源代码及必要的资源
COPY src/ ./src/
COPY app.py ./
COPY README.md ./

# 设置环境变量，确保 uv run 能够找到同步的环境
ENV PATH="/app/.venv/bin:$PATH"

# 暴露 Streamlit 默认端口
EXPOSE 8501

# 健康检查
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 启动命令
ENTRYPOINT ["uv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
