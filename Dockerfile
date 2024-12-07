FROM python:3.12-slim-bullseye
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1

COPY . /app
WORKDIR /app

RUN uv sync --frozen --no-cache

ENV PATH="/app/.venv/bin:$PATH"
CMD ["fastapi", "run", "--host", "0.0.0.0", "app/main.py"]