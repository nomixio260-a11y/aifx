# Self-hosted forecasting server: runs a cycle every few minutes and serves the web page.
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY aifx ./aifx
RUN pip install --no-cache-dir .
VOLUME ["/data"]
EXPOSE 8000
CMD ["aifx", "serve", "--state", "/data/state", "--site", "/data/site", "--host", "0.0.0.0", "--port", "8000", "--interval", "5"]
