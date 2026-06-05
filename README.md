# Production-Ready LangGraph API

A highly optimized, secure, and observable REST API built with **FastAPI**, **LangGraph**, and **Docker**. This template transforms a standard LLM agent into a robust microservice ready for enterprise deployment.

## ✨ Features

- **Robust LLM Orchestration**: Powered by LangGraph with built-in retry logic and automatic model fallbacks (e.g., switches from OpenAI to Gemini if an outage occurs).
- **Security Pipeline**: 
  - **Input Sanitization**: Detects and blocks prompt injection attacks before they reach the LLM.
  - **PII Masking**: Automatically detects and redacts emails, phone numbers, SSNs, and credit cards in both user inputs and agent outputs.
  - **Output Validation**: Filters potentially harmful LLM responses.
- **Performance Optimization**: 
  - **Caching Layer**: In-memory response caching with TTL to eliminate LLM latency on repeated questions and save token costs.
  - **Rate Limiting**: Protects endpoints against spam and DoS using `slowapi`.
- **Observability**: 
  - **LangSmith**: Full native tracing for all LangGraph agent invocations.
  - **Metrics & Logging**: Structured JSON logging and custom metrics (latency, token estimation, cache hit rates) accessible via endpoints.
- **DevOps Ready**: 
  - **Dockerized**: Highly optimized `Dockerfile` running as a secure, non-root user with health checks and layer caching.
  - **Infrastructure as Code**: Includes a `render.yml` for seamless, zero-downtime deployments to Render.
  - Built using **`uv`** for lightning-fast dependency resolution.

## 🚀 Getting Started

### Prerequisites
- [uv](https://github.com/astral-sh/uv) (Fast Python package manager)
- Docker & Docker Compose
- API Keys for OpenAI, Gemini, and LangChain (LangSmith)

### Environment Setup
Create a `.env` file in the root directory. *(Note: Your `.env` is already ignored in git to prevent accidental key leaks)*.
```env
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AI...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_PROJECT=prod-app
APP_ENV=development
LOG_LEVEL=INFO
RATE_LIMIT=20/minute
CACHE_TTL_SECONDS=300
MAX_RETRIES=3
```

### Running Locally (Without Docker)
1. Install dependencies:
   ```bash
   uv sync
   ```
2. Start the API:
   ```bash
   uv run uvicorn app.main:app --reload --port 8000
   ```

### Running with Docker (Recommended)
Build and spin up the production container:
```bash
docker compose up --build
```
*(On Windows, view the health check at `http://localhost:8000/health`, not `0.0.0.0`)*.

## 📡 API Endpoints

- **`POST /chat`**: The primary chat endpoint. Expects a JSON body: `{"message": "string", "thread_id": "string"}`.
- **`GET /health`**: Used by Docker/Kubernetes/Render to verify the server and internal components are operational.
- **`GET /metrics`**: Exposes application telemetry (requests, average latency, error rates, cache hit rates).
- **`GET /cache/stats`**: Shows the internal performance of the caching layer.

## 🧪 Testing
The repository includes fast, deterministic unit tests for the security and caching pipelines that do not require LLM calls.
```bash
uv run pytest
```
