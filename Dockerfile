# ---- Stage 1: build the React frontend into static files ----
FROM node:24-slim AS frontend
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: Python API that also serves the built frontend ----
FROM python:3.13-slim AS app
WORKDIR /app

# libgomp1 is the OpenMP runtime that xgboost links against.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt ./
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY src/ ./src/
COPY model/play_success_model.joblib ./model/play_success_model.joblib
COPY --from=frontend /frontend/dist ./frontend/dist

ENV PYTHONUNBUFFERED=1
# Render (and most hosts) inject $PORT; default to 8000 for local `docker run`.
EXPOSE 8000
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
