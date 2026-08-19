#!/bin/bash

# -----------------------------------------------------------------------
# Focused Research Agent — Hugging Face Spaces startup script
#
# Starts FastAPI backend on port 8000 (internal) and Streamlit on
# port 7860 (the port Hugging Face Spaces exposes publicly).
# Both run in the same container. FastAPI starts first and Streamlit
# waits until the backend is ready before starting.
# -----------------------------------------------------------------------

set -e

echo "Starting Focused Research Agent..."

# Start FastAPI backend in the background on port 8000
uv run uvicorn \
    --factory focused_research_agent.api.app:create_app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 &

FASTAPI_PID=$!
echo "FastAPI started with PID $FASTAPI_PID"

# Wait for FastAPI to be ready before starting Streamlit
echo "Waiting for FastAPI to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "FastAPI is ready."
        break
    fi
    echo "Attempt $i/30 — waiting..."
    sleep 2
done

# Start Streamlit on port 7860 in the foreground
# HF Spaces requires the process to run in the foreground
echo "Starting Streamlit on port 7860..."
uv run streamlit run src/focused_research_agent/ui/Home.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false