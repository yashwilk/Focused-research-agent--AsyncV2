FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Create non-root user — required by Hugging Face Spaces
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy project files as root first
COPY pyproject.toml .
COPY uv.lock .
COPY src/ src/
COPY start.sh .

# Give appuser ownership of everything
RUN chown -R appuser:appuser /app

# Switch to non-root user BEFORE installing dependencies
USER appuser

# Now uv sync runs as appuser — no permission issues
RUN uv sync --frozen --no-dev

# Make start script executable
RUN chmod +x start.sh

# Expose Streamlit port
EXPOSE 7860

CMD ["./start.sh"]