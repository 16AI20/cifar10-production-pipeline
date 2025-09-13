# Base image
FROM python:3.10-slim-bookworm

# Upgrade all packages to their latest security-patched versions
RUN apt-get update && apt-get install -y git curl && \
    apt-get upgrade -y && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Set working directory inside the container
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock* ./
COPY .python-version ./
COPY run.sh ./
COPY conf/ ./conf/
COPY src/ ./src/

# Install dependencies with uv
RUN uv sync --frozen

# Make run.sh executable
RUN chmod +x run.sh

# Set entrypoint to allow arg-passing
ENTRYPOINT ["./run.sh"]
