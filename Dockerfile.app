FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install the dependencies (without the dev ones)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH"

COPY src ./src
COPY data ./data
COPY bin ./bin

# Train the model when building the image, so the API has a model.pkl
RUN python -m src.modelling.main data/abalone.csv

# 8001: API, 4201: prefect UI
EXPOSE 8001 4201

CMD ["bash", "bin/run_services.sh"]
