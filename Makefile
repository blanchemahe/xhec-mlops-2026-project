.PHONY: install train api test lint docker

# Create the virtual environment and install the pre-commit hooks
install:
	uv sync
	uv run pre-commit install

# Train the model on data/abalone.csv
train:
	uv run python -m src.modelling.main data/abalone.csv

# Run the API (http://127.0.0.1:8001/docs and /dashboard), train first
api:
	uv run uvicorn src.web_service.main:app --port 8001

test:
	uv run pytest

# Same checks as the CI (ruff, formatting, notebooks)
lint:
	uv run pre-commit run --all-files

# Build the image (the model is trained at build) and run it: API on :8000, Prefect UI on :4200
docker:
	docker build -f Dockerfile.app -t abalone-app .
	docker run -p 0.0.0.0:8000:8001 -p 0.0.0.0:4200:4201 abalone-app
