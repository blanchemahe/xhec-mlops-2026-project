.PHONY: install train api test

# Create the virtual environment and install the pre-commit hooks
install:
	uv sync
	uv run pre-commit install

# Train the model (needs data/abalone.csv, see the README)
train:
	uv run python -m src.modelling.main data/abalone.csv

# Run the API (http://127.0.0.1:8001/docs and /dashboard), train first
api:
	uv run uvicorn src.web_service.main:app --port 8001

test:
	uv run pytest
