.PHONY: help install test lint run up down logs seed

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: ## Install dev dependencies
	pip install -r requirements-dev.txt

test: ## Run unit tests — no Docker, no network, no services
	pytest --cov=app --cov-report=term-missing

test-integration: ## Run integration tests against LIVE Postgres/Mongo/Redis (make services first)
	pytest tests/integration

test-all: ## Unit + integration
	pytest tests/unit tests/integration

services: ## Start only the datastores, for integration tests
	docker compose up -d postgres mongo redis
	@echo "Waiting for health..."
	@sleep 8
	@echo "Now run: make test-integration"

lint: ## Lint and format-check
	ruff check app tests

run: ## Run the API locally with reload
	uvicorn app.main:app --reload --port 8000

MODEL ?= llama3.2:1b

up: ## Start the full stack (MODEL=llama3.2:1b by default)
	docker compose up -d --build
	@echo "Pulling $(MODEL) (first run only)..."
	docker compose exec ollama ollama pull $(MODEL)
	@echo "API docs -> http://localhost:8000/docs"

up-nollm: ## Start without Ollama — /search works, /query needs an LLM
	docker compose up -d --build api postgres mongo redis
	@echo "API docs -> http://localhost:8000/docs"

down: ## Stop the stack
	docker compose down

logs: ## Tail API logs
	docker compose logs -f api
