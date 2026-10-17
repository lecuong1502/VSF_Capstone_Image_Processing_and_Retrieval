SHELL := /bin/bash
.DEFAULT_GOAL := help

SEARCH_DIR := services/search
UV_SEARCH := env -u VIRTUAL_ENV uv run --directory $(SEARCH_DIR)
EXPORT := uv export --quiet --frozen --no-hashes --no-emit-project --all-groups

.PHONY: help setup requirements lint fmt test pipeline restore eval loadtest

help: ## List available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

setup: ## Install Python dependencies (root project + search-api)
	@test -f .env || { cp .env.example .env && echo "Created .env from .env.example – fill in your values."; }
	uv sync
	env -u VIRTUAL_ENV uv sync --directory $(SEARCH_DIR)

requirements: ## Regenerate requirements.txt files from uv.lock (run after uv add/remove)
	$(EXPORT) -o requirements.txt
	env -u VIRTUAL_ENV $(EXPORT) --directory $(SEARCH_DIR) -o requirements.txt

lint: ## Ruff (lint + format check), mypy, requirements.txt in sync with uv.lock
	@diff -q <($(EXPORT) | grep -v '^#') <(grep -v '^#' requirements.txt) >/dev/null || { echo "requirements.txt is stale – run: make requirements"; exit 1; }
	@diff -q <(env -u VIRTUAL_ENV $(EXPORT) --directory $(SEARCH_DIR) | grep -v '^#') <(grep -v '^#' $(SEARCH_DIR)/requirements.txt) >/dev/null || { echo "$(SEARCH_DIR)/requirements.txt is stale – run: make requirements"; exit 1; }
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
	$(UV_SEARCH) ruff check .
	$(UV_SEARCH) ruff format --check .
	$(UV_SEARCH) mypy

fmt: ## Auto-format and auto-fix lint issues
	uv run ruff format .
	uv run ruff check --fix .
	$(UV_SEARCH) ruff format .
	$(UV_SEARCH) ruff check --fix .

test: ## Run unit tests
	uv run pytest
	$(UV_SEARCH) pytest

pipeline: ## Run the offline pipeline (ingest → … → index)
	@echo "Not implemented yet – see milestones M1/M2 (#6 onwards)." && exit 1

restore: ## Restore Qdrant snapshot + thumbnails
	@echo "Not implemented yet – see #20." && exit 1

eval: ## Run retrieval and moderation evaluation
	@echo "Not implemented yet – see #30." && exit 1

loadtest: ## Run k6 load tests and build the performance report
	@echo "Not implemented yet – see #32." && exit 1
