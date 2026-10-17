SHELL := /bin/bash
.DEFAULT_GOAL := help

# uv projects: the root (pipeline + eval) and the standalone search-api.
PROJECTS := . services/search
# Ignore any activated virtualenv so each project always uses its own .venv.
UV := env -u VIRTUAL_ENV uv
EXPORT := $(UV) export --quiet --frozen --no-hashes --all-groups

.PHONY: help setup requirements lint fmt test pipeline restore eval loadtest

help: ## List available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

setup: ## Create .env, install Python dependencies and git hooks
	@test -f .env || { cp .env.example .env && echo "Created .env from .env.example – fill in your values."; }
	@for p in $(PROJECTS); do $(UV) sync --directory $$p || exit 1; done
	@test -d .git && $(UV) run pre-commit install || true
	@echo "ONNX weights are not downloaded yet – added with #11."

requirements: ## Regenerate requirements.txt files from uv.lock (run after uv add/remove)
	@for p in $(PROJECTS); do $(EXPORT) --directory $$p -o requirements.txt || exit 1; done

lint: ## Lockfiles + requirements.txt in sync, ruff (lint + format check), mypy
	@for p in $(PROJECTS); do \
	  $(UV) lock --quiet --check --directory $$p || { echo "$$p/uv.lock is stale – run: uv lock --directory $$p"; exit 1; }; \
	  diff -q <($(EXPORT) --directory $$p | grep -v '^#') <(grep -v '^#' $$p/requirements.txt) >/dev/null \
	    || { echo "$$p/requirements.txt is stale – run: make requirements"; exit 1; }; \
	done
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	@for p in $(PROJECTS); do $(UV) run --directory $$p mypy || exit 1; done

fmt: ## Auto-format and auto-fix lint issues
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

test: ## Run unit tests
	@for p in $(PROJECTS); do $(UV) run --directory $$p pytest || exit 1; done

pipeline: ## Run the offline pipeline (ingest → … → index)
	@echo "Not implemented yet – see milestones M1/M2 (#6 onwards)." && exit 1

restore: ## Restore Qdrant snapshot + thumbnails
	@echo "Not implemented yet – see #20." && exit 1

eval: ## Run retrieval and moderation evaluation
	@echo "Not implemented yet – see #30." && exit 1

loadtest: ## Run k6 load tests and build the performance report
	@echo "Not implemented yet – see #32." && exit 1
