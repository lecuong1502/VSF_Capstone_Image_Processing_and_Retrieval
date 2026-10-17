SHELL := /bin/bash
.DEFAULT_GOAL := help

# uv projects: the root (pipeline + eval) and the standalone search-api.
PROJECTS := . services/search
# Ignore any activated virtualenv so each project always uses its own .venv.
UV := env -u VIRTUAL_ENV uv
EXPORT := $(UV) export --quiet --frozen --no-hashes --all-groups
GATEWAY := services/gateway
WEB := apps/web

.PHONY: help setup sync requirements lint lint-py lint-go lint-web fmt test test-py test-go build-web \
	gemini-cost gemini-smoke ingest pipeline restore eval loadtest

help: ## List available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

setup: ## Create .env, install Python/Go/Node dependencies and git hooks
	@test -f .env || { cp .env.example .env && echo "Created .env from .env.example – fill in your values."; }
	@$(MAKE) --no-print-directory sync
	cd $(GATEWAY) && go mod download
	cd $(WEB) && npm ci
	@$(UV) run pre-commit install || echo "WARNING: pre-commit hooks not installed – run: uv run pre-commit install"
	@echo "ONNX weights are not downloaded yet – added with #11."

sync: ## Install locked Python dependencies for every uv project (used by CI)
	@for p in $(PROJECTS); do $(UV) sync --locked --directory $$p || exit 1; done

requirements: ## Regenerate requirements.txt files from uv.lock (run after uv add/remove)
	@for p in $(PROJECTS); do $(EXPORT) --directory $$p -o requirements.txt || exit 1; done

lint: lint-py lint-go lint-web ## Lint everything (Python, Go, web)

lint-py: ## Lockfiles + requirements.txt in sync, ruff (lint + format check), mypy
	@for p in $(PROJECTS); do \
	  $(UV) lock --quiet --check --directory $$p || { echo "$$p/uv.lock is stale – run: uv lock --directory $$p"; exit 1; }; \
	  diff -q <($(EXPORT) --directory $$p | grep -v '^#') <(grep -v '^#' $$p/requirements.txt) >/dev/null \
	    || { echo "$$p/requirements.txt is stale – run: make requirements"; exit 1; }; \
	done
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	@for p in $(PROJECTS); do $(UV) run --directory $$p mypy || exit 1; done

lint-go: ## gofmt check + go vet
	@test -z "$$(gofmt -l $(GATEWAY))" || { gofmt -l $(GATEWAY); echo "Go files need gofmt"; exit 1; }
	cd $(GATEWAY) && go vet ./...

lint-web: ## ESLint + Prettier check
	cd $(WEB) && npm run -s lint && npm run -s format:check

fmt: ## Auto-format and auto-fix lint issues (Python, Go, web)
	$(UV) run ruff format .
	$(UV) run ruff check --fix .
	gofmt -w $(GATEWAY)
	cd $(WEB) && npm run -s format

test: test-py test-go ## Run all unit tests

test-py: ## pytest for each Python project
	@for p in $(PROJECTS); do $(UV) run --directory $$p pytest || exit 1; done

test-go: ## go test for the gateway
	cd $(GATEWAY) && go test ./...

build-web: ## Type-check and build the web app
	cd $(WEB) && npm run build

gemini-cost: ## Print Gemini spend per step vs GEMINI_BUDGET_USD
	$(UV) run python -m pipeline.gemini cost

gemini-smoke: ## One real Gemini call (~$0.0001) to check the key, logging and pricing
	$(UV) run python -m pipeline.gemini smoke

ingest: ## Merge the verified CSVs from RAW_DATA_DIR into $DATA_DIR/ingest (Parquet + DuckDB)
	$(UV) run python -m pipeline.ingest

pipeline: ingest ## Run the offline pipeline (ingest → … → index)
	@echo "Steps after ingest are not implemented yet – see milestones M1/M2."

restore: ## Restore Qdrant snapshot + thumbnails
	@echo "Not implemented yet – see #20." && exit 1

eval: ## Run retrieval and moderation evaluation
	@echo "Not implemented yet – see #30." && exit 1

loadtest: ## Run k6 load tests and build the performance report
	@echo "Not implemented yet – see #32." && exit 1
