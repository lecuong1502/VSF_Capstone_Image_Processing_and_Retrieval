SHELL := /bin/bash
.DEFAULT_GOAL := help

# uv projects: the root (pipeline + eval) and the standalone search-api.
PROJECTS := . services/search
# Ignore any activated virtualenv so each project always uses its own .venv.
UV := env -u VIRTUAL_ENV uv
EXPORT := $(UV) export --quiet --frozen --no-hashes --all-groups
GATEWAY := services/gateway
WEB := apps/web

.PHONY: help setup sync requirements lint lint-py lint-go lint-web fmt test test-py test-go test-web build-web \
	gemini-cost gemini-smoke ingest sample profile download dedup dedup-sheet embed embed-bench review gemini-label heads pipeline restore eval loadtest

help: ## List available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

setup: ## Create .env, install Python/Go/Node dependencies and git hooks
	@test -f .env || { cp .env.example .env && echo "Created .env from .env.example – fill in your values."; }
	@$(MAKE) --no-print-directory sync
	cd $(GATEWAY) && go mod download
	cd $(WEB) && npm ci
	@$(UV) run pre-commit install || echo "WARNING: pre-commit hooks not installed – run: uv run pre-commit install"
	@echo "EMBED_MODEL weights (FP16 ONNX) download to MODELS_DIR on the first make embed."

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

test: test-py test-go test-web ## Run all unit tests

test-py: ## pytest for each Python project
	@for p in $(PROJECTS); do $(UV) run --directory $$p pytest || exit 1; done

test-go: ## go test for the gateway
	cd $(GATEWAY) && go test ./...

test-web: ## vitest for the web app
	cd $(WEB) && npm test --silent

build-web: ## Type-check and build the web app
	cd $(WEB) && npm run build

gemini-cost: ## Print Gemini spend per step vs GEMINI_BUDGET_USD
	$(UV) run python -m pipeline.gemini cost

gemini-smoke: ## One real Gemini call (~$0.0001) to check the key, logging and pricing
	$(UV) run python -m pipeline.gemini smoke

ingest: ## Merge the verified CSVs from RAW_DATA_DIR into $DATA_DIR/ingest (Parquet + DuckDB)
	$(UV) run python -m pipeline.ingest

sample: ## Regenerate the committed data/sample/ fixture from RAW_DATA_DIR (landmarks only, no annotators)
	$(UV) run python -m pipeline.sample

profile: ## HTML data profile of the ingest tables → reports/data_profile.html
	$(UV) run python -m pipeline.profile

download: ## Download all images (img2dataset → WebDataset + WebP thumbs); re-run retries transient failures
	$(UV) run python -m pipeline.download

dedup: ## PDQ hashes + SHA/PDQ duplicate clusters → $DATA_DIR/dedup (re-run hashes only new images)
	$(UV) run python -m pipeline.hashing run

dedup-sheet: ## 300 stratified PDQ pairs to label → reports/dedup_pairs.html
	$(UV) run python -m pipeline.hashing label-sheet

embed: ## Embed every downloaded image with EMBED_MODEL (FP16 ONNX on GPU) → $DATA_DIR/embeddings/<model>/
	$(UV) run python -m pipeline.embed run

embed-bench: ## Images/s per batch size for EMBED_MODEL on this GPU
	$(UV) run python -m pipeline.embed bench

gemini-label: ## Gemini moderation labels for 5k images (paid, Batch API); pilot: python -m pipeline.gemini_label run --limit 100
	$(UV) run python -m pipeline.gemini_label run  # refuses until the pilot (--limit 100) has run

heads: ## Train, calibrate and apply the moderation heads → $DATA_DIR/heads/<version>/, $DATA_DIR/moderation/
	$(UV) run python -m pipeline.heads

review: build-web ## Serve the /review tool on http://127.0.0.1:8765/review (queues: python -m pipeline.review queue …)
	$(UV) run python -m pipeline.review serve

pipeline: ingest download dedup embed heads ## Run the offline pipeline (ingest → … → index)
	@echo "Steps after ingest are not implemented yet – see milestones M1/M2."

restore: ## Restore Qdrant snapshot + thumbnails
	@echo "Not implemented yet – see #20." && exit 1

eval: ## Run retrieval and moderation evaluation
	@echo "Not implemented yet – see #30." && exit 1

loadtest: ## Run k6 load tests and build the performance report
	@echo "Not implemented yet – see #32." && exit 1
