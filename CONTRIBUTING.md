# Contributing

Thanks for contributing to **VSF Capstone – Image Processing & Retrieval**. This guide describes the shared workflow that keeps code, data and evaluation results consistent and reproducible.

Before you start, please read:

- [README.md](README.md) – project overview
- The design spec (Vietnamese): [`Thiết kế lọc, kiểm duyệt và truy xuất ảnh du lịch Việt Nam_ Cascade embedding và Hybrid Search.md`](<Thiết kế lọc, kiểm duyệt và truy xuất ảnh du lịch Việt Nam_ Cascade embedding và Hybrid Search.md>) – architecture, schema, roadmap (§10) and acceptance criteria
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) and [SECURITY.md](SECURITY.md)

## Table of contents

1. [Issues & feature requests](#1-issues--feature-requests)
2. [Repository layout](#2-repository-layout)
3. [Development setup](#3-development-setup)
4. [Git workflow](#4-git-workflow)
5. [Code style](#5-code-style)
6. [Testing & evaluation](#6-testing--evaluation)
7. [Data, models, secrets & cost](#7-data-models-secrets--cost)
8. [Pull requests & review](#8-pull-requests--review)

## 1. Issues & feature requests

- Search existing [issues](../../issues) first to avoid duplicates.
- Use the provided templates: **Bug report** or **Feature request**.
- For bugs, include reproduction steps, expected vs. actual behavior, logs, and versions (commit SHA, OS, GPU/CUDA if relevant).
- Planned work is tracked as issues grouped into milestones (`M0`–`M5`, `Backlog`) on the project board. Reference the spec section an issue implements (e.g. "spec §5.5").
- **Do not** open public issues for security vulnerabilities – follow [SECURITY.md](SECURITY.md).

## 2. Repository layout

```text
apps/web/                 # React + TS (Vite)
services/gateway/         # Go: static files, /thumbs, rate limiting
services/search/          # FastAPI: router, hybrid, rerank, kg, llm
pipeline/                 # ingest → download → hashing → embed → heads → ocr → caption → kg → index
eval/                     # test set, judge, retrieval/moderation eval, bake-off
bench/  loadtest/k6/      # benchmarks & load tests
deploy/                   # docker-compose, prometheus
docs/                     # architecture.md, performance.md, slides.md
data/sample/              # small committed fixture for tests & CI (the rest of data/ is git-ignored)
```

Put new code in the directory of the component it belongs to. If you need a new top-level component, explain why in the PR.

## 3. Development setup

Requirements: Python 3.12, [uv](https://github.com/astral-sh/uv), Go (version in `services/gateway/go.mod`), Node.js 22 (`apps/web/.nvmrc`), Docker + Docker Compose. An NVIDIA GPU (≥ 6 GB VRAM) is needed for `embed` and `train_heads`.

```bash
git clone https://github.com/lecuong1502/VSF_Capstone_Image_Processing_and_Retrieval.git
cd VSF_Capstone_Image_Processing_and_Retrieval
cp .env.example .env          # set RAW_DATA_DIR, GEMINI_API_KEY, ... (never commit .env)
make setup                    # create .env, install dependencies + pre-commit hooks (EMBED_MODEL weights download on the first make embed)
make lint test build-web      # must pass before opening a PR (same checks as CI)
```

Run the stack locally:

```bash
make pipeline                 # or: make restore (Qdrant snapshot + thumbnails)
docker compose -f deploy/docker-compose.yml up -d
```

## 4. Git workflow

- `main` is always green and **protected – no direct pushes**.
- Branch off the latest `main` using `<type>/<short-description>`:

  | Prefix | Use for | Example |
  |---|---|---|
  | `feat/` | new functionality | `feat/multi-hop-router` |
  | `fix/` | bug fixes | `fix/pdq-hash-overflow` |
  | `exp/` | experiments, model bake-offs | `exp/siglip2-vs-jina-clip` |
  | `docs/` | documentation | `docs/architecture-diagram` |
  | `chore/` | CI, build, dependencies | `chore/trivy-scan` |

- Follow [Conventional Commits](https://www.conventionalcommits.org/):

  ```text
  <type>(<scope>): <short imperative summary>

  feat(search): add RRF fusion for hybrid retrieval
  fix(pipeline): retry img2dataset on HTTP 429
  perf(embed): batch ONNX inference with FP16
  ```

  `type`: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `build`, `ci`, `chore`.
  Suggested `scope`: `web`, `gateway`, `search`, `pipeline`, `moderation`, `eval`, `bench`, `deploy`, `infra`, `docs`.

- Keep commits small and meaningful. Rebase onto `main` before requesting review; PRs are **squash-merged**.

## 5. Code style

**Python** (`services/search`, `pipeline`, `eval`)
- `ruff format` + `ruff check`; type-check with `mypy`. All of these run as pre-commit hooks; run them on everything with `uv run pre-commit run --all-files`.
- Type hints on all public functions; use `pathlib` and `logging` (no `print` in services).
- Pipeline steps must be **idempotent** (re-running never duplicates data) and configured via CLI flags or environment variables – no hard-coded paths.
- Manage dependencies with `uv` (`uv add` / `uv remove`), then run `make requirements` and commit `pyproject.toml`, `uv.lock` and `requirements.txt` together. pip users can install with `pip install -r requirements.txt`.

**Go** (`services/gateway`)
- Clean `gofmt` / `go vet`; `go test ./...` passes.

**TypeScript** (`apps/web`)
- ESLint + Prettier; `npm run build` with no type errors.

**General**
- No commented-out code, debug files, or notebooks with large outputs.
- Identifiers, comments and docs in English. User-facing UI text and query data are Vietnamese.

## 6. Testing & evaluation

- Every logic change needs unit tests (`pytest`, `go test`).
- Changes that affect **quality** (models, thresholds τ, router, reranker, Gemini prompts) must include before/after numbers on the test set in `eval/`:
  - Moderation: precision/recall/F1 (with 95% CI), precision(pass), escalation rate.
  - Retrieval: nDCG@10, Recall@10/50, MRR@10 per query type (single / multi-intent / multi-hop / negative).
- Changes that affect **performance** (search-api, gateway, index) must include k6 results: p50/p95/p99 per query type. Budget: **p95 < 3 s**, target p95 < 2 s.
- Docker image changes must report image size (search-api < 600 MB, gateway < 30 MB).
- Record seed, model version and commit SHA so results are reproducible.

## 7. Data, models, secrets & cost

- **Never commit**: images, raw CSV/Parquet, model weights (`*.onnx`, `*.pt`, `*.safetensors`), Qdrant snapshots, or anything under `data/` except `data/sample/`, nor `models/`.
- `data/sample/` may only contain landmark entities (no people) and must not contain annotator columns (`assigned_member`, `verified_at`).
- **Never commit secrets**: `.env`, API keys (Gemini, …). Add new variables to `.env.example` with dummy values.
- If a secret is committed by accident, revoke it immediately and notify the maintainer – rewriting Git history is not enough.
- **People in images**: face *detection* only (counting, face-area ratio, shot type). No face recognition, no face embeddings. Person-related queries are resolved via the knowledge graph and `entity_id`, never by looking at faces.
- **Gemini cost**: every Gemini call goes through the spend guard (`GEMINI_BUDGET_USD`), is cached by content hash, and uses the Batch API for offline jobs. PRs that spend money must report actual tokens and USD.
- Respect the copyright/licence of image sources and models; credit the source when adding a dataset or model.

## 8. Pull requests & review

1. Make sure `make lint test build-web` passes locally (the same checks CI runs).
2. Open a PR against `main` and fill in the [PR template](.github/pull_request_template.md).
3. Link the issue with `Closes #<number>`.
4. Keep PRs small (ideally < 400 changed lines excluding lockfiles); split large work.
5. CI must be green (lint, test, build, Trivy, smoke e2e) before merging.
6. Reviewers focus on correctness, acceptance criteria, eval/benchmark numbers, security and maintainability.
7. Reply to every comment; the comment author resolves the thread.

Thanks for contributing! 🚀
