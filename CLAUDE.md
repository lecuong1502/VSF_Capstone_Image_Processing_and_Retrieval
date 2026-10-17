# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

Vietnamese travel image moderation (cascade "embed once, use many") and hybrid retrieval (single / multi-intent / multi-hop / negative queries, p95 < 3 s). Solo capstone project.

- Design spec (Vietnamese, source of truth): `Thiết kế lọc, kiểm duyệt và truy xuất ảnh du lịch Việt Nam_ Cascade embedding và Hybrid Search.md`. Issues cite it as "spec §N".
- Work is tracked as GitHub issues in milestones `M0`–`M5` + `Backlog` on the project board (https://github.com/users/lecuong1502/projects/1). Each issue has acceptance criteria and a `Depends on:` line – check dependencies before starting.

## Language

- Chat with the user in **Vietnamese**.
- Everything committed or posted to GitHub (code, comments, docs, README, issues, PRs, commit messages) is in **English**.
- User-facing UI text, queries, captions and test-set data are Vietnamese.

## Layout

```text
apps/web/            React + Vite + TS (search UI, /review tool)
services/gateway/    Go gateway (distroless)
services/search/     FastAPI search-api – standalone uv project with its own uv.lock (built into Docker)
pipeline/            offline pipeline package (root uv project)
eval/                evaluation package (root uv project)
tests/               tests for pipeline/ and eval/
bench/ loadtest/k6/  benchmarks, k6 scripts
deploy/              docker-compose, prometheus
docs/                architecture, performance, slides
data/sample/         only committed data: ~50 landmark-only rows, no annotator columns
```

## Commands

```bash
make setup      # uv sync for the root project and services/search
make lint       # ruff check + ruff format --check + mypy
make test       # pytest
make fmt        # ruff format + ruff check --fix
make requirements  # regenerate requirements.txt files from uv.lock (lint fails if stale)
```

Python 3.12, dependencies managed only with `uv` (`uv add <pkg>`, never `pip install`). Run tools through `uv run`. `requirements.txt` files are generated from `uv.lock` for pip users – never edit them by hand.

To implement an issue, follow the repo skill `.claude/skills/work-on-issue/SKILL.md`. Go and Node toolchains are added to `make lint/test` by the tickets that create the gateway and web app.

## Conventions

- Branches `<type>/<desc>` (`feat/`, `fix/`, `exp/`, `docs/`, `chore/`); Conventional Commits; one issue ≈ one PR, `Closes #N` in the PR body; squash merge. Never push to `main`.
- Pipeline steps are idempotent and configured by CLI flags / env vars (see `.env.example`) – no hard-coded paths.
- Type hints on public functions, `logging` not `print` in services, `pathlib` for paths.
- Quality changes report eval numbers; performance changes report k6 p50/p95/p99 (see CONTRIBUTING §6).

## Decisions (do not re-litigate)

- **Data**: input is the six `*_verified.csv` files in `RAW_DATA_DIR` (default `~/Desktop/vinhomes-crawler/detect_human`). Scale target is **62,204 unique URLs** (not 80k). Labels keyed by `(image_url, entity_id)`; on conflict the newer export wins (`Cuong_entity_new_*` > `bo_sung_*` > `Cuong_phan_cong_*`) and conflicting pairs are flagged and excluded from training/eval; empty `verify` = `unlabeled`.
- **Never commit** anything under `data/` except `data/sample/`, model weights, `.env`, images, thumbnails, snapshots. The repo is public.
- **People**: face detection only – no face recognition, no face embeddings. Person/relation queries resolve via the KG and `entity_id`.
- **Gemini** (`gemini-3.5-flash-lite`, the only LLM): every call goes through the shared client with spend guard (`GEMINI_BUDGET_USD`, starting at $5), content-hash cache and Batch API for offline jobs. Report tokens and USD in PRs that spend money.
- **Models**: pipeline runs end-to-end with `jinaai/jina-clip-v2` first; bake-off vs. `siglip2-base` and `siglip2-so400m` later. Must fit a 6 GB RTX 4050.
- **Deployment**: local `docker compose` + images on GHCR. Public deployment is Backlog.
- **Test set**: full spec scale (150/100/60/30 queries, pool top-20, 20% human check).
