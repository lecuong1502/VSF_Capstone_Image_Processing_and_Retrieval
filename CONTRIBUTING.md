# Hướng dẫn đóng góp

Cảm ơn bạn đã đóng góp cho **VSF Capstone – Image Processing & Retrieval**. Tài liệu này mô tả quy trình làm việc chung để code, dữ liệu và kết quả đánh giá luôn nhất quán, tái lập được.

Trước khi bắt đầu, hãy đọc:

- [README.md](README.md) – tổng quan dự án
- Tài liệu thiết kế: [`Thiết kế lọc, kiểm duyệt và truy xuất ảnh du lịch Việt Nam_ Cascade embedding và Hybrid Search.md`](<Thiết kế lọc, kiểm duyệt và truy xuất ảnh du lịch Việt Nam_ Cascade embedding và Hybrid Search.md>) – kiến trúc, schema, roadmap (mục 10) và tiêu chí nghiệm thu
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) và [SECURITY.md](SECURITY.md)

## Mục lục

1. [Báo lỗi & đề xuất tính năng](#1-báo-lỗi--đề-xuất-tính-năng)
2. [Cấu trúc repo](#2-cấu-trúc-repo)
3. [Thiết lập môi trường](#3-thiết-lập-môi-trường)
4. [Quy trình Git](#4-quy-trình-git)
5. [Quy ước code](#5-quy-ước-code)
6. [Kiểm thử & đánh giá](#6-kiểm-thử--đánh-giá)
7. [Dữ liệu, model & bí mật](#7-dữ-liệu-model--bí-mật)
8. [Pull Request & review](#8-pull-request--review)

## 1. Báo lỗi & đề xuất tính năng

- Tìm trong [Issues](../../issues) trước để tránh trùng lặp.
- Dùng template có sẵn: **Bug report** hoặc **Feature request**.
- Với lỗi: ghi rõ bước tái hiện, kết quả mong đợi/thực tế, log, phiên bản (commit SHA, OS, GPU/CUDA nếu liên quan).
- Với task trong roadmap: ghi số task (ví dụ `Task #6`) và tiêu chí nghiệm thu tương ứng.
- Lỗ hổng bảo mật **không** mở issue công khai – làm theo [SECURITY.md](SECURITY.md).

## 2. Cấu trúc repo

```text
apps/web/                 # React + TS (Vite)
services/gateway/         # Go: static, /thumbs, rate-limit
services/search/          # FastAPI: router, hybrid, rerank, kg, llm
pipeline/                 # ingest → download → hashing → embed → heads → ocr → caption → kg → index
eval/                     # testset, judge, retrieval/moderation eval, bake-off
bench/  loadtest/k6/      # benchmark & load test
deploy/                   # docker-compose, cloudflared, prometheus
docs/                     # slides.md, architecture.md, performance.md
```

Đặt code mới đúng thư mục theo thành phần. Nếu cần thư mục/thành phần mới, nêu lý do trong PR.

## 3. Thiết lập môi trường

Yêu cầu: Python 3.12, [uv](https://github.com/astral-sh/uv), Go (bản ổn định mới nhất), Node.js LTS, Docker + Docker Compose. GPU NVIDIA là tuỳ chọn (cần cho `embed`, `train_heads`).

```bash
git clone <repo-url> && cd VSF_Capstone_Image_Processing_and_Retrieval
cp .env.example .env          # điền GEMINI_API_KEY, ... (KHÔNG commit .env)
make setup                    # cài dependency + tải weights ONNX
pre-commit install            # bật hook lint/format trước khi commit
make lint test                # phải xanh trước khi mở PR
```

Chạy hệ thống local:

```bash
make restore                  # hoặc: make pipeline
docker compose -f deploy/docker-compose.yml up -d
```

## 4. Quy trình Git

- Nhánh chính: `main` – luôn ở trạng thái build/test xanh, **không push trực tiếp**.
- Tạo nhánh từ `main` mới nhất, đặt tên theo dạng `<loại>/<mô-tả-ngắn>`:

  | Loại | Dùng cho | Ví dụ |
  |---|---|---|
  | `feat/` | tính năng mới | `feat/multi-hop-router` |
  | `fix/` | sửa lỗi | `fix/pdq-hash-overflow` |
  | `exp/` | thí nghiệm, bake-off model | `exp/siglip2-vs-jina-clip` |
  | `docs/` | tài liệu | `docs/architecture-diagram` |
  | `chore/` | CI, build, dependency | `chore/trivy-scan` |

- Commit theo [Conventional Commits](https://www.conventionalcommits.org/):

  ```text
  <type>(<scope>): <mô tả ngắn, thể mệnh lệnh>

  feat(search): add RRF fusion for hybrid retrieval
  fix(pipeline): retry img2dataset on HTTP 429
  perf(embed): batch ONNX inference with FP16
  ```

  `type`: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `build`, `ci`, `chore`.
  `scope` gợi ý: `web`, `gateway`, `search`, `pipeline`, `eval`, `bench`, `deploy`, `docs`.

- Commit nhỏ, mỗi commit một thay đổi có nghĩa. Rebase lên `main` trước khi xin review; PR được merge bằng **squash merge**.

## 5. Quy ước code

**Python** (`services/search`, `pipeline`, `eval`)
- Format & lint bằng `ruff format` + `ruff check`; kiểm tra kiểu bằng `mypy`.
- Type hint cho mọi hàm public; dùng `pathlib`, `logging` (không dùng `print` trong service).
- Script pipeline phải **idempotent** (chạy lại không tạo dữ liệu trùng) và nhận cấu hình qua CLI/biến môi trường, không hard-code đường dẫn.
- Quản lý dependency bằng `uv`; cập nhật `uv.lock` khi thay đổi `pyproject.toml`.

**Go** (`services/gateway`)
- `gofmt`/`go vet` sạch; `go test ./...` xanh.

**TypeScript** (`apps/web`)
- ESLint + Prettier; `npm run build` không lỗi kiểu.

**Chung**
- Không commit code đã comment-out, file debug, notebook có output lớn.
- Tên biến/hàm bằng tiếng Anh; docstring/comment có thể dùng tiếng Việt hoặc tiếng Anh nhưng nhất quán trong một file.

## 6. Kiểm thử & đánh giá

- Mọi thay đổi logic phải có unit test (`pytest`, `go test`).
- Thay đổi ảnh hưởng tới **chất lượng** (model, ngưỡng τ, router, rerank, prompt Gemini) phải kèm số liệu trước/sau trên testset trong `eval/`:
  - Kiểm duyệt: precision/recall/F1 (kèm CI 95%), precision(pass), tỉ lệ escalation.
  - Truy xuất: nDCG@10, Recall@k, MRR theo từng loại truy vấn (single / multi-intent / multi-hop).
- Thay đổi ảnh hưởng tới **hiệu năng** (search-api, gateway, index) phải kèm kết quả k6: p50/p95 theo loại truy vấn. Ngân sách: **p95 < 3s**, mục tiêu p95 < 2s.
- Thay đổi Docker image phải ghi kích thước image (search-api < 600MB, gateway < 30MB).
- Ghi seed, phiên bản model và commit SHA để kết quả tái lập được.

## 7. Dữ liệu, model & bí mật

- **Không commit**: ảnh, CSV/Parquet dữ liệu thô, weights model (`*.onnx`, `*.pt`, `*.safetensors`), snapshot Qdrant, thư mục `data/`, `models/`.
- **Không commit bí mật**: `.env`, API key (Gemini, Cloudflare Tunnel token, ...). Thêm biến mới vào `.env.example` với giá trị giả.
- Nếu lỡ commit bí mật: thu hồi (revoke) key ngay và báo maintainer – xoá khỏi lịch sử Git là chưa đủ.
- Tôn trọng bản quyền/giấy phép của nguồn ảnh và model; ghi rõ nguồn khi thêm dataset hoặc model mới.
- Không đưa dữ liệu cá nhân nhạy cảm vào testset, log hay ví dụ trong tài liệu.

## 8. Pull Request & review

1. Đảm bảo `make lint test` xanh ở local.
2. Mở PR vào `main`, điền đầy đủ [PR template](.github/pull_request_template.md).
3. Liên kết issue bằng `Closes #<số>`.
4. Giữ PR nhỏ (lý tưởng < 400 dòng thay đổi, không tính lockfile); tách PR lớn thành nhiều phần.
5. Cần ít nhất **1 approve** và CI xanh (lint, test, build, Trivy, smoke e2e) trước khi merge.
6. Người review tập trung vào: tính đúng đắn, tiêu chí nghiệm thu, số liệu eval/benchmark, bảo mật, khả năng bảo trì.
7. Phản hồi mọi comment; người mở comment là người đánh dấu resolved.

Cảm ơn bạn đã đóng góp! 🚀
