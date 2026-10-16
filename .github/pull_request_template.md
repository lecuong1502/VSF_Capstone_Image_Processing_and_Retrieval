## Mô tả

<!-- Thay đổi gì và tại sao? Tóm tắt ngắn gọn. -->

Closes #

## Loại thay đổi

- [ ] `feat` – Tính năng mới
- [ ] `fix` – Sửa lỗi
- [ ] `perf` – Cải thiện hiệu năng
- [ ] `refactor` – Tái cấu trúc, không đổi hành vi
- [ ] `exp` – Thí nghiệm / bake-off model
- [ ] `docs` – Tài liệu
- [ ] `build` / `ci` / `chore` – Build, CI/CD, dependency
- [ ] ⚠️ Breaking change (schema, API, định dạng index/snapshot)

## Thành phần bị ảnh hưởng

- [ ] `apps/web`
- [ ] `services/gateway`
- [ ] `services/search`
- [ ] `pipeline`
- [ ] `eval` / `bench` / `loadtest`
- [ ] `deploy` / CI
- [ ] `docs`

## Task roadmap & tiêu chí nghiệm thu

<!-- Nếu PR thuộc roadmap (mục 10 tài liệu thiết kế), ghi số task và tiêu chí. Xoá nếu không áp dụng. -->

- Task #:
- Tiêu chí nghiệm thu:
- Kết quả:

## Cách kiểm tra

<!-- Các bước để reviewer tái hiện/kiểm tra. -->

```bash

```

## Kết quả đánh giá / benchmark

<!-- Bắt buộc nếu thay đổi model, ngưỡng, router, rerank, prompt, index hoặc hiệu năng. Xoá nếu không áp dụng. -->

| Chỉ số | Trước | Sau |
|---|---|---|
| | | |

<!--
Gợi ý chỉ số:
- Kiểm duyệt: P/R/F1 (CI 95%), precision(pass), tỉ lệ escalation
- Truy xuất: nDCG@10, Recall@k, MRR theo single / multi-intent / multi-hop
- Hiệu năng: p50/p95 (k6), QPS, ảnh/s, kích thước Docker image
Ghi kèm: seed, phiên bản model, commit SHA, phần cứng.
-->

## Ảnh chụp màn hình

<!-- Với thay đổi UI. Xoá nếu không áp dụng. -->

## Checklist

- [ ] Đã đọc `CONTRIBUTING.md`
- [ ] Tiêu đề PR theo Conventional Commits (`type(scope): mô tả`)
- [ ] `make lint test` xanh ở local
- [ ] Đã thêm/cập nhật test cho logic thay đổi
- [ ] Đã cập nhật tài liệu liên quan (README, `docs/`, `.env.example`)
- [ ] Không commit bí mật, dữ liệu thô, ảnh hoặc weights model
- [ ] Dependency mới (nếu có) đã được khai báo và cập nhật lockfile
- [ ] Không làm vượt ngân sách latency (p95 < 3s) hoặc kích thước image

## Ghi chú cho reviewer

<!-- Điểm cần chú ý, câu hỏi mở, phần chưa xong hoặc việc tiếp theo. -->
