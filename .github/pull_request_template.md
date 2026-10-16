## Summary

<!-- What does this PR change and why? Keep it short. -->

Closes #

## Type of change

- [ ] `feat` – New feature
- [ ] `fix` – Bug fix
- [ ] `perf` – Performance improvement
- [ ] `refactor` – No behavior change
- [ ] `exp` – Experiment / model bake-off
- [ ] `docs` – Documentation
- [ ] `build` / `ci` / `chore` – Build, CI/CD, dependencies
- [ ] ⚠️ Breaking change (schema, API, index/snapshot format)

## Affected components

- [ ] `apps/web`
- [ ] `services/gateway`
- [ ] `services/search`
- [ ] `pipeline`
- [ ] `eval` / `bench` / `loadtest`
- [ ] `deploy` / CI
- [ ] `docs`

## Acceptance criteria

<!-- Copy the checklist from the linked issue and tick what is done. -->

- [ ]

## How to test

<!-- Steps for the reviewer to reproduce / verify. -->

```bash

```

## Evaluation / benchmark results

<!-- Required when changing models, thresholds, router, reranker, prompts, index or performance. Delete if not applicable. -->

| Metric | Before | After |
|---|---|---|
| | | |

<!--
Suggested metrics:
- Moderation: P/R/F1 (95% CI), precision(pass), escalation rate
- Retrieval: nDCG@10, Recall@k, MRR@10 per single / multi-intent / multi-hop / negative
- Performance: p50/p95/p99 (k6), QPS, images/s, Docker image size
Also record: seed, model version, commit SHA, hardware.
-->

## Cost

<!-- Required for PRs that call Gemini. Delete if not applicable. -->

- Tokens (in / out):
- Spend (USD):

## Screenshots

<!-- For UI changes. Delete if not applicable. -->

## Checklist

- [ ] I have read `CONTRIBUTING.md`
- [ ] PR title follows Conventional Commits (`type(scope): summary`)
- [ ] `make lint test` passes locally
- [ ] Tests added/updated for changed logic
- [ ] Docs updated where relevant (README, `docs/`, `.env.example`)
- [ ] No secrets, raw data, images or model weights committed
- [ ] New dependencies declared and lockfile updated
- [ ] Latency budget (p95 < 3 s) and image size budgets still met

## Notes for reviewers

<!-- Points of attention, open questions, follow-ups. -->
