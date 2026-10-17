---
name: work-on-issue
description: End-to-end workflow for implementing a GitHub issue in this repo – check dependencies, branch, implement test-first, verify, open the PR and update the project board. Use when the user asks to work on, implement, start or pick up an issue/ticket (e.g. "work on #6", "implement #3", "next ticket", or the same requests in Vietnamese).
---

Follow these steps in order. Chat with the user in Vietnamese; everything written to the repo or GitHub is in English (see `CLAUDE.md`).

## 1. Understand the issue

```bash
gh issue view <N> --json title,body,milestone,labels,state
```

- Read the linked spec section ("spec §N") in the design spec file at the repo root.
- Check every issue in `Depends on:` is closed (`gh issue view <dep> --json state`). If one is still open, tell the user and ask whether to proceed anyway or pick it up first.
- If the scope is ambiguous or a decision is genuinely the user's (data rules, cost, privacy, public behavior), ask before coding. Do not re-ask decisions already listed in `CLAUDE.md`.
- If the issue has label `cost:gemini`, state the expected spend and confirm before making any paid API call.

## 2. Branch

```bash
git checkout main && git pull
git checkout -b <type>/<short-desc>      # type from the issue title: feat|fix|exp|docs|chore
```

Move the issue to **In Progress** on the board (project 1, owner `lecuong1502`):

```bash
P=$(gh project view 1 --owner lecuong1502 --format json --jq .id)
IT=$(gh project item-list 1 --owner lecuong1502 --limit 200 --format json --jq '.items[]|select(.content.number==<N>).id')
F=$(gh project field-list 1 --owner lecuong1502 --format json --jq '.fields[]|select(.name=="Status")')
gh project item-edit --project-id "$P" --id "$IT" --field-id "$(jq -r .id <<<"$F")" \
  --single-select-option-id "$(jq -r '.options[]|select(.name=="In Progress").id' <<<"$F")"
```

## 3. Implement

- Work test-first at the module's public interface: write a failing test in `tests/` (or `services/search/tests/`), make it pass, refactor.
- Tests use `data/sample/` or small inline fixtures – never the real dataset or network. Mock Gemini.
- Dependencies only via `uv add` (`--dev` for tooling, `--directory services/search` for the API), then `make requirements`.
- New settings go into `.env.example` with a comment; read them from the environment, expand `~` with `Path.expanduser()`.
- Respect the hard rules: nothing under `data/` except `data/sample/`, no weights or secrets, face detection only, every Gemini call through the shared client with the spend guard.
- Run single test files and `uv run mypy` often; the full `make lint test` at the end.

## 4. Verify against acceptance criteria

- Go through each acceptance-criteria checkbox in the issue and confirm it with evidence (command output, numbers, report path). If one cannot be met, say so – do not tick it.
- For quality or performance work, collect the metrics CONTRIBUTING §6 requires; for `cost:gemini` work, record tokens and USD.
- Run `/code-review` (or ask the user to) before opening the PR when the change is non-trivial.

## 5. Commit and open the PR

- Conventional Commit message (`type(scope): summary`), body ending with `Closes #<N>` and the co-author trailer from the system reminder.
- `git push -u origin <branch>` and `gh pr create --base main` filling `.github/pull_request_template.md`: summary, ticked acceptance criteria with evidence, how to test, metrics/cost tables when relevant.
- Never push to `main`; never merge the PR yourself – the user reviews and merges.

## 6. Report back (in Vietnamese)

Give the PR link, what was built, how each acceptance criterion was verified, anything skipped or deferred, and the next unblocked issue(s) on the board.
