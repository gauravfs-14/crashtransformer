# 🤝 Contributing

Thanks for your interest in contributing!

## Getting started

1. Fork the repository
2. Create a feature branch
3. Setup environment: `python crashtransformer.py setup`
4. Run a smoke test: `make data && python crashtransformer.py run --csv data/test_data_5rows.csv`

## Code style

- Write clear, maintainable Python with descriptive names
- Prefer early returns, avoid deep nesting
- Add concise comments only where non-obvious
- Keep unrelated refactors out of PRs

## Tests & checks

- Add/adjust tests if applicable
- Ensure lints pass locally
- Verify docs build: `python crashtransformer.py docs`

## PR guidelines

- One focused change per PR
- Include a summary, screenshots (if relevant), and migration notes
- Update docs if user-visible behavior changes

## Reporting issues

- Use the issue template when available
- Provide steps to reproduce, expected vs actual, and logs (redacted)

## Security

- Never include secrets in issues or PRs
- See `SECURITY.md` for reporting vulnerabilities
