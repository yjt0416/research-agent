# Contributing to Research Agent Copilot

Thanks for taking the time to contribute.

## Development Setup

1. Create and activate a virtual environment.
2. Install dependencies with `pip install -r backend\requirements.txt`.
3. Copy `.env.example` to `.env` and provide the required API configuration.
4. Start the app with `uvicorn app.main:app --app-dir backend --reload`.

## Recommended Workflow

1. Open an issue for bugs, UX gaps, or feature proposals when the change is not trivial.
2. Create a focused branch for your work.
3. Keep pull requests small and easy to review.
4. Add or update tests when behavior changes.
5. Run `pytest -q` before submitting.

## Coding Expectations

- Prefer clear, maintainable changes over clever ones.
- Keep tool execution safe and explicit.
- Preserve repository cleanliness by avoiding generated local artifacts in commits.
- Do not commit private keys, `.env`, logs, or personal data.

## Pull Request Checklist

- The change is scoped to a single feature, fix, or refactor.
- Tests pass locally, or any missing coverage is clearly explained.
- README or API docs are updated if behavior changed.
- New configuration is reflected in `.env.example` when relevant.

## Areas That Need Help

- Better streaming UX in the chat interface
- Artifact preview and management
- Evaluation automation and regression checks
- Additional safe tools for research workflows

Questions are welcome. If something is unclear, opening a draft PR with notes is completely fine.
