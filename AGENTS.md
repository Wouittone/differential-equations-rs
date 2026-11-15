# Git/GitHub workflow

This repository uses a human-in-the-loop PR workflow for agent edits:

- The agent (Mistral Vibe) works in its own clone of this repo. That clone sets
  a repo-local git identity (`user.name`/`user.email` in `.git/config`), so
  every commit the agent makes is authored as `Mistral Vibe <vibe@mistral.ai>`.
  The human owner's global git config is never modified.
- The agent works on `vibe/<topic>` branches cut from `origin/main` and pushes
  them with the repository credentials.
- Pull requests are opened by the **differential-equation-rs-gh-warden GitHub
  App** (`differential-equation-rs-gh-warden[bot]`), installed on this
  repository with write access to contents and pull requests. Because the PR
  author is the app bot and not the owner, the owner can review and **approve**
  every agent PR with their own account (GitHub blocks self-approval).
- `main` is protected: direct pushes are blocked, the CI status checks are
  required, and one approving review is required. The agent never merges its
  own pull requests; the owner merges once CI is green and the PR is approved.
- Machine-local editor configuration (`.vscode/`) is gitignored and never
  committed. Review instructions for the GitHub App live in
  `.github/github-app.yml`.
