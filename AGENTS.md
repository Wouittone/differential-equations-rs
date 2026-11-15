# Git/GitHub workflow

This repository uses a human-in-the-loop PR workflow for agent edits:

- The agent (Mistral Vibe) works in its own clone of this repo. That clone sets
  a repo-local git identity (`user.name`/`user.email` in `.git/config`), so
  every commit the agent makes is authored as `Mistral Vibe <vibe@mistral.ai>`.
  The human owner's global git config is never modified.
- The agent works on `vibe/<topic>` branches cut from `origin/main`, pushes the
  branch, and opens a pull request against `main` using the repository account
  (the only write-capable credential available to the agent).
- `main` is protected: direct pushes are blocked and the CI status checks are
  required. The agent never merges its own pull requests. The human owner
  reviews every agent PR — commits are visibly authored by the agent — and
  merges with their own account once CI is green.
- Machine-local editor configuration (`.vscode/`) is gitignored and never
  committed. Review instructions for the GitHub App live in
  `.github/github-app.yml`.

Note: GitHub does not allow a PR author to approve their own PR, and agent PRs
are opened with the owner's account, so the review gate is procedural (the
agent never self-merges) rather than enforced by a required-review rule on
`main`.
