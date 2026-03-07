---
name: gitpush
description: Safely push code to GitHub while preventing sensitive files. Use when the user asks to push, publish, or sync code to GitHub, or mentions git push/branching.
---

# gitpush — Safe GitHub Push Workflow

This CLI tool automates secure code pushing to GitHub with built-in safeguards. Here's what it does:

## Key Safety Features

**Pre-push checks:**
- Git identity verification (name and email configuration)
- ".gitignore must exist" enforcement
- Secret file scanning (blocks `.env`, `.claude/`, credentials, keys)
- README update prompts
- Explicit confirmation before any commit/push

## Workflow Overview

1. **Identity Check** — Verifies git config; asks if unconfigured
2. **Repo Selection** — Lists recent repos or accepts manual URL entry
3. **Branch Selection** — Chooses target branch or creates new one
4. **Best-Practices Audit** — Ensures ".gitignore exists" and "covers dotenv variants" and scans for secrets
5. **README Check** — Prompts to update documentation
6. **Final Confirmation** — Shows summary with author, files, and branch before proceeding
7. **Execute** — Runs `git add`, `git commit`, and `git push`

## Critical Rules

The workflow includes "Chat about this" escape hatches at every blocking step. If selected, execution stops completely—the push never happens until you explicitly resume.

Never force-pushes unless requested. Always stages files explicitly (never `git add .`).
