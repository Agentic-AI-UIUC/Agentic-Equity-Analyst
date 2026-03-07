---
name: gitpush
description: Safely push code to GitHub while preventing sensitive files. Use when the user asks to push, publish, or sync code to GitHub, or mentions git push/branching.
---

# gitpush

## Configuration

> Run `setup.sh` from the repo root to set these automatically.
> Or replace manually before use:

```
GIT_NAME  = YOUR_GITHUB_USERNAME
GIT_EMAIL = YOUR_EMAIL
```

---

## Purpose
Push code to GitHub following developer best practices — git identity pre-check, .gitignore enforcement, secret scanning, and explicit confirmation before every push.

## Required Safety Rules
- Never push `.env`, `.claude/**`, credentials, or secrets.
- **A `.gitignore` must exist before staging.** If it's missing, create one.
- **`.env` and all variants must be in `.gitignore`.** Check before staging.
- Always show a final confirmation summary and get explicit approval before committing or pushing.
- Never force-push unless explicitly requested.

---

## Workflow

### Step 1 — Gather info + Identity check

Run in parallel:
```bash
git status
git diff
git remote -v
git branch -a
git config user.name
git config user.email
```

**Identity logic (do NOT ask if already set):**

- If `git config user.name` and `git config user.email` are both non-empty → use them as-is, no question needed
- If either is empty → use AskUserQuestion:

```
question: "Git identity isn't set. What name and email should commits use?"
header: "Git identity"
options:
  - label: "YOUR_GITHUB_USERNAME / YOUR_EMAIL"
    description: "Use the configured account (set via setup.sh)"
  - label: "Enter manually"
    description: "I'll type my name and email"
```

Then run:
```bash
git config user.name "<confirmed-name>"
git config user.email "<confirmed-email>"
```

---

### Step 1.5 — Repo selection (only if no remote set) (BLOCKING — use AskUserQuestion)

**Skip this step if `git remote -v` returned a remote.**

If no remote:
```bash
gh repo list --limit 5 --json name,url,updatedAt --sort updated
```

```
question: "No remote is set. Which GitHub repo should this push to?"
header: "Target repo"
options:
  - label: "<repo-name-1>"
    description: "Last updated: <updatedAt>"
  - label: "<repo-name-2>"
    description: "Last updated: <updatedAt>"
  - label: "Paste a link"
    description: "I'll provide the full repo URL"
```

Then: `git remote add origin <url>`

If `gh` fails or isn't authenticated → go straight to "Paste a link".

---

### Step 2 — Branch selection (BLOCKING — use AskUserQuestion)

```
question: "Which branch do you want to push to?"
header: "Target branch"
options:
  - label: "<current branch>"
    description: "Push to current branch (currently checked out)"
  - label: "main"
    description: "Push to main — this is the default branch"
  - label: "New branch"
    description: "Create and push to a new branch"
```

If "New branch": ask for the name, then `git checkout -b <name>`.

---

### Step 3 — Best-practices audit (BLOCKING — fix before staging)

Run all checks. Do not proceed until each passes:

**3a. `.gitignore` must exist**
```bash
test -f .gitignore || echo "MISSING"
```
If missing → create one with at minimum:
```
.env
.env.*
.env.local
.claude/
node_modules/
*.pem
*.key
```

**3b. Dotenv variants must be covered**

Check `.gitignore` covers all of:
- `.env`
- `.env.*`
- `.env.local`

If any are missing, add them to `.gitignore` before staging.

**3c. Scan staged + modified files for secrets**

Flag and stop if any of these are staged or modified:
- `.env`, `.env.*`, `.env.local`
- `.claude/**`
- `credentials.json`, `secrets.*`, `*.pem`, `*.key`
- Files containing obvious secrets (API keys, tokens, private keys)

If flagged → stop and ask the user what to do. Do not proceed until resolved.

---

### Step 4 — README check (BLOCKING — use AskUserQuestion)

```
question: "Does the README need to be updated before pushing?"
header: "README"
options:
  - label: "No, it's fine"
    description: "Proceed without changes"
  - label: "Yes, update it"
    description: "Tell me what changed and I'll update it"
  - label: "No README exists — create one"   ← only if README is missing
    description: "Generate a basic README from repo contents"
```

---

### Step 5 — Final confirmation (BLOCKING — use AskUserQuestion)

Display:
```
Ready to push:
  Repo:    <remote URL>
  Branch:  <branch>
  Files:   <staged files>
  Commit:  "<message>"
  Author:  <user.name> <<user.email>>  ✓
  .gitignore: ✓   Secrets scan: ✓
```

```
question: "Push to <branch> on <repo>?"
header: "Confirm push"
options:
  - label: "Yes, push it"
    description: "Commit and push"
  - label: "No, cancel"
    description: "Abort — nothing will be pushed"
```

**Do not run git commit or git push until "Yes, push it" is selected.**

---

### Step 6 — Execute
1. `git add <files>` (never `git add .` — be specific)
2. `git commit -m "<message>"`
3. `git push origin <branch>`

---

## Example Flow
User: "push my changes"
1. Run info checks — identity already set → skip hook
2. Remote exists → skip repo selection
3. AskUserQuestion → branch
4. Audit .gitignore → add missing `.env.*` if needed
5. Scan for secrets → clean
6. AskUserQuestion → README
7. Show summary → AskUserQuestion → confirm
8. Commit and push
