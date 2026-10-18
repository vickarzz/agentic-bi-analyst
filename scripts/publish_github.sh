#!/usr/bin/env bash
# Publish this folder to GitHub in one go.
#
#   bash scripts/publish_github.sh <github-username> [repo-name]
#   bash scripts/publish_github.sh vickarzz
#
# Before running: create an EMPTY public repo on github.com with the same name (no README),
# and have a personal access token ready (GitHub → Settings → Developer settings → Tokens).
# When git asks for a password, paste the token.
set -euo pipefail
cd "$(dirname "$0")/.."

USER_NAME=${1:?"usage: bash scripts/publish_github.sh <github-username> [repo-name]"}
REPO=${2:-agentic-bi-analyst}
URL="https://github.com/${USER_NAME}/${REPO}.git"

command -v git >/dev/null 2>&1 || { echo "git not found. On macOS run: xcode-select --install"; exit 1; }

if [ -z "$(git config --global user.name || true)" ]; then
  read -rp "Your name for git commits: " NAME; git config --global user.name "$NAME"
fi
if [ -z "$(git config --global user.email || true)" ]; then
  read -rp "Your GitHub email: " EMAIL; git config --global user.email "$EMAIL"
fi

[ -d .git ] || git init -q
git add .
if git diff --cached --quiet; then
  echo "Nothing new to commit."
else
  git commit -q -m "${COMMIT_MESSAGE:-Agentic BI Analyst: governed BI agent + multi-agent business review}"
  echo "Committed."
fi
git branch -M main
if git remote get-url origin >/dev/null 2>&1; then git remote set-url origin "$URL"; else git remote add origin "$URL"; fi

echo "Pushing to $URL ..."
git push -u origin main

cat <<EOF

Done: https://github.com/${USER_NAME}/${REPO}

Next (one time, about 5 minutes):
  1. Open https://share.streamlit.io → Create app → Deploy a public app from GitHub
  2. Repository ${USER_NAME}/${REPO} · branch main · main file app/streamlit_app.py
  3. Advanced settings → Python 3.12 → Deploy
  4. Put the app link in the repo's About → Website, the README and LinkedIn Featured

Later updates: edit files, then run this script again. Streamlit redeploys automatically.
EOF
