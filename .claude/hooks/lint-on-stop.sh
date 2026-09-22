#!/bin/bash
# Lint once per turn; send errors back to Claude to fix.
INPUT=$(cat)
if [ "$(echo "$INPUT" | jq -r '.stop_hook_active')" = "true" ]; then
  exit 0   # already sent Claude back once; let it stop
fi
cd "$CLAUDE_PROJECT_DIR"
if ! OUT=$(pixi run lint 2>&1); then
  echo "Lint failed, fix before finishing:" >&2
  echo "$OUT" | tail -40 >&2
  exit 2
fi
exit 0