#!/bin/bash
# Format Python files right after Claude edits them.
FILE=$(jq -r '.tool_input.file_path // empty')
[[ "$FILE" == *.py ]] || exit 0
cd "$CLAUDE_PROJECT_DIR" && pixi run fmt >/dev/null 2>&1
exit 0