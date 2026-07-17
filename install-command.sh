#!/usr/bin/env bash
set -euo pipefail

# Install a user-local command that always dispatches to this clone's environment.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CLI="$ROOT/.venv/bin/hunyuan3d"
COMMAND_DIR="${HUNYUAN3D_COMMAND_DIR:-${HOME}/.local/bin}"
SHELL_RC="${HUNYUAN3D_SHELL_RC:-${HOME}/.bashrc}"
COMMAND="$COMMAND_DIR/hunyuan3d"
MARKER="# Added by Hunyuan3D CLI installer"

if [[ ! -x "$CLI" ]]; then
  echo "hunyuan3d is not installed in $ROOT/.venv; run ./bootstrap.sh first." >&2
  exit 2
fi

mkdir -p "$COMMAND_DIR"
cat > "$COMMAND" <<EOF
#!/usr/bin/env bash
exec "$CLI" "\$@"
EOF
chmod 755 "$COMMAND"

touch "$SHELL_RC"
if ! grep -Fqx "$MARKER" "$SHELL_RC"; then
  {
    printf '\n%s\n' "$MARKER"
    printf 'export PATH="%s:$PATH"\n' "$COMMAND_DIR"
  } >> "$SHELL_RC"
fi
echo "Installed $COMMAND. Open a new Bash terminal to use hunyuan3d by name."
