"""
JJ Execution Server
===================

Instructions:

# Build the image
docker build -t jj-exec-server .

# Run the container
docker run -p 5003:5003 jj-exec-server

# Test it
curl -X POST http://localhost:5003/repo/create -H "Content-Type: application/json"
curl -X POST http://localhost:5003/repo/<id>/execute \
  -H "Content-Type: application/json" \
  -d '{"args": ["log"]}'

Flask server that manages jj repositories inside the container.
Each rollout gets its own temp directory. The server exposes endpoints for:
  - Creating/destroying repos
  - Running jj commands
  - Writing files
  - Capturing repo state for scoring
"""

import os
import shlex
import shutil
import subprocess
import uuid
from pathlib import Path

from flask import Flask, jsonify, request

app = Flask(__name__)

# Active repos: repo_id -> directory path
REPOS: dict[str, str] = {}

# Where repos live inside the container
REPO_BASE = "/tmp/jj_repos"
os.makedirs(REPO_BASE, exist_ok=True)

# Environment for jj — prevent interactive prompts
JJ_ENV = {
    **os.environ,
    "JJ_EDITOR": "true",
}

# Config flags passed to every jj command for user identity
JJ_CONFIG_FLAGS = [
    "--config", 'user.name="Train Bot"',
    "--config", 'user.email="train@example.com"',
]

COMMAND_TIMEOUT = 30

# Safety: allowed jj subcommands
ALLOWED_SUBCOMMANDS = {
    "abandon", "describe", "diff", "edit", "log", "new", "rebase",
    "resolve", "restore", "revert", "show", "split", "squash",
    "status", "undo", "bookmark", "file",
}
BLOCKED_SUBCOMMANDS = {"git", "config", "workspace", "util", "operation"}


def _get_repo_dir(repo_id: str):
    """Look up a repo directory, returning (path, None) or (None, error_response)."""
    if repo_id not in REPOS:
        return None, (jsonify({"error": f"Unknown repo_id: {repo_id}"}), 404)
    return REPOS[repo_id], None


def _run_jj(repo_dir: str, args: list[str]) -> dict:
    """Run a jj command and return result dict."""
    cmd = ["jj"] + JJ_CONFIG_FLAGS + args + ["--no-pager"]
    try:
        result = subprocess.run(
            cmd,
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT,
            env=JJ_ENV,
        )
        return {
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": f"Command timed out after {COMMAND_TIMEOUT}s",
            "returncode": -1,
            "timed_out": True,
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": f"Execution error: {e}",
            "returncode": -1,
            "timed_out": False,
        }


def _capture_state(repo_dir: str) -> dict:
    """Capture full repo state for scoring."""
    log_result = _run_jj(repo_dir, ["log", "--no-graph", "-r", "all()"])
    status_result = _run_jj(repo_dir, ["status"])
    diff_result = _run_jj(repo_dir, ["diff"])

    # Check for conflicts via jj resolve --list (empty output = no conflicts)
    resolve_result = _run_jj(repo_dir, ["resolve", "--list"])
    has_conflicts = bool(resolve_result["stdout"].strip())

    # Read all files (excluding .jj and .git directories)
    file_contents = {}
    repo_path = Path(repo_dir)
    skip_dirs = {".jj", ".git"}
    for file_path in repo_path.rglob("*"):
        if file_path.is_file() and not (skip_dirs & set(file_path.parts)):
            rel_path = str(file_path.relative_to(repo_path))
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                file_contents[rel_path] = content
                if "<<<<<<" in content or ">>>>>>>" in content:
                    has_conflicts = True
            except Exception:
                pass

    return {
        "file_contents": file_contents,
        "log_output": log_result["stdout"],
        "status_output": status_result["stdout"],
        "diff_output": diff_result["stdout"],
        "has_conflicts": has_conflicts,
        "is_empty": len(file_contents) == 0,
    }


# ─── Endpoints ────────────────────────────────────────────────────────────────


@app.route("/health", methods=["GET"])
def health():
    """Health check — also verifies jj is available."""
    try:
        result = subprocess.run(
            ["jj", "--version"], capture_output=True, text=True, timeout=5
        )
        return jsonify({
            "status": "ok",
            "jj_version": result.stdout.strip(),
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/repo/create", methods=["POST"])
def repo_create():
    """Create a new empty jj repo. Returns repo_id."""
    repo_id = str(uuid.uuid4())[:12]
    repo_dir = os.path.join(REPO_BASE, repo_id)
    os.makedirs(repo_dir, exist_ok=True)

    # Initialize jj repo (jj 0.35+ requires 'git init')
    result = _run_jj(repo_dir, ["git", "init"])
    if result["returncode"] != 0:
        shutil.rmtree(repo_dir, ignore_errors=True)
        return jsonify({"error": f"jj init failed: {result['stderr']}"}), 500

    REPOS[repo_id] = repo_dir
    return jsonify({"repo_id": repo_id})


@app.route("/repo/<repo_id>/setup", methods=["POST"])
def repo_setup(repo_id: str):
    """
    Run a sequence of scramble operations to create a broken repo state.

    Body: {"scramble_ops": [{"type": "write_file"|"jj_command"|"delete_file", ...}, ...]}
    """
    repo_dir, err = _get_repo_dir(repo_id)
    if err:
        return err

    data = request.json or {}
    scramble_ops = data.get("scramble_ops", [])

    for op in scramble_ops:
        op_type = op.get("type")

        if op_type == "write_file":
            file_path = Path(repo_dir) / op["path"]
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(op["content"], encoding="utf-8")

        elif op_type == "jj_command":
            result = _run_jj(repo_dir, op["args"])
            if result["returncode"] != 0:
                # Log but don't abort — some scramble ops may intentionally fail
                pass

        elif op_type == "delete_file":
            file_path = Path(repo_dir) / op["path"]
            if file_path.exists():
                file_path.unlink()

    state = _capture_state(repo_dir)
    return jsonify({"state": state})


@app.route("/repo/<repo_id>/execute", methods=["POST"])
def repo_execute(repo_id: str):
    """
    Execute a jj command (model-issued, with allowlist check).

    Body: {"command": "squash -m 'Fix merge'"}
    The command should NOT include the "jj" prefix.
    """
    repo_dir, err = _get_repo_dir(repo_id)
    if err:
        return err

    data = request.json or {}
    command_str = data.get("command", "")

    # Validate
    parts = command_str.strip().split()
    if not parts:
        return jsonify({"stdout": "", "stderr": "Empty command", "returncode": 1})

    subcommand = parts[0]
    if subcommand in BLOCKED_SUBCOMMANDS:
        return jsonify({
            "stdout": "",
            "stderr": f"Blocked subcommand: {subcommand}",
            "returncode": 1,
        })
    if subcommand not in ALLOWED_SUBCOMMANDS:
        return jsonify({
            "stdout": "",
            "stderr": f"Unknown or disallowed subcommand: {subcommand}",
            "returncode": 1,
        })

    # Parse respecting quotes
    try:
        args = shlex.split(command_str)
    except ValueError as e:
        return jsonify({
            "stdout": "",
            "stderr": f"Failed to parse command: {e}",
            "returncode": 1,
        })

    result = _run_jj(repo_dir, args)
    return jsonify(result)


@app.route("/repo/<repo_id>/write_file", methods=["POST"])
def repo_write_file(repo_id: str):
    """
    Write a file in the repo (for conflict resolution).

    Body: {"path": "calc.py", "content": "..."}
    """
    repo_dir, err = _get_repo_dir(repo_id)
    if err:
        return err

    data = request.json or {}
    path = data.get("path", "")
    content = data.get("content", "")

    if not path:
        return jsonify({"stdout": "", "stderr": "No path provided", "returncode": 1})

    try:
        file_path = Path(repo_dir) / path
        # Prevent path traversal
        resolved = file_path.resolve()
        repo_resolved = Path(repo_dir).resolve()
        if not str(resolved).startswith(str(repo_resolved)):
            return jsonify({
                "stdout": "",
                "stderr": "Path traversal attempt blocked",
                "returncode": 1,
            })
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return jsonify({
            "stdout": f"Wrote {len(content)} bytes to {path}",
            "stderr": "",
            "returncode": 0,
        })
    except Exception as e:
        return jsonify({
            "stdout": "",
            "stderr": f"Failed to write file: {e}",
            "returncode": 1,
        })


@app.route("/repo/<repo_id>/state", methods=["GET"])
def repo_state(repo_id: str):
    """Capture and return the full repo state."""
    repo_dir, err = _get_repo_dir(repo_id)
    if err:
        return err

    state = _capture_state(repo_dir)
    return jsonify(state)


@app.route("/repo/<repo_id>/cleanup", methods=["DELETE"])
def repo_cleanup(repo_id: str):
    """Delete a repo and free resources."""
    repo_dir, err = _get_repo_dir(repo_id)
    if err:
        return err

    shutil.rmtree(repo_dir, ignore_errors=True)
    del REPOS[repo_id]
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003)
