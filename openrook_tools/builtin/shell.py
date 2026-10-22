"""Builtin shell tool — command execution with structured output.

Captures stdout, stderr, and exit code. Normalizes timeouts and
execution errors into ToolResult values.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from typing import Any, Sequence

from core.types import ToolCapability, ToolCategory, ToolResult
from openrook_tools.executor import ToolHandler

# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------

CAPABILITIES: list[ToolCapability] = [
    ToolCapability(
        name="shell.run_command",
        description=(
            "Run a shell command through the local system shell and capture "
            "stdout, stderr, and exit code. Supports normal shell syntax such "
            "as pipes, redirects, environment expansion, and &&."
        ),
        category=ToolCategory.DESTRUCTIVE,
        arg_schema={
            "cmd": {"type": "string", "required": True},
            "cwd": {"type": "string", "required": False},
            "timeout_seconds": {"type": "number", "required": False, "default": 30, "maximum": 600},
        },
    ),
]

# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT = 30
_MAX_TIMEOUT = 600  # hard cap — model-supplied timeouts cannot exceed this

# Characters/sequences that can chain or smuggle a second command past the
# first-token allowlist check. Deliberately conservative: allowlist mode
# trades shell features (pipes, redirection, substitution, backgrounding,
# and ``${...}`` expansion — env vars can smuggle content into arguments
# even without command substitution) for containment.
_CHAINING_SEQUENCES = (";", "|", "&", "`", "$(", "${", ">", "<", "\n")


def _allowlist_violation(cmd: str, allowlist: Sequence[str]) -> str | None:
    """Return a rejection reason if ``cmd`` violates the allowlist policy.

    Policy (only applied when the allowlist is non-empty): the first token
    of the command — the executable word, after shlex splitting — must
    exactly match an allowlist entry, and the command must not contain
    shell chaining/redirection metacharacters that could smuggle a second
    command. Returns ``None`` when the command is permitted.
    """
    for seq in _CHAINING_SEQUENCES:
        if seq in cmd:
            shown = seq.replace("\n", "\\n")
            return (
                f"shell_command_allowlist policy: command contains {shown!r}; "
                "chaining/redirection metacharacters are not allowed when a "
                "command allowlist is active"
            )
    try:
        tokens = shlex.split(cmd)
    except ValueError as exc:
        return f"shell_command_allowlist policy: could not parse command ({exc})"
    if not tokens:
        return "shell_command_allowlist policy: command has no executable token"
    executable = tokens[0]
    if executable not in allowlist:
        return (
            f"shell_command_allowlist policy: executable {executable!r} is not "
            "in the configured allowlist"
        )
    return None


def _coerce_timeout(raw: Any) -> float | None:
    """Validate a model-supplied timeout. Returns seconds, or None if invalid.

    Non-positive (and NaN) values fall back to the default; anything above
    the hard cap is clamped to ``_MAX_TIMEOUT``.
    """
    try:
        timeout = float(raw)
    except (TypeError, ValueError):
        return None
    if not timeout > 0:  # also catches NaN
        return float(_DEFAULT_TIMEOUT)
    return min(timeout, float(_MAX_TIMEOUT))


def _make_run_command(
    allowlist: Sequence[str] | None = None,
    default_cwd: str | None = None,
) -> ToolHandler:
    """Build the ``shell.run_command`` handler.

    Args:
        allowlist: Optional command allowlist. ``None`` or empty =
            unrestricted (legacy behavior). Non-empty = first-token exact
            match required and chaining metacharacters rejected.
        default_cwd: Working directory used when the caller omits ``cwd``.
            ``None`` falls back to the server process cwd (legacy behavior);
            production wiring passes the configured tools workspace.
    """
    active_allowlist = list(allowlist or [])

    def _run_command(args: dict[str, Any]) -> ToolResult:
        cmd = str(args["cmd"])
        cwd = args.get("cwd") or None
        raw_timeout = args.get("timeout_seconds", _DEFAULT_TIMEOUT)
        timeout = _coerce_timeout(raw_timeout)
        if timeout is None:
            return ToolResult(
                tool_name="shell.run_command",
                success=False,
                output="",
                error=f"Invalid timeout_seconds: {raw_timeout!r} (must be a number)",
                metadata={"cmd": cmd, "timeout_seconds": raw_timeout},
            )
        if not cmd.strip():
            return ToolResult(
                tool_name="shell.run_command",
                success=False,
                output="",
                error="Command is empty",
                metadata={"cmd": cmd},
            )
        if active_allowlist:
            violation = _allowlist_violation(cmd, active_allowlist)
            if violation is not None:
                return ToolResult(
                    tool_name="shell.run_command",
                    success=False,
                    output="",
                    error=violation,
                    metadata={"cmd": cmd, "policy": "shell_command_allowlist", "execution_started": False},
                )

        if cwd is not None:
            # Explicit caller-supplied cwd: honored, but a missing directory
            # is a deterministic tool error rather than a raw OSError.
            cwd = str(cwd)
            if not os.path.isdir(cwd):
                return ToolResult(
                    tool_name="shell.run_command",
                    success=False,
                    output="",
                    error=f"cwd does not exist or is not a directory: {cwd}",
                    metadata={"cmd": cmd, "cwd": cwd},
                )
        elif default_cwd is not None:
            # Default to the configured tools workspace (created on demand,
            # mirroring the filesystem tool sandbox).
            os.makedirs(default_cwd, exist_ok=True)
            cwd = default_cwd

        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                executable="/bin/bash",
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_name="shell.run_command",
                success=False,
                output="",
                error=f"Command timed out after {timeout}s",
                metadata={"cmd": cmd, "timeout_seconds": timeout},
                side_effect_summary="process killed after timeout",
            )
        except OSError as exc:
            return ToolResult(
                tool_name="shell.run_command",
                success=False,
                output="",
                error=f"OSError: {exc}",
                metadata={"cmd": cmd},
            )

        output_parts: list[str] = []
        if proc.stdout:
            output_parts.append(proc.stdout)
        if proc.stderr:
            output_parts.append(f"[stderr]\n{proc.stderr}")
        output = "\n".join(output_parts)

        success = proc.returncode == 0
        return ToolResult(
            tool_name="shell.run_command",
            success=success,
            output=output,
            error=None if success else f"Exit code {proc.returncode}",
            metadata={
                "cmd": cmd,
                "exit_code": proc.returncode,
            },
            side_effect_summary="command executed" if success else f"command failed (exit {proc.returncode})",
        )

    return _run_command


# ---------------------------------------------------------------------------
# Handler map
# ---------------------------------------------------------------------------


def create_handlers(
    *,
    allowlist: Sequence[str] | None = None,
    default_cwd: str | None = None,
) -> dict[str, ToolHandler]:
    """Return shell handlers configured with an optional command allowlist
    and a default working directory for calls that omit ``cwd``.
    """
    return {
        "shell.run_command": _make_run_command(
            allowlist=allowlist,
            default_cwd=default_cwd,
        ),
    }


# Legacy export — unrestricted, process-cwd default. Kept for direct test
# wiring; production callers go through ``create_handlers``.
HANDLERS: dict[str, ToolHandler] = create_handlers()
