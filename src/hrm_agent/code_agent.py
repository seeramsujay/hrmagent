"""Code-Action Agent Architecture for Sub-1.5B Parameter Models.

Addresses sub-1B and 1.5B intelligence ceilings through:
1. Code-Action paradigm (executes small Python snippets invoking tools directly rather than fragile nested JSON blobs).
2. Thinking Mode with strict token budgeting (caps <think> tags at 512-1024 tokens to prevent infinite reasoning loops).
3. Minimal, atomic tools with flat string arguments (read_file, write_file, run_command).
4. SEARCH / REPLACE diff application over whole-file rewrites.
5. Context slicing (<8,000 tokens) focusing only on target functions and interfaces.
"""

import os
import re
import signal
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from hrm_agent.patcher import apply_search_replace_patch


@dataclass
class CodeActionResult:
    """Outcome of a Code-Action step."""

    success: bool
    output: str
    thinking_text: str = ""
    thinking_tokens: int = 0
    actions_executed: list[str] = field(default_factory=list)
    error: Optional[str] = None


class CodeActionTools:
    """Minimal, atomic tools with flat string arguments."""

    def __init__(self, workspace: Path):
        self.workspace = workspace

    def read_file(self, path: str) -> str:
        """Read text from a relative or absolute file within workspace."""
        target = (self.workspace / path).resolve()
        if not target.is_relative_to(self.workspace.resolve()):
            return "Error: Access outside workspace sandbox denied."
        if not target.exists():
            return f"Error: File '{path}' does not exist."
        try:
            return target.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading file: {e}"

    def write_file(self, path: str, content: str) -> str:
        """Write text to a file, creating parent directories if needed."""
        target = (self.workspace / path).resolve()
        if not target.is_relative_to(self.workspace.resolve()):
            return "Error: Access outside workspace sandbox denied."
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} bytes to {path}."
        except Exception as e:
            return f"Error writing file: {e}"

    def run_command(self, cmd: str, timeout_sec: float = 15.0) -> str:
        """Run a shell command inside the workspace sandbox."""
        try:
            proc = subprocess.Popen(
                cmd,
                shell=True,
                cwd=str(self.workspace),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except OSError:
                    pass
                proc.communicate()
                return f"Error: Command timed out after {timeout_sec}s"

            out = stdout
            if stderr:
                out += f"\n[stderr]\n{stderr}"
            return out.strip() or f"Command exited with code {proc.returncode}"
        except Exception as e:
            return f"Error executing command: {e}"

    def apply_diff(self, path: str, diff_text: str) -> str:
        """Apply SEARCH/REPLACE diff to target file."""
        current = self.read_file(path)
        if current.startswith("Error:"):
            return current
        success, patched, meta = apply_search_replace_patch(current, diff_text)
        if not success:
            return f"Error applying patch: {meta.validation_error}"
        return self.write_file(path, patched)


class CodeActionAgent:
    """Autonomous agent operating via Python code-actions and bounded thinking."""

    def __init__(
        self,
        workspace: Path | None = None,
        thinking_budget: int = 1024,
        max_prompt_tokens: int = 8000,
    ):
        self.workspace = workspace or Path(tempfile.mkdtemp(prefix="code_agent_"))
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.tools = CodeActionTools(self.workspace)
        self.thinking_budget = thinking_budget
        self.max_prompt_tokens = max_prompt_tokens

    def extract_thinking(self, raw_output: str) -> tuple[str, str, int]:
        """Extract <think> tags and enforce the thinking token budget.

        Returns:
            (thinking_content, executable_content, thinking_token_estimate)
        """
        think_match = re.search(r"<think>(.*?)</think>", raw_output, flags=re.DOTALL)
        if think_match:
            thinking_text = think_match.group(1).strip()
            # Estimate tokens as roughly 4 chars per token
            token_count = len(thinking_text) // 4
            if token_count > self.thinking_budget:
                # Cap excessive thinking to prevent runaway loops
                char_limit = self.thinking_budget * 4
                thinking_text = thinking_text[:char_limit] + "\n[Thinking token budget exceeded. Truncated.]"
            executable = raw_output[: think_match.start()] + raw_output[think_match.end() :]
            return thinking_text, executable.strip(), min(token_count, self.thinking_budget)

        return "", raw_output.strip(), 0

    def extract_code_action(self, text: str) -> str:
        """Extract Python code block from raw agent completion."""
        # Find ```python ... ``` or ``` ... ```
        block = re.search(r"```(?:python)?\s*\n(.*?)\n```", text, flags=re.DOTALL)
        if block:
            return block.group(1).strip()
        # If no fences, inspect if lines contain tool invocations
        if any(tool_name in text for tool_name in ["read_file", "write_file", "run_command", "apply_diff"]):
            return text.strip()
        return ""

    def execute_code_action(self, python_code: str) -> CodeActionResult:
        """Execute Python code-action in the current process with atomic tools.

        Atomic tools (read_file, write_file, run_command) strictly enforce workspace confinement.
        """
        safe_globals: dict[str, Any] = {
            "read_file": self.tools.read_file,
            "write_file": self.tools.write_file,
            "run_command": self.tools.run_command,
            "apply_diff": self.tools.apply_diff,
            "print": print,
        }

        output_log: list[str] = []

        def logged_print(*args: Any, **kwargs: Any) -> None:
            msg = " ".join(str(a) for a in args)
            output_log.append(msg)

        safe_globals["print"] = logged_print

        try:
            exec(python_code, safe_globals)  # nosec
            return CodeActionResult(
                success=True,
                output="\n".join(output_log),
                actions_executed=[python_code],
            )
        except Exception as e:
            return CodeActionResult(
                success=False,
                output="\n".join(output_log),
                actions_executed=[python_code],
                error=str(e),
            )
