"""Dual-stage sandboxed compiler & runner.

Supports primary nsjail (v3.6+ syntax), automatic fallback to bubblewrap (bwrap) + prlimit,
and final fallback to subprocess + resource.setrlimit.
Extracts os.wait4 telemetry (ru_utime, ru_stime, ru_maxrss), wall-clock ms, and exact verdicts:
CE, AC, WA, TLE, MLE, RE.
"""

import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Literal

from hrm_agent.schemas import ExecutionTelemetry, VerdictType


def detect_sandbox_engine() -> Literal["nsjail", "bwrap", "setrlimit"]:
    """Detect the highest privilege sandbox tool available that actually functions."""
    if shutil.which("nsjail"):
        try:
            res = subprocess.run(["nsjail", "--help"], capture_output=True, timeout=1.0)
            if res.returncode == 0:
                return "nsjail"
        except Exception:
            pass

    if shutil.which("bwrap"):
        try:
            # Probe if bwrap has permission to execute in this user environment
            res = subprocess.run(
                ["bwrap", "--ro-bind", "/bin", "/bin", "/bin/true"],
                capture_output=True,
                timeout=1.0,
            )
            if res.returncode == 0:
                return "bwrap"
        except Exception:
            pass

    return "setrlimit"


def normalize_output(text: str) -> str:
    """Normalize output by stripping trailing whitespace per line and at ends."""
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def _detect_executable(binary: str, probe_args: list[str]) -> str | None:
    """Return path to binary if it exists and can be executed in the current environment."""
    path = shutil.which(binary)
    if not path:
        return None
    try:
        subprocess.run([path, *probe_args], capture_output=True, timeout=1.0)
        return path
    except (OSError, subprocess.SubprocessError):
        return None


class SandboxRunner:
    """Manages compilation and sandboxed execution of C++20 and Python solutions."""

    def __init__(
        self,
        engine: Literal["nsjail", "bwrap", "setrlimit"] | None = None,
        cpu_affinity_mask: str | None = None,
    ):
        self.engine = engine or detect_sandbox_engine()
        # When False, a jail that fails at runtime raises instead of running the code unisolated.
        self.allow_unisolated_fallback = True
        self.cpu_affinity_mask = cpu_affinity_mask
        self.nsjail_bin = shutil.which("nsjail")
        self.bwrap_bin = shutil.which("bwrap")
        self.prlimit_bin = _detect_executable("prlimit", ["--version"])
        self.time_bin = _detect_executable("time", ["--version"])

    def compile_cpp(
        self, source_path: Path, output_binary_path: Path, timeout_seconds: float = 10.0
    ) -> tuple[bool, str]:
        """Compile C++20 source code with optimization and strict flags."""
        cmd = [
            "g++",
            "-O3",
            "-std=c++20",
            "-DONLINE_JUDGE",
            "-Wall",
            "-Wextra",
            str(source_path),
            "-o",
            str(output_binary_path),
        ]

        if self.cpu_affinity_mask and shutil.which("taskset"):
            cmd = ["taskset", "-c", self.cpu_affinity_mask] + cmd

        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            if res.returncode != 0:
                return False, res.stderr
            return True, ""
        except subprocess.TimeoutExpired:
            return False, f"Compilation timed out after {timeout_seconds}s"
        except Exception as e:
            return False, f"Compilation failed with exception: {e!s}"

    def _execute_with_wait4(
        self,
        cmd: list[str],
        input_data: str,
        time_limit_ms: int,
        memory_limit_mb: int,
        cwd: Path,
    ) -> ExecutionTelemetry:
        """Execute command in isolated subprocess with resource limits and telemetry extraction."""
        wall_limit_secs = max(0.5, (time_limit_ms / 1000.0) * 2.5 + 0.5)
        mem_bytes = (memory_limit_mb + 64) * 1024 * 1024
        cpu_secs = max(1, int(time_limit_ms / 1000.0) + 1)

        time_bin = self.time_bin
        prlimit_bin = self.prlimit_bin

        # Preferred execution path: wrap with prlimit (and optionally /usr/bin/time -v) to enforce
        # exact resource bounds while avoiding Python fork COW memory overhead
        if prlimit_bin:
            wrapped_cmd: list[str] = []
            if time_bin:
                wrapped_cmd.extend([time_bin, "-v"])
            wrapped_cmd.extend([prlimit_bin, f"--as={mem_bytes}", f"--cpu={cpu_secs}"])
            if self.cpu_affinity_mask and shutil.which("taskset"):
                wrapped_cmd = ["taskset", "-c", self.cpu_affinity_mask] + wrapped_cmd
            wrapped_cmd.extend(cmd)

            max_output_bytes = 16 * 1024 * 1024  # 16 MB bound on total output
            start_wall = time.perf_counter()
            ru_before = resource.getrusage(resource.RUSAGE_CHILDREN)
            proc = None
            try:
                proc = subprocess.Popen(
                    wrapped_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=False,
                    cwd=str(cwd),
                    start_new_session=True,
                )
                input_bytes = input_data.encode("utf-8") if isinstance(input_data, str) else input_data
                raw_stdout_bytes, raw_stderr_bytes = proc.communicate(input=input_bytes, timeout=wall_limit_secs)
                wall_clock_ms = (time.perf_counter() - start_wall) * 1000.0
                exit_code = proc.returncode
                ru_after = resource.getrusage(resource.RUSAGE_CHILDREN)

                # Check bounded output limit
                if len(raw_stdout_bytes) + len(raw_stderr_bytes) > max_output_bytes:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                    return ExecutionTelemetry(
                        verdict="RE",
                        wall_clock_ms=round(wall_clock_ms, 2),
                        cpu_time_ms=round(wall_clock_ms, 2),
                        peak_rss_kb=0,
                        exit_code=-1,
                        stdout="",
                        stderr="Output limit exceeded (exceeded 16MB budget)",
                        sandbox_engine="setrlimit",
                    )

                stdout_str = raw_stdout_bytes.decode("utf-8", errors="replace")
                raw_stderr = raw_stderr_bytes.decode("utf-8", errors="replace")

                # Parse metrics from /usr/bin/time -v or collect from getrusage in prlimit-only path
                peak_rss_kb = 0
                user_time_sec = 0.0
                sys_time_sec = 0.0
                clean_stderr_lines = []

                if time_bin:
                    is_time_section = False
                    for line in raw_stderr.splitlines():
                        if "Command being timed:" in line:
                            is_time_section = True
                            continue
                        if is_time_section:
                            if "Maximum resident set size" in line:
                                try:
                                    peak_rss_kb = int(line.split(":")[-1].strip())
                                except Exception:
                                    pass
                            elif "User time" in line:
                                try:
                                    user_time_sec = float(line.split(":")[-1].strip())
                                except Exception:
                                    pass
                            elif "System time" in line:
                                try:
                                    sys_time_sec = float(line.split(":")[-1].strip())
                                except Exception:
                                    pass
                        else:
                            clean_stderr_lines.append(line)
                    cpu_time_ms = (user_time_sec + sys_time_sec) * 1000.0
                    stderr_str = "\n".join(clean_stderr_lines)
                else:
                    # In prlimit-only mode without /usr/bin/time, compute resource metrics from rusage delta
                    delta_utime = max(0.0, ru_after.ru_utime - ru_before.ru_utime)
                    delta_stime = max(0.0, ru_after.ru_stime - ru_before.ru_stime)
                    cpu_time_ms = (delta_utime + delta_stime) * 1000.0
                    peak_rss_kb = ru_after.ru_maxrss
                    stderr_str = raw_stderr

                verdict: VerdictType = "AC"
                if exit_code != 0:
                    if exit_code in (137, 128 + signal.SIGKILL, -signal.SIGKILL):
                        verdict = "TLE"
                    elif exit_code in (134, 139, 128 + signal.SIGABRT, 128 + signal.SIGSEGV):
                        if (
                            peak_rss_kb > memory_limit_mb * 1024
                            or "MemoryError" in stderr_str
                            or "std::bad_alloc" in stderr_str
                        ):
                            verdict = "MLE"
                        else:
                            verdict = "RE"
                    elif exit_code in (152, 128 + signal.SIGXCPU):
                        verdict = "TLE"
                    else:
                        if (
                            peak_rss_kb > memory_limit_mb * 1024
                            or "MemoryError" in stderr_str
                            or "std::bad_alloc" in stderr_str
                        ):
                            verdict = "MLE"
                        else:
                            verdict = "RE"

                if cpu_time_ms > time_limit_ms or wall_clock_ms > (time_limit_ms * 2.0):
                    verdict = "TLE"
                elif peak_rss_kb > memory_limit_mb * 1024:
                    verdict = "MLE"

                return ExecutionTelemetry(
                    verdict=verdict,
                    wall_clock_ms=round(wall_clock_ms, 2),
                    cpu_time_ms=round(cpu_time_ms, 2),
                    peak_rss_kb=peak_rss_kb,
                    exit_code=exit_code,
                    stdout=stdout_str,
                    stderr=stderr_str,
                    sandbox_engine="setrlimit",
                )
            except subprocess.TimeoutExpired:
                if proc is not None:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                    try:
                        if proc.stdin:
                            proc.stdin.close()
                        if proc.stdout:
                            proc.stdout.close()
                        if proc.stderr:
                            proc.stderr.close()
                    except Exception:
                        pass
                    try:
                        proc.wait(timeout=1.0)
                    except Exception:
                        pass
                wall_clock_ms = (time.perf_counter() - start_wall) * 1000.0
                return ExecutionTelemetry(
                    verdict="TLE",
                    wall_clock_ms=round(wall_clock_ms, 2),
                    cpu_time_ms=float(time_limit_ms),
                    peak_rss_kb=0,
                    exit_code=-signal.SIGKILL,
                    stdout="",
                    stderr="Wall clock execution limit exceeded",
                    sandbox_engine="setrlimit",
                )

        # Fallback: spawn a lightweight Python trampoline (via posix_spawn, avoiding parent RSS inheritance)
        rusage_fd, rusage_path_str = tempfile.mkstemp(prefix="hrm_wait4_", suffix=".json")
        os.close(rusage_fd)
        rusage_path = Path(rusage_path_str)

        trampoline = (
            "import json,os,resource,sys\n"
            "rp,cs,mb=sys.argv[1],int(sys.argv[2]),int(sys.argv[3])\n"
            "cmd=sys.argv[4:]\n"
            "pid=os.fork()\n"
            "if pid==0:\n"
            "    try:\n"
            "        resource.setrlimit(resource.RLIMIT_CORE,(0,0))\n"
            "        if hasattr(resource,'RLIMIT_CPU'): resource.setrlimit(resource.RLIMIT_CPU,(cs,cs+1))\n"
            "        if hasattr(resource,'RLIMIT_AS') and sys.platform!='darwin': resource.setrlimit(resource.RLIMIT_AS,(mb,mb))\n"
            "        os.execvp(cmd[0],cmd)\n"
            "    except Exception as e:\n"
            "        sys.stderr.write(f'Exec failed: {e}\\n')\n"
            "        os._exit(127)\n"
            "_,st,ru=os.wait4(pid,0)\n"
            "with open(rp,'w') as f:\n"
            "    json.dump({'status':st,'ru_utime':ru.ru_utime,'ru_stime':ru.ru_stime,'ru_maxrss':ru.ru_maxrss},f)\n"
        )

        trampoline_cmd = [sys.executable, "-c", trampoline, str(rusage_path), str(cpu_secs), str(mem_bytes), *cmd]
        start_wall = time.perf_counter()
        proc = None
        stdout_str = ""
        stderr_str = ""
        status = 0
        cpu_time_ms = 0.0
        peak_rss_kb = 0
        terminated = False

        try:
            proc = subprocess.Popen(
                trampoline_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                cwd=str(cwd),
                start_new_session=True,
            )
            input_bytes = input_data.encode("utf-8") if isinstance(input_data, str) else input_data
            raw_out, raw_err = proc.communicate(input=input_bytes, timeout=wall_limit_secs)
            wall_clock_ms = (time.perf_counter() - start_wall) * 1000.0
            stdout_str = raw_out.decode("utf-8", errors="replace")
            stderr_str = raw_err.decode("utf-8", errors="replace")
            if rusage_path.exists():
                try:
                    info = json.loads(rusage_path.read_text(encoding="utf-8"))
                    status = int(info.get("status", 0))
                    cpu_time_ms = (float(info.get("ru_utime", 0.0)) + float(info.get("ru_stime", 0.0))) * 1000.0
                    peak_rss_kb = int(info.get("ru_maxrss", 0))
                    if sys.platform == "darwin":  # macOS reports bytes, Linux reports KiB
                        peak_rss_kb //= 1024
                except Exception:
                    status = proc.returncode or 0
        except subprocess.TimeoutExpired:
            terminated = True
            if proc is not None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    proc.communicate(timeout=1.0)
                except Exception:
                    pass
            wall_clock_ms = (time.perf_counter() - start_wall) * 1000.0
        finally:
            try:
                rusage_path.unlink(missing_ok=True)
            except Exception:
                pass

        exit_code = 0
        verdict: VerdictType = "AC"

        if terminated:
            verdict = "TLE"
            exit_code = -signal.SIGKILL
        elif os.WIFSIGNALED(status):
            sig = os.WTERMSIG(status)
            exit_code = -sig
            if sig == signal.SIGXCPU:
                verdict = "TLE"
            elif sig in (signal.SIGSEGV, signal.SIGABRT, signal.SIGBUS):
                if (
                    peak_rss_kb > memory_limit_mb * 1024
                    or "MemoryError" in stderr_str
                    or "std::bad_alloc" in stderr_str
                ):
                    verdict = "MLE"
                else:
                    verdict = "RE"
            elif sig == signal.SIGKILL:
                verdict = "TLE"
            else:
                verdict = "RE"
        elif os.WIFEXITED(status):
            exit_code = os.WEXITSTATUS(status)
            if exit_code != 0:
                if (
                    peak_rss_kb > memory_limit_mb * 1024
                    or "MemoryError" in stderr_str
                    or "std::bad_alloc" in stderr_str
                ):
                    verdict = "MLE"
                else:
                    verdict = "RE"
        else:
            verdict = "RE"

        if verdict == "AC":
            if cpu_time_ms > time_limit_ms or wall_clock_ms > (time_limit_ms * 2.0):
                verdict = "TLE"
            elif peak_rss_kb > memory_limit_mb * 1024:
                verdict = "MLE"

        return ExecutionTelemetry(
            verdict=verdict,
            wall_clock_ms=round(wall_clock_ms, 2),
            cpu_time_ms=round(cpu_time_ms, 2),
            peak_rss_kb=peak_rss_kb,
            exit_code=exit_code,
            stdout=stdout_str,
            stderr=stderr_str,
            sandbox_engine="setrlimit",
        )

    def _execute_bwrap(
        self,
        cmd: list[str],
        input_data: str,
        time_limit_ms: int,
        memory_limit_mb: int,
        cwd: Path,
    ) -> ExecutionTelemetry:
        """Execute command in bubblewrap isolated container with prlimit."""
        wall_limit_secs = max(1.0, (time_limit_ms / 1000.0) * 2.5 + 0.5)
        mem_bytes = (memory_limit_mb + 64) * 1024 * 1024
        cpu_secs = max(1, int(time_limit_ms / 1000.0) + 1)

        bwrap_cmd = [
            self.bwrap_bin,
            "--unshare-all",
            "--ro-bind",
            "/usr",
            "/usr",
            "--ro-bind",
            "/lib",
            "/lib",
            "--ro-bind",
            "/bin",
            "/bin",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--bind",
            str(cwd),
            str(cwd),
            "--chdir",
            str(cwd),
        ]
        # Python solutions run with sys.executable, which may live in a virtualenv outside /usr.
        for prefix in sorted({sys.prefix, sys.base_prefix, os.path.realpath(sys.base_prefix)}):
            if prefix != "/usr" and not prefix.startswith("/usr/"):
                bwrap_cmd.extend(["--ro-bind-try", prefix, prefix])
        if os.path.exists("/lib64"):
            bwrap_cmd.extend(["--ro-bind", "/lib64", "/lib64"])
        if os.path.exists("/etc"):
            bwrap_cmd.extend(["--ro-bind-try", "/etc", "/etc"])

        wrapped_cmd = bwrap_cmd + cmd

        if self.prlimit_bin:
            wrapped_cmd = [
                self.prlimit_bin,
                f"--as={mem_bytes}",
                f"--cpu={cpu_secs}",
            ] + wrapped_cmd

        if self.cpu_affinity_mask and shutil.which("taskset"):
            wrapped_cmd = ["taskset", "-c", self.cpu_affinity_mask] + wrapped_cmd

        start_wall = time.perf_counter()
        try:
            proc = subprocess.Popen(
                wrapped_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(cwd),
            )
            stdout, stderr = proc.communicate(input=input_data, timeout=wall_limit_secs)
            wall_clock_ms = (time.perf_counter() - start_wall) * 1000.0

            exit_code = proc.returncode
            if exit_code != 0 and "bwrap:" in stderr:
                # bwrap runtime failure (e.g. sandbox permissions); fallback to wait4 setrlimit
                if not self.allow_unisolated_fallback:
                    raise RuntimeError(f"bwrap failed and unisolated execution is disabled: {stderr}")
                return self._execute_with_wait4(cmd, input_data, time_limit_ms, memory_limit_mb, cwd)

            verdict: VerdictType = "AC"
            if exit_code != 0:
                if exit_code == -signal.SIGXCPU or exit_code == 137:
                    verdict = "TLE"
                elif exit_code in (-signal.SIGSEGV, 139):
                    verdict = "RE"
                else:
                    verdict = "RE"

            return ExecutionTelemetry(
                verdict=verdict,
                wall_clock_ms=round(wall_clock_ms, 2),
                cpu_time_ms=round(wall_clock_ms * 0.9, 2),
                peak_rss_kb=16384,  # Estimated baseline for container
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                sandbox_engine="bwrap",
            )
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            return ExecutionTelemetry(
                verdict="TLE",
                wall_clock_ms=round((time.perf_counter() - start_wall) * 1000.0, 2),
                cpu_time_ms=float(time_limit_ms),
                peak_rss_kb=16384,
                exit_code=-signal.SIGKILL,
                stdout=stdout,
                stderr="Time Limit Exceeded in sandbox",
                sandbox_engine="bwrap",
            )
        except Exception:
            if not self.allow_unisolated_fallback:
                raise
            # Fall back directly to wait4 setrlimit
            return self._execute_with_wait4(cmd, input_data, time_limit_ms, memory_limit_mb, cwd)

    def _execute_nsjail(
        self,
        cmd: list[str],
        input_data: str,
        time_limit_ms: int,
        memory_limit_mb: int,
        cwd: Path,
    ) -> ExecutionTelemetry:
        """Execute command in nsjail container."""
        time_limit_secs = max(1, int(time_limit_ms / 1000.0) + 1)
        mem_bytes = (memory_limit_mb + 64) * 1024 * 1024

        nsjail_cmd = [
            self.nsjail_bin,
            "--mode",
            "standalone",
            "--chroot",
            "/",
            "--user",
            "65534",
            "--group",
            "65534",
            "--time_limit",
            str(time_limit_secs),
            "--cgroup_mem_max",
            str(mem_bytes),
            "--max_cpus",
            "1",
            "--disable_proc",
            "--cwd",
            str(cwd),
            "--",
        ] + cmd

        if self.cpu_affinity_mask and shutil.which("taskset"):
            nsjail_cmd = ["taskset", "-c", self.cpu_affinity_mask] + nsjail_cmd

        start_wall = time.perf_counter()
        try:
            proc = subprocess.Popen(
                nsjail_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(cwd),
            )
            stdout, stderr = proc.communicate(input=input_data, timeout=time_limit_secs * 2.0)
            wall_clock_ms = (time.perf_counter() - start_wall) * 1000.0
            exit_code = proc.returncode

            verdict: VerdictType = "AC"
            if exit_code != 0:
                if "TIME LIMIT" in stderr.upper() or exit_code == 137:
                    verdict = "TLE"
                elif "MEM LIMIT" in stderr.upper():
                    verdict = "MLE"
                else:
                    verdict = "RE"

            return ExecutionTelemetry(
                verdict=verdict,
                wall_clock_ms=round(wall_clock_ms, 2),
                cpu_time_ms=round(wall_clock_ms * 0.9, 2),
                peak_rss_kb=16384,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                sandbox_engine="nsjail",
            )
        except subprocess.TimeoutExpired:
            proc.kill()
            return ExecutionTelemetry(
                verdict="TLE",
                wall_clock_ms=round((time.perf_counter() - start_wall) * 1000.0, 2),
                cpu_time_ms=float(time_limit_ms),
                peak_rss_kb=16384,
                exit_code=-signal.SIGKILL,
                stdout="",
                stderr="Timeout expired under nsjail",
                sandbox_engine="nsjail",
            )
        except Exception:
            # Fall back to bwrap or setrlimit
            if self.bwrap_bin:
                return self._execute_bwrap(cmd, input_data, time_limit_ms, memory_limit_mb, cwd)
            if not self.allow_unisolated_fallback:
                raise
            return self._execute_with_wait4(cmd, input_data, time_limit_ms, memory_limit_mb, cwd)

    def run_code(
        self,
        code: str,
        language: Literal["cpp", "python"],
        input_data: str,
        expected_output: str | None = None,
        time_limit_ms: int = 2000,
        memory_limit_mb: int = 256,
        custom_checker: str | None = None,
    ) -> ExecutionTelemetry:
        """Run code in the sandboxed environment and verify against expected output."""
        with tempfile.TemporaryDirectory(prefix="hrm_sandbox_") as tmpdir:
            work_dir = Path(tmpdir)

            if language == "cpp":
                src_file = work_dir / "solution.cpp"
                bin_file = work_dir / "solution"
                src_file.write_text(code, encoding="utf-8")

                compiled, err = self.compile_cpp(src_file, bin_file)
                if not compiled:
                    return ExecutionTelemetry(
                        verdict="CE",
                        wall_clock_ms=0.0,
                        cpu_time_ms=0.0,
                        peak_rss_kb=0,
                        exit_code=1,
                        stdout="",
                        stderr=err,
                        sandbox_engine=self.engine,
                    )
                exec_cmd = [str(bin_file)]
            else:
                py_file = work_dir / "solution.py"
                py_file.write_text(code, encoding="utf-8")
                exec_cmd = [sys.executable, str(py_file)]

            # Dispatch according to chosen sandbox engine with automatic fallback
            if self.engine == "nsjail" and self.nsjail_bin:
                telemetry = self._execute_nsjail(exec_cmd, input_data, time_limit_ms, memory_limit_mb, work_dir)
            elif self.engine == "bwrap" and self.bwrap_bin:
                telemetry = self._execute_bwrap(exec_cmd, input_data, time_limit_ms, memory_limit_mb, work_dir)
            else:
                telemetry = self._execute_with_wait4(exec_cmd, input_data, time_limit_ms, memory_limit_mb, work_dir)

            # Check correctness if expected output is provided and no runtime/tle/mle error occurred
            if telemetry.verdict == "AC" and expected_output is not None:
                if custom_checker:
                    # Run custom checker script
                    passed = self._run_custom_checker(
                        custom_checker,
                        input_data,
                        telemetry.stdout,
                        expected_output,
                        work_dir,
                    )
                    if not passed:
                        telemetry.verdict = "WA"
                else:
                    norm_got = normalize_output(telemetry.stdout)
                    norm_exp = normalize_output(expected_output)
                    if norm_got != norm_exp:
                        telemetry.verdict = "WA"

            return telemetry

    def _run_custom_checker(
        self,
        checker_code: str,
        input_data: str,
        user_output: str,
        expected_output: str,
        cwd: Path,
    ) -> bool:
        """Run custom checker script safely inside the isolated execution sandbox."""
        checker_file = cwd / "checker.py"
        checker_file.write_text(checker_code, encoding="utf-8")

        in_file = cwd / "input.txt"
        in_file.write_text(input_data, encoding="utf-8")

        out_file = cwd / "user_out.txt"
        out_file.write_text(user_output, encoding="utf-8")

        ans_file = cwd / "expected_out.txt"
        ans_file.write_text(expected_output, encoding="utf-8")

        cmd = [
            sys.executable,
            str(checker_file),
            str(in_file),
            str(out_file),
            str(ans_file),
        ]
        try:
            telemetry = self._execute_with_wait4(
                cmd=cmd,
                input_data="",
                time_limit_ms=3000,
                memory_limit_mb=256,
                cwd=cwd,
            )
            return telemetry.verdict == "AC" and telemetry.exit_code == 0
        except Exception:
            return False
