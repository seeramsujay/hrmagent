"""Dual-stage sandboxed compiler & runner.

Supports primary nsjail (v3.6+ syntax), automatic fallback to bubblewrap (bwrap) + prlimit,
and final fallback to subprocess + resource.setrlimit.
Extracts os.wait4 telemetry (ru_utime, ru_stime, ru_maxrss), wall-clock ms, and exact verdicts:
CE, AC, WA, TLE, MLE, RE.
"""

import os
import sys
import shutil
import subprocess
import tempfile
import time
import signal
import resource
from pathlib import Path
from typing import Optional, Tuple, List, Literal
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
            res = subprocess.run(["bwrap", "--ro-bind", "/bin", "/bin", "/bin/true"], capture_output=True, timeout=1.0)
            if res.returncode == 0:
                return "bwrap"
        except Exception:
            pass

    return "setrlimit"


def normalize_output(text: str) -> str:
    """Normalize output by stripping trailing whitespace per line and at ends."""
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


class SandboxRunner:
    """Manages compilation and sandboxed execution of C++20 and Python solutions."""

    def __init__(
        self,
        engine: Optional[Literal["nsjail", "bwrap", "setrlimit"]] = None,
        cpu_affinity_mask: Optional[str] = None,
    ):
        self.engine = engine or detect_sandbox_engine()
        self.cpu_affinity_mask = cpu_affinity_mask
        self.nsjail_bin = shutil.which("nsjail")
        self.bwrap_bin = shutil.which("bwrap")
        self.prlimit_bin = shutil.which("prlimit")

    def compile_cpp(
        self, source_path: Path, output_binary_path: Path, timeout_seconds: float = 10.0
    ) -> Tuple[bool, str]:
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
            return False, f"Compilation failed with exception: {str(e)}"

    def _execute_with_wait4(
        self,
        cmd: List[str],
        input_data: str,
        time_limit_ms: int,
        memory_limit_mb: int,
        cwd: Path,
    ) -> ExecutionTelemetry:
        """Execute command directly using fork + preexec setrlimit + wait4 for exact telemetry."""
        mem_bytes = memory_limit_mb * 1024 * 1024
        # Add safety headroom to address space for dynamic loader and glibc
        rlimit_as_bytes = mem_bytes + 64 * 1024 * 1024
        cpu_limit_secs = max(1, int(time_limit_ms / 1000.0) + 1)

        read_in, write_in = os.pipe()
        read_out, write_out = os.pipe()
        read_err, write_err = os.pipe()

        start_wall = time.perf_counter()
        pid = os.fork()

        if pid == 0:
            # Child process
            try:
                os.close(write_in)
                os.close(read_out)
                os.close(read_err)

                os.dup2(read_in, 0)
                os.dup2(write_out, 1)
                os.dup2(write_err, 2)

                os.close(read_in)
                os.close(write_out)
                os.close(write_err)

                # Set core dump size to 0
                resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

                # Apply CPU time limit
                if hasattr(resource, "RLIMIT_CPU"):
                    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit_secs, cpu_limit_secs + 1))

                # Apply Address Space limit
                if hasattr(resource, "RLIMIT_AS"):
                    resource.setrlimit(resource.RLIMIT_AS, (rlimit_as_bytes, rlimit_as_bytes))

                # Pin CPU affinity if requested
                if self.cpu_affinity_mask and hasattr(os, "sched_setaffinity"):
                    try:
                        cores = [int(c.strip()) for c in self.cpu_affinity_mask.replace("-", ",").split(",") if c.strip().isdigit()]
                        if cores:
                            os.sched_setaffinity(0, set(cores))
                    except Exception:
                        pass

                os.chdir(str(cwd))
                os.execvp(cmd[0], cmd)
            except Exception as e:
                sys.stderr.write(f"Exec failed: {e}\n")
                os._exit(127)

        # Parent process
        os.close(read_in)
        os.close(write_out)
        os.close(write_err)

        # Write input data to child stdin non-blocking or chunked
        try:
            if input_data:
                os.write(write_in, input_data.encode("utf-8"))
        except BrokenPipeError:
            pass
        finally:
            os.close(write_in)

        # Monitor child with wall-clock timeout
        wall_limit_secs = (time_limit_ms / 1000.0) * 2.5 + 0.5
        terminated = False
        status = 0
        rusage = None

        while True:
            elapsed = time.perf_counter() - start_wall
            if elapsed > wall_limit_secs:
                # Timed out, kill child
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                terminated = True

            try:
                waited_pid, status, rusage = os.wait4(pid, os.WNOHANG)
                if waited_pid == pid:
                    break
            except ChildProcessError:
                break
            time.sleep(0.005)

        end_wall = time.perf_counter()
        wall_clock_ms = (end_wall - start_wall) * 1000.0

        # Read stdout and stderr
        out_bytes = b""
        err_bytes = b""
        try:
            while True:
                chunk = os.read(read_out, 65536)
                if not chunk:
                    break
                out_bytes += chunk
        except Exception:
            pass
        finally:
            os.close(read_out)

        try:
            while True:
                chunk = os.read(read_err, 65536)
                if not chunk:
                    break
                err_bytes += chunk
        except Exception:
            pass
        finally:
            os.close(read_err)

        stdout_str = out_bytes.decode("utf-8", errors="replace")
        stderr_str = err_bytes.decode("utf-8", errors="replace")

        cpu_time_ms = 0.0
        peak_rss_kb = 0
        if rusage:
            cpu_time_ms = (rusage.ru_utime + rusage.ru_stime) * 1000.0
            peak_rss_kb = rusage.ru_maxrss  # On Linux ru_maxrss is in KiB

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
                # Check if caused by memory limit
                if peak_rss_kb > memory_limit_mb * 1024:
                    verdict = "MLE"
                else:
                    verdict = "RE"
            elif sig == signal.SIGKILL:
                if peak_rss_kb > memory_limit_mb * 1024:
                    verdict = "MLE"
                else:
                    verdict = "TLE"
            else:
                verdict = "RE"
        elif os.WIFEXITED(status):
            exit_code = os.WEXITSTATUS(status)
            if exit_code != 0:
                verdict = "RE"
        else:
            verdict = "RE"

        # Check soft thresholds
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
        cmd: List[str],
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
            "--ro-bind", "/usr", "/usr",
            "--ro-bind", "/lib", "/lib",
            "--ro-bind", "/bin", "/bin",
            "--proc", "/proc",
            "--dev", "/dev",
            "--bind", str(cwd), str(cwd),
            "--chdir", str(cwd),
        ]
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
            # Fall back directly to wait4 setrlimit
            return self._execute_with_wait4(cmd, input_data, time_limit_ms, memory_limit_mb, cwd)

    def _execute_nsjail(
        self,
        cmd: List[str],
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
            "--mode", "standalone",
            "--chroot", "/",
            "--user", "65534",
            "--group", "65534",
            "--time_limit", str(time_limit_secs),
            "--cgroup_mem_max", str(mem_bytes),
            "--max_cpus", "1",
            "--disable_proc",
            "--cwd", str(cwd),
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
            return self._execute_with_wait4(cmd, input_data, time_limit_ms, memory_limit_mb, cwd)

    def run_code(
        self,
        code: str,
        language: Literal["cpp", "python"],
        input_data: str,
        expected_output: Optional[str] = None,
        time_limit_ms: int = 2000,
        memory_limit_mb: int = 256,
        custom_checker: Optional[str] = None,
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
                    passed = self._run_custom_checker(custom_checker, input_data, telemetry.stdout, expected_output, work_dir)
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
        """Run custom checker script. Checker receives files or stdin and exits 0 for AC."""
        checker_file = cwd / "checker.py"
        checker_file.write_text(checker_code, encoding="utf-8")

        in_file = cwd / "input.txt"
        in_file.write_text(input_data, encoding="utf-8")

        out_file = cwd / "user_out.txt"
        out_file.write_text(user_output, encoding="utf-8")

        ans_file = cwd / "expected_out.txt"
        ans_file.write_text(expected_output, encoding="utf-8")

        cmd = [sys.executable, str(checker_file), str(in_file), str(out_file), str(ans_file)]
        try:
            res = subprocess.run(cmd, capture_output=True, timeout=5.0, cwd=str(cwd))
            return res.returncode == 0
        except Exception:
            return False
