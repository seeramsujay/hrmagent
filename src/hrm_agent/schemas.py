"""Pydantic v2 schemas for the HRM competitive programming agent."""

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ParadigmType = Literal[
    "greedy",
    "dynamic_programming",
    "two_pointers",
    "binary_search",
    "graph_bfs_dfs",
    "shortest_path",
    "disjoint_set_union",
    "segment_tree_fenwick",
    "math_number_theory",
    "constructive",
    "bitmask",
    "divide_and_conquer",
    "flow_matching",
    "brute_force_backtrack",
    "game_theory",
]

VerdictType = Literal["AC", "WA", "TLE", "MLE", "RE", "CE", "UNKNOWN"]

CriticStatusType = Literal["PASS", "OPTIMIZE", "BUG"]

BottleneckClassType = Literal[
    "none",
    "asymptotic_time",
    "constant_factor",
    "memory_limit",
    "integer_overflow",
    "edge_case_miss",
    "logic_flaw",
    "io_overhead",
]


class ProblemSpec(BaseModel):
    """Specification of a competitive programming problem."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(..., description="Unique problem identifier (e.g. CF-1800C or CHEF-FLOW001)")
    title: str = Field(default="", description="Problem title")
    description: str = Field(..., description="Detailed problem narrative and statement")
    input_format: str = Field(default="", description="Input description specification")
    output_format: str = Field(default="", description="Output format specification")
    constraints: str = Field(default="", description="Numerical bounds (e.g., 1 <= N <= 2*10^5)")
    time_limit_ms: int = Field(default=2000, description="Time limit in milliseconds")
    memory_limit_mb: int = Field(default=256, description="Memory limit in megabytes")
    sample_inputs: list[str] = Field(default_factory=list, description="Sample input strings")
    sample_outputs: list[str] = Field(default_factory=list, description="Expected sample output strings")
    checker_code: str | None = Field(default=None, description="Optional custom checker script (Python/C++)")
    rating: int | None = Field(default=None, description="Difficulty rating (e.g. Codeforces rating)")
    tags: list[str] = Field(default_factory=list, description="Category tags")

    @field_validator("sample_inputs", "sample_outputs")
    @classmethod
    def clean_samples(cls, v: list[str]) -> list[str]:
        return [s.replace("\r\n", "\n") for s in v]


class PlannerBlueprint(BaseModel):
    """Node 1 System 2 Algorithmic Blueprint (<250 tokens JSON)."""

    model_config = ConfigDict(extra="ignore")

    analysis: str = Field(..., description="Dense algorithmic reasoning preceding constrained slots")
    paradigm: ParadigmType = Field(..., description="Primary algorithmic design paradigm")
    data_structures: list[str] = Field(default_factory=list, description="Key data structures used")
    time_complexity: str = Field(..., description="Regex-compatible time complexity e.g. O(N log N)")
    space_complexity: str = Field(..., description="Space complexity e.g. O(N)")
    use_int64: bool = Field(default=True, description="Flag indicating if 64-bit integer type is necessary")
    edge_cases: list[str] = Field(default_factory=list, description="Explicit boundary cases to guard")
    steps: list[str] = Field(
        default_factory=list,
        description="Ordered step-by-step implementation instructions",
    )


class UnifiedDiffPatch(BaseModel):
    """Node 2 Unified-diff patch representation."""

    model_config = ConfigDict(extra="ignore")

    raw_diff: str = Field(..., description="Raw unified diff text (--- a/... +++ b/...)")
    source_file: str = Field(default="solution.cpp", description="Relative file path being patched")
    target_file: str = Field(default="solution.cpp", description="Target patched file path")
    hunks_count: int = Field(default=0, description="Number of parsed diff hunks")
    applied_cleanly: bool = Field(default=False, description="Whether patch applied without offset fuzz")
    validation_error: str | None = Field(default=None, description="Error message if patching failed")


class ExecutionTelemetry(BaseModel):
    """Sandbox execution and resource consumption telemetry."""

    model_config = ConfigDict(extra="ignore")

    verdict: VerdictType = Field(..., description="Execution status verdict")
    wall_clock_ms: float = Field(..., description="Total wall-clock duration in milliseconds")
    cpu_time_ms: float = Field(..., description="Sum of ru_utime and ru_stime in milliseconds")
    peak_rss_kb: int = Field(..., description="Peak resident set size in kilobytes from ru_maxrss")
    exit_code: int = Field(default=0, description="Process exit code")
    stdout: str = Field(default="", description="Program standard output")
    stderr: str = Field(default="", description="Program standard error")
    sandbox_engine: Literal["nsjail", "bwrap", "setrlimit", "mock"] = Field(
        default="setrlimit", description="Active isolation mechanism utilized"
    )


class JevVerdict(BaseModel):
    """Node 3 Jev Critic & Complexity Optimizer verdict."""

    model_config = ConfigDict(extra="ignore")

    analysis: str = Field(..., description="Reasoning analyzing complexity, constants, and edge cases")
    status: CriticStatusType = Field(..., description="PASS, OPTIMIZE, or BUG")
    bottleneck_class: BottleneckClassType = Field(..., description="Identified bottleneck class")
    escalate_to_planner: bool = Field(..., description="True if algorithmic paradigm redesign required")
    target_lines: str = Field(default="none", description="Line span target for patch e.g. L15-L28")
    patch_directive: str = Field(..., description="Explicit guidance for diff patcher or replanning")


class LLMTelemetry(BaseModel):
    """Telemetry for local LLM inference invocations."""

    model_config = ConfigDict(extra="ignore")

    prompt_tokens: int = Field(default=0, description="Number of tokens in input prompt")
    completion_tokens: int = Field(default=0, description="Number of generated completion tokens")
    prefill_ms: float = Field(default=0.0, description="Prompt ingestion latency in ms")
    decode_ms: float = Field(default=0.0, description="Token generation latency in ms")
    tokens_per_second: float = Field(default=0.0, description="Effective decode generation throughput")
    total_latency_ms: float = Field(default=0.0, description="Total wall time for generation in ms")


class AgentTrace(BaseModel):
    """Complete structured JSONL log record for HRM execution."""

    model_config = ConfigDict(extra="ignore")

    problem_id: str
    start_time: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    end_time: str | None = None
    solved: bool = False
    final_verdict: str = "UNKNOWN"
    macro_steps: int = 0
    micro_iterations: int = 0
    critic_iterations: int = 0
    planner_blueprints: list[PlannerBlueprint] = Field(default_factory=list)
    candidate_codes: list[str] = Field(default_factory=list)
    patches: list[str] = Field(default_factory=list)
    telemetry_history: list[ExecutionTelemetry] = Field(default_factory=list)
    critic_verdicts: list[JevVerdict] = Field(default_factory=list)
    llm_metrics: list[LLMTelemetry] = Field(default_factory=list)
    final_code: str | None = None
