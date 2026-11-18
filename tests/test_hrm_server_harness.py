"""Integration and unit test suite for localhost HRM server and harness client."""

import json

from starlette.testclient import TestClient

from hrm_agent.harness import HRMHarnessClient, load_harness_problems
from hrm_agent.schemas import ProblemSpec
from hrm_agent.server import create_app


def test_server_health_and_models():
    """Verify /health and /v1/models endpoints."""
    app = create_app(mode="mock")
    client = TestClient(app)

    # 1. Root
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"
    assert "/v1/chat/completions" in data["endpoints"]

    # 2. Health
    h_resp = client.get("/health")
    assert h_resp.status_code == 200
    h_data = h_resp.json()
    assert h_data["status"] == "healthy"
    assert "hrm-dual-pair" in h_data["models"]

    # 3. Models
    m_resp = client.get("/v1/models")
    assert m_resp.status_code == 200
    m_data = m_resp.json()
    ids = [m["id"] for m in m_data["data"]]
    assert "hrm-dual-pair" in ids
    assert "hrm-architect-critic" in ids
    assert "hrm-coder-patcher" in ids


def test_openai_chat_completions():
    """Verify OpenAI-compatible /v1/chat/completions endpoint."""
    app = create_app(mode="mock")
    client = TestClient(app)

    # 1. Dual-Pair model routing
    req_dual = {
        "model": "hrm-dual-pair",
        "messages": [
            {"role": "system", "content": "You are an assistant."},
            {"role": "user", "content": "Write a python function to compute sum."},
        ],
        "temperature": 0.2,
        "max_tokens": 128,
    }
    resp = client.post("/v1/chat/completions", json=req_dual)
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "hrm-dual-pair"
    assert len(data["choices"]) == 1
    assert "message" in data["choices"][0]
    assert len(data["choices"][0]["message"]["content"]) > 0
    assert data["usage"]["total_tokens"] > 0

    # 2. Architect model routing
    req_arch = {
        "model": "hrm-architect-critic",
        "messages": [{"role": "user", "content": "Formulate a blueprint."}],
    }
    resp_arch = client.post("/v1/chat/completions", json=req_arch)
    assert resp_arch.status_code == 200
    assert resp_arch.json()["model"] == "hrm-architect-critic"

    # 3. Coder model routing
    req_coder = {
        "model": "hrm-coder-patcher",
        "messages": [{"role": "user", "content": "Emit code."}],
    }
    resp_coder = client.post("/v1/chat/completions", json=req_coder)
    assert resp_coder.status_code == 200
    assert resp_coder.json()["model"] == "hrm-coder-patcher"


def test_openai_text_completions():
    """Verify OpenAI-compatible legacy /v1/completions endpoint."""
    app = create_app(mode="mock")
    client = TestClient(app)

    payload = {
        "model": "hrm-dual-pair",
        "prompt": "def factorial(n):",
        "max_tokens": 64,
    }
    resp = client.post("/v1/completions", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "text_completion"
    assert len(data["choices"]) == 1
    assert len(data["choices"][0]["text"]) > 0


def test_solve_endpoint():
    """Verify /solve endpoint executing full closed-loop HRM pipeline."""
    app = create_app(mode="mock")
    client = TestClient(app)

    problem = {
        "id": "SERVER-TEST-01",
        "title": "Minimal Sum",
        "description": "Calculate sum of numbers.",
        "input_format": "N followed by numbers",
        "output_format": "Sum",
        "constraints": "1 <= N <= 100",
        "sample_inputs": ["1\n3\n1 2 3\n"],
        "sample_outputs": ["6\n"],
    }
    resp = client.post("/solve?language=python", json=problem)
    assert resp.status_code == 200
    trace = resp.json()
    assert trace["problem_id"] == "SERVER-TEST-01"
    assert "solved" in trace
    assert "final_verdict" in trace
    assert "macro_steps" in trace
    assert trace["final_code"] is not None


def test_evaluate_endpoint():
    """Verify /evaluate batch evaluation endpoint."""
    app = create_app(mode="mock")
    client = TestClient(app)

    problems = [
        {
            "id": "EVAL-01",
            "title": "Sum",
            "description": "Sum of integers",
            "sample_inputs": ["1\n2\n1 2\n"],
            "sample_outputs": ["3\n"],
        },
        {
            "id": "EVAL-02",
            "title": "Max",
            "description": "Max of integers",
            "sample_inputs": ["1\n2\n1 2\n"],
            "sample_outputs": ["2\n"],
        },
    ]

    resp = client.post("/evaluate", json={"problems": problems, "language": "python"})
    assert resp.status_code == 200
    res_data = resp.json()
    assert res_data["total_problems"] == 2
    assert "pass_at_1" in res_data
    assert len(res_data["results"]) == 2


def test_load_harness_problems(tmp_path):
    """Verify loading benchmark problems from an explicit JSON fixture and fallback."""
    # 1. Test explicit corpus fixture conversion
    corpus_file = tmp_path / "test_corpus.json"
    raw_data = [
        {
            "id": "FIXTURE-101",
            "title": "Fixture Challenge",
            "description": "Calculate square of integer.",
            "input_format": "Integer N",
            "output_format": "Integer N^2",
            "constraints": "1 <= N <= 10^4",
            "sample_inputs": ["5\n"],
            "sample_outputs": ["25\n"],
            "rating": 1300,
        }
    ]
    with open(corpus_file, "w", encoding="utf-8") as f:
        json.dump(raw_data, f)

    specs = load_harness_problems(str(corpus_file), limit=1)
    assert len(specs) == 1
    assert specs[0].id == "FIXTURE-101"
    assert specs[0].title == "Fixture Challenge"
    assert specs[0].sample_inputs == ["5\n"]
    assert specs[0].sample_outputs == ["25\n"]
    assert specs[0].rating == 1300

    # 2. Test synthetic fallback when file does not exist
    fallback_specs = load_harness_problems(str(tmp_path / "non_existent.json"), limit=2)
    assert len(fallback_specs) == 2
    assert fallback_specs[0].id == "HARNESS-SYNTH-1"


def test_harness_client_integration():
    """Verify HRMHarnessClient against in-process FastAPI TestClient."""
    app = create_app(mode="mock")
    test_client = TestClient(app)

    # Wrap TestClient in a custom transport for HRMHarnessClient
    client = HRMHarnessClient(base_url="http://testserver")

    # Patch _get and _post to route via test_client
    client._get = lambda path, timeout=30.0: test_client.get(path).json()
    client._post = lambda path, payload, timeout=120.0: test_client.post(path, json=payload).json()

    # Test health
    health = client.check_health()
    assert health["status"] == "healthy"

    # Test models
    models = client.list_models()
    assert len(models) >= 3

    # Test chat completion
    chat = client.chat_completion(
        messages=[{"role": "user", "content": "Hello"}],
        model="hrm-dual-pair",
    )
    assert "choices" in chat

    # Test solve
    spec = ProblemSpec(
        id="CLIENT-01",
        description="Print 42",
        sample_inputs=["\n"],
        sample_outputs=["42\n"],
    )
    solve = client.solve_problem(spec, language="python")
    assert solve["problem_id"] == "CLIENT-01"
