"""Example: Interfacing an Evaluation Harness with the Localhost HRM Server.

This script demonstrates how standard evaluation frameworks (Aider, SWE-bench,
lm-evaluation-harness, or custom benchmark harnesses) can point their OpenAI client
to http://127.0.0.1:8000/v1 and evaluate the HRM Dual-Pair Agent.
"""

import json
import urllib.request


def query_openai_chat(
    base_url: str = "http://127.0.0.1:8000/v1",
    model: str = "hrm-dual-pair",
    prompt: str = "Given an array of integers, write a python function to find the maximum sum subarray.",
):
    """Query localhost server using standard OpenAI /v1/chat/completions schema."""
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert competitive programming assistant.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 512,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def query_hrm_solve(
    base_url: str = "http://127.0.0.1:8000",
    problem_spec: dict | None = None,
):
    """Query dedicated HRM closed-loop solve endpoint with sandbox verification."""
    url = f"{base_url}/solve?language=python"
    spec = problem_spec or {
        "id": "EXAMPLE-01",
        "title": "Reverse Array",
        "description": "Read N numbers and print them in reverse order.",
        "input_format": "N followed by N integers",
        "output_format": "Reversed sequence",
        "constraints": "1 <= N <= 10^5",
        "sample_inputs": ["3\n1 2 3\n"],
        "sample_outputs": ["3 2 1\n"],
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(spec).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


if __name__ == "__main__":
    print("=" * 60)
    print("Testing HRM Localhost Server Harness Integration")
    print("=" * 60)

    try:
        # 1. Test OpenAI Chat Protocol
        print("\n1. Testing OpenAI Chat Protocol (/v1/chat/completions)...")
        chat_res = query_openai_chat()
        print("Model:", chat_res["model"])
        print("Usage:", chat_res["usage"])
        print("Content preview:\n", chat_res["choices"][0]["message"]["content"][:200])

        # 2. Test Direct HRM Closed-Loop Solve Protocol
        print("\n2. Testing Direct Closed-Loop Solve Protocol (/solve)...")
        solve_res = query_hrm_solve()
        print("Verdict:", solve_res["final_verdict"])
        print("Solved:", solve_res["solved"])
        print("Macro steps:", solve_res["macro_steps"])
        print("Micro iterations:", solve_res["micro_iterations"])
        print("Final synthesized code:\n", solve_res["final_code"])

    except Exception as e:
        print("\nError connecting to server:", e)
        print("Make sure to start the server first with: `hrm serve --port 8000`")
