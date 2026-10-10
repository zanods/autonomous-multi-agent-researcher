from __future__ import annotations

import argparse

from dotenv import load_dotenv

load_dotenv()  # so TAVILY_API_KEY etc. can live in a local .env file

from agents.llm import has_api_key
from graph.workflow import build_workflow


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous multi-agent researcher (Sprint 1 skeleton)"
    )
    parser.add_argument("--query", required=True, help="Research question")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use a real LLM (requires OPENAI_API_KEY, see .env.example)",
    )
    args = parser.parse_args()

    mock_mode = not args.live
    if args.live and not has_api_key():
        parser.error("--live needs GOOGLE_API_KEY, GROQ_API_KEY, CEREBRAS_API_KEY, OPENROUTER_API_KEY or OPENAI_API_KEY set (see .env.example)")

    app = build_workflow()
    initial_state = {
        "query": args.query,
        "tasks": [],
        "findings": [],
        "run_log": [],
        "current_task_id": None,
        "report": None,
        "mock_mode": mock_mode,
    }

    # Stream node by node so progress is visible live — a silent 20-minute
    # hang taught us never to wait for the final invoke() again.
    print("--- progress (live) ---", flush=True)
    last_log_len = 0
    state = initial_state
    for state in app.stream(initial_state, stream_mode="values"):
        lines = state.get("run_log", [])
        for line in lines[last_log_len:]:
            print(line, flush=True)
        last_log_len = len(lines)
    result = state

    print("\n--- report ---")
    print(result["report"])

    budget = result.get("token_budget")
    if budget is not None and budget.usage:
        print("\n--- token budget ---")
        print(budget.report())


if __name__ == "__main__":
    main()
