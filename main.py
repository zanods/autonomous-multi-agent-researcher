from __future__ import annotations

import argparse
import os

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
    if args.live and not os.getenv("OPENAI_API_KEY"):
        parser.error("--live needs OPENAI_API_KEY set (see .env.example)")

    app = build_workflow()
    result = app.invoke(
        {
            "query": args.query,
            "tasks": [],
            "findings": [],
            "run_log": [],
            "current_task_id": None,
            "report": None,
            "mock_mode": mock_mode,
        }
    )

    print(result["report"])
    print("\n--- run log ---")
    for line in result["run_log"]:
        print(line)


if __name__ == "__main__":
    main()
