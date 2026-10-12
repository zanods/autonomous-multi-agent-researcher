"""Eval scenario data: fabricated-method traps plus normal queries.

Trap pattern (graphene / FractalHash): a query about something with no real
evidence behind it. The correct behavior is FAIL after retries — the pipeline
must never verify fabricated claims.

Each scenario:
  id        unique slug
  kind      "trap" (must FAIL) or "normal" (must verify)
  query     research question fed to the workflow
  tasks     [(title, description)] the mocked planner returns
  notes     research notes the mocked researcher returns every attempt
  evidence  [(ref, title, text)] the mocked retriever returns
  checker   "fail_all" | "pass_all" — mocked fact-checker LLM verdicts
"""

SCENARIOS: list[dict] = [
    {
        "id": "trap-fractalhash",
        "kind": "trap",
        "query": (
            "Report what the evidence says about the 'FractalHash' parallel "
            "hashing algorithm's claimed 47x GPU speedup."
        ),
        "tasks": [
            (
                "FractalHash evidence review",
                "Assess the claimed 47x GPU speedup of FractalHash.",
            )
        ],
        "notes": (
            "FractalHash achieves a 47x speedup on GPU clusters for parallel "
            "hashing workloads. It outperforms cuckoo hashing significantly "
            "in all benchmarks."
        ),
        "evidence": [],  # no real evidence exists — the trap
        "checker": "fail_all",
    },
    {
        "id": "trap-graphene",
        "kind": "trap",
        "query": "Report what the evidence says about graphene thermal pad performance.",
        "tasks": [
            (
                "Graphene pad evidence review",
                "Assess graphene thermal pad cooling claims.",
            )
        ],
        "notes": (
            "Graphene thermal pads reduce CPU temperatures by 25 degrees "
            "Celsius in all workloads. Every data center should switch "
            "to graphene pads immediately."
        ),
        "evidence": [],  # no real evidence exists — the trap
        "checker": "fail_all",
    },
    {
        "id": "normal-quicksort",
        "kind": "normal",
        "query": "What is the worst-case time complexity of quicksort?",
        "tasks": [
            (
                "Quicksort complexity",
                "State the worst-case Big-O bound of quicksort.",
            )
        ],
        "notes": (
            "Quicksort has O(n log n) average time and O(n^2) worst-case "
            "time complexity."
        ),
        "evidence": [
            (
                "https://en.wikipedia.org/wiki/Quicksort",
                "Quicksort",
                "Quicksort has O(n log n) average time and O(n^2) worst-case "
                "time complexity.",
            ),
        ],
        "checker": "pass_all",
    },
    {
        "id": "normal-strassen",
        "kind": "normal",
        "query": "What is the asymptotic complexity of Strassen's matrix multiplication?",
        "tasks": [
            (
                "Strassen complexity",
                "State the Big-O bound of Strassen's algorithm.",
            )
        ],
        "notes": (
            "Strassen's algorithm multiplies matrices in O(n^2.81) time, "
            "improving over the naive O(n^3) bound."
        ),
        "evidence": [
            (
                "https://en.wikipedia.org/wiki/Strassen_algorithm",
                "Strassen algorithm",
                "Strassen's algorithm multiplies matrices in O(n^2.81) time, "
                "improving over the naive O(n^3) bound.",
            ),
        ],
        "checker": "pass_all",
    },
]
