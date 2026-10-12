"""Eval scenario data: fabricated-method traps plus normal queries.

Trap pattern (graphene / FractalHash): a query about something with no real
evidence behind it. The correct behavior is FAIL after retries — the pipeline
must never verify fabricated claims.

Each scenario:
  id        unique slug
  kind      "trap" (must FAIL) or "normal" (must verify) or "adversarial"
            (must FAIL — verifier stress cases)
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
    # --- Adversarial verifier stress cases: all must FAIL. ---
    {
        # (a) Negated claim: says the opposite of the evidence. The claim
        # negation guard blocks exact-match; the LLM stage must FAIL it.
        "id": "trap-negation",
        "kind": "adversarial",
        "query": "Do vapor chambers spread heat laterally?",
        "tasks": [
            (
                "Vapor chamber negation check",
                "Verify the lateral heat spreading claim.",
            )
        ],
        "notes": (
            "Vapor chambers do not spread heat laterally across heatsinks."
        ),
        "evidence": [
            (
                "https://en.wikipedia.org/wiki/Vapor_chamber",
                "Vapor chamber",
                "Vapor chambers spread heat laterally across heatsinks.",
            ),
        ],
        "checker": "fail_all",
    },
    {
        # (b) Changed units: 5°C of evidence becomes 5°F in the claim.
        # "fahrenheit" matches no evidence keyword, so it must FAIL.
        "id": "trap-units",
        "kind": "adversarial",
        "query": "What temperature drop do graphene pads achieve?",
        "tasks": [
            (
                "Graphene pad units check",
                "Verify the temperature drop figure.",
            )
        ],
        "notes": (
            "Graphene pads achieve a temperature drop of 5 degrees "
            "Fahrenheit in testing."
        ),
        "evidence": [
            (
                "https://example.com/pads",
                "Pad review",
                "Graphene pads achieve a temperature drop of 5 degrees "
                "Celsius in testing.",
            ),
        ],
        "checker": "fail_all",
    },
    {
        # (c) Half-sentence quote: "effective" is lifted out of a sentence
        # whose actual verdict is "ineffective without proper airflow".
        "id": "trap-halfquote",
        "kind": "adversarial",
        "query": "Are vapor chambers effective?",
        "tasks": [
            (
                "Vapor chamber effectiveness",
                "Verify the effectiveness claim.",
            )
        ],
        "notes": "The study found vapor chambers effective.",
        "evidence": [
            (
                "https://example.com/study",
                "Cooling study",
                "The study found vapor chambers ineffective without proper "
                "airflow.",
            ),
        ],
        "checker": "fail_all",
    },
    {
        # (d) Contradictory sources: the claim matches one chunk while
        # another credible chunk denies it — cherry-picking must not pass.
        # Both sources score above the low-credibility bar on purpose, so
        # this exercises contradiction routing, not the weak-source rule.
        "id": "trap-contradiction",
        "kind": "adversarial",
        "query": "Do vapor chambers reduce temperatures by 15 degrees?",
        "tasks": [
            (
                "Contradictory evidence check",
                "Verify against disagreeing sources.",
            )
        ],
        "notes": (
            "Vapor chambers reduce temperatures by 15 degrees Celsius."
        ),
        "evidence": [
            (
                "https://www.intel.com/vapor-chamber-whitepaper",
                "Intel whitepaper",
                "Vapor chambers reduce temperatures by 15 degrees Celsius "
                "according to vendor tests.",
            ),
            (
                "https://www.tomshardware.com/vapor-chamber-test",
                "Tom's Hardware",
                "Independent tests show vapor chambers do not reduce "
                "temperatures significantly.",
            ),
        ],
        "checker": "fail_all",
    },
    {
        # (e) Non-existent citation: the claim leans on source [7], which was
        # never retrieved. Keyword matching ignores the bogus marker; the
        # claim fails for lack of evidence.
        "id": "trap-bogus-citation",
        "kind": "adversarial",
        "query": "What do studies say about vapor chamber cooling?",
        "tasks": [
            (
                "Bogus citation check",
                "Verify the cited study claim.",
            )
        ],
        "notes": (
            "A 2024 landmark study [7] proves vapor chambers cool 50 "
            "degrees in all cases."
        ),
        "evidence": [
            (
                "https://en.wikipedia.org/wiki/Vapor_chamber",
                "Vapor chamber",
                "Vapor chambers are heat spreading devices used in "
                "electronics cooling.",
            ),
        ],
        "checker": "fail_all",
    },
]
