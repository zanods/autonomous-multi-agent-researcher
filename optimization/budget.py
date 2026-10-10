from __future__ import annotations

from dataclasses import dataclass, field

from optimization.tokens import count_tokens, truncate_to_tokens

# USD per 1M input tokens — approximate, check current provider pricing.
PRICES = {
    "gpt-4o-mini": 0.15,
    "gpt-4o": 2.50,
    "gemini-2.0-flash": 0.10,
}


@dataclass
class TokenBudget:
    """Per-node input token budgets. spend() trims over-limit text and tracks usage.

    Two kinds of numbers are kept:
    - usage: heuristic estimates of input text (used for trimming).
    - measured: API-reported input/output tokens per node, via record().
      The report prefers measured numbers when they exist.
    """

    limits: dict[str, int] = field(
        default_factory=lambda: {
            "planner": 2000,
            "researcher": 6000,
            "fact_checker": 4000,
            "synthesizer": 8000,
        }
    )
    usage: dict[str, int] = field(default_factory=dict)
    measured: dict[str, dict[str, int]] = field(default_factory=dict)

    def spend(self, node: str, text: str) -> str:
        n = count_tokens(text)
        self.usage[node] = self.usage.get(node, 0) + n
        limit = self.limits.get(node)
        if limit and n > limit:
            return truncate_to_tokens(text, limit)
        return text

    def record(self, node: str, response) -> None:
        """Accumulate API-reported token usage from a LangChain response message.

        Reads the standard `usage_metadata` first, then falls back to the
        OpenAI-style `response_metadata["token_usage"]`. Silently skips
        responses that carry no usage info (e.g. mock mode).
        """
        usage = self._extract_usage(response)
        if usage is None:
            return
        slot = self.measured.setdefault(node, {"input": 0, "output": 0})
        slot["input"] += usage[0]
        slot["output"] += usage[1]

    @staticmethod
    def _extract_usage(response) -> tuple[int, int] | None:
        meta = getattr(response, "usage_metadata", None) or {}
        if meta.get("input_tokens") is not None or meta.get("output_tokens") is not None:
            return (meta.get("input_tokens") or 0, meta.get("output_tokens") or 0)
        raw = getattr(response, "response_metadata", None) or {}
        tu = raw.get("token_usage") or {}
        if tu:
            return (tu.get("prompt_tokens") or 0, tu.get("completion_tokens") or 0)
        return None

    def total_tokens(self) -> int:
        return sum(self.usage.values())

    def total_measured(self) -> int:
        return sum(v["input"] + v["output"] for v in self.measured.values())

    def estimated_cost_usd(self, model: str = "gpt-4o-mini") -> float:
        return self.total_tokens() / 1_000_000 * PRICES.get(model, 0.0)

    def measured_cost_usd(self, model: str = "gpt-4o-mini") -> float:
        return self.total_measured() / 1_000_000 * PRICES.get(model, 0.0)

    def report(self, model: str = "gpt-4o-mini") -> str:
        if self.measured:
            lines = [
                f"  {node}: {m['input'] + m['output']} tokens "
                f"(in {m['input']} / out {m['output']})"
                for node, m in self.measured.items()
            ]
            total = self.total_measured()
            lines.append(
                f"  total: {total} tokens "
                f"(~${self.measured_cost_usd(model):.4f} @ {model}) [measured]"
            )
            return "\n".join(lines)
        lines = [f"  {node}: {used} tokens (est.)" for node, used in self.usage.items()]
        lines.append(
            f"  total: {self.total_tokens()} tokens "
            f"(~${self.estimated_cost_usd(model):.4f} @ {model}) [estimated]"
        )
        return "\n".join(lines)
