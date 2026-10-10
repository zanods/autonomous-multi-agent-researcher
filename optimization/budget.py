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
    """Per-node input token budgets. spend() trims over-limit text and tracks usage."""

    limits: dict[str, int] = field(
        default_factory=lambda: {
            "planner": 2000,
            "researcher": 6000,
            "fact_checker": 4000,
            "synthesizer": 8000,
        }
    )
    usage: dict[str, int] = field(default_factory=dict)

    def spend(self, node: str, text: str) -> str:
        n = count_tokens(text)
        self.usage[node] = self.usage.get(node, 0) + n
        limit = self.limits.get(node)
        if limit and n > limit:
            return truncate_to_tokens(text, limit)
        return text

    def total_tokens(self) -> int:
        return sum(self.usage.values())

    def estimated_cost_usd(self, model: str = "gpt-4o-mini") -> float:
        return self.total_tokens() / 1_000_000 * PRICES.get(model, 0.0)

    def report(self, model: str = "gpt-4o-mini") -> str:
        lines = [f"  {node}: {used} tokens" for node, used in self.usage.items()]
        lines.append(
            f"  total: {self.total_tokens()} tokens "
            f"(~${self.estimated_cost_usd(model):.4f} @ {model})"
        )
        return "\n".join(lines)
