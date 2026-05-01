"""Strategy registry and factory.

`build_strategy(name, throws_log_path)` returns a configured strategy by
short name. Each strategy implements the `Strategy` ABC so game.py can
hold the result without caring which one it got.
"""

from __future__ import annotations

from pathlib import Path

from .base import Strategy
from .oscillation import OscillationStrategy
from .shared import classify_outcome
from .simple import SimpleRimStrategy

# Registry of strategies available via the --strategy CLI flag. Adding a
# new strategy is "register the class here" plus "import it above".
STRATEGIES: dict[str, type[Strategy]] = {
    "simple": SimpleRimStrategy,
    "oscillation": OscillationStrategy,
}


def build_strategy(name: str, throws_log_path: Path | str) -> Strategy:
    """Construct a strategy by short name. Raises KeyError on unknown
    name so the CLI parser surfaces the typo immediately."""
    try:
        cls = STRATEGIES[name]
    except KeyError as exc:
        known = ", ".join(sorted(STRATEGIES))
        raise KeyError(f"unknown strategy '{name}' (known: {known})") from exc
    return cls(throws_log_path)


__all__ = [
    "Strategy",
    "SimpleRimStrategy",
    "OscillationStrategy",
    "STRATEGIES",
    "build_strategy",
    "classify_outcome",
]
