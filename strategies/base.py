"""Strategy protocol — the contract `game.py` calls each frame.

Every concrete strategy (simple_rim, oscillation, future variants) implements
these four methods. Keeping the surface small lets us swap strategies at
runtime via a CLI flag without touching the game loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from basketball_tracker import BasketballSample
from rim_tracker import RimSample


class Strategy(ABC):
    # Subclasses can override to request a specific motion-history window
    # length from RimMotionTracker. The simple strategy is happy with the
    # tracker's default; oscillation needs a longer window to fit a period.
    REQUIRED_HISTORY_WINDOW_S: float | None = None

    # Subclasses set this so game.py can pass the right argument to
    # rim_motion.predict() when logging throws. Kept as an attribute so the
    # log line stays accurate if a strategy uses a different flight time.
    BALL_FLIGHT_S: float = 1.5

    @abstractmethod
    def should_throw(
        self,
        ball: BasketballSample | None,
        rim: RimSample | None,
        rim_motion=None,
    ) -> bool:
        """Return True if the loop should click a throw this frame."""

    @abstractmethod
    def mark_thrown(self) -> None:
        """Called immediately after the loop clicks a throw."""

    @abstractmethod
    def notify_outcome(self, outcome: str) -> None:
        """Called once per finalized throw with classify_outcome's verdict."""

    @abstractmethod
    def notify_score(self, score: int | None) -> None:
        """Called whenever a fresh score reading is available."""

    def notify_game_reset(self) -> None:
        """Called when game.py detects game-over or forces a lobby
        recovery. Strategies that carry sticky cross-frame state (e.g.
        the oscillation strategy's mode latch, or last-score memory)
        should clear it here — the next game starts at score 0 with a
        stationary rim, and stale state from the previous game would
        misclassify the new game's early frames.

        Default no-op for strategies that don't carry such state."""
