"""Back-compat shim — the real code now lives in the `strategies` package.

The simple-rim strategy was split out (along with new strategies) into a
package so it's easier to find, maintain, and swap. This module preserves
the historical import paths used by callers like game.py:

    from simple_rim_strategy import SimpleRimStrategy, classify_outcome

so existing entry points keep working without churn. New code should
import directly from the package, e.g.:

    from strategies import SimpleRimStrategy, build_strategy
    from strategies.shared import classify_outcome
"""

from strategies.shared import classify_outcome
from strategies.simple import SimpleRimStrategy

__all__ = ["SimpleRimStrategy", "classify_outcome"]
