# Idleon basketball minigame — game model

This document captures how the in-game basketball minigame actually behaves. The bot's accuracy depends on these mechanics being modeled correctly; the strategies code (`strategies/`) makes assumptions that only make sense in this context.

## Two phases, gated by score

Throughout one game, the rim's behavior changes based on the current score.

### Sub-10 phase (`score < 10`) — **fixed rim, teleports on make**

- The rim is **completely stationary**.
- The rim **stays in the exact same `(x, y)` position until you make a shot**.
  - Verified in data: after a missed throw in this phase, the next throw's rim is at the same position 98.5% of the time (within 5 px).
- When you make a shot, the rim **teleports to a new fixed position** for the next shot.
  - Verified in data: after a made throw, the next throw's rim has moved 98.8% of the time.
- The rim stays at the new position until the next make, or the game ends.

Important consequence: if the bot misses, it gets *another attempt at the exact same rim position*. The correction logic (dy streaks, make quarantine, etc.) is supposed to use that repeatability — first attempt informs second attempt at the same physical target.

### Score 10+ phase — **horizontally oscillating rim**

- The rim oscillates horizontally between some `x_min` and `x_max`.
- It keeps oscillating continuously until you make a shot.
- After a make, the rim resets (presumably to new bounds / a new position) and the oscillation continues.
- Vertically the rim is essentially fixed during the oscillation (matches what `OscillationStrategy` assumes via `RIM_Y_MATCH_TOLERANCE_PX`).
- Period and amplitude are estimated live by `RimOscillationModel` (strategies/oscillation_model.py).

## Game-end rule

The game ends after **3 consecutive missed throws** (independent of phase). This is why per-throw accuracy compounds heavily into per-game peak score:

- At 50% per-throw make rate, the chance of missing 3 in a row inside any 3-shot window is ~12.5%. Average game length is short.
- Reaching score 10 (the moving-rim phase) requires never having 3 misses in a row across 10 makes — only ~14% of games in the data have managed this.
- Reaching score 20 has never happened in 517 logged games.

## What the bot records (`throws.jsonl`)

Each line is one finalized throw. Key fields:

- `game_id` — UUIDv7 reset on each game restart; all throws from one game share an id.
- `ts` — local ISO timestamp at click time.
- `ball_x`, `ball_y` — basketball position at the moment of click. Screen coordinates.
- `rim_x`, `rim_y` — rim position at the moment of click. Screen coordinates.
- `stroke` — `"up"` / `"down"` / `null`. Direction the ball was moving (in y) at click time. Computed from a 5-sample ball-y history.
- `rim_moving` — whether the rim was moving at click time. Pinned to `False` for scores below `MOVING_RIM_MIN_SCORE` regardless of tracker output, because the rim is provably stationary there and any "moving" reading is tracker jitter.
- `score` — the score AFTER this throw (so `scored=True` means score went up by ≥1).
- `scored` — `True` if the score increased after this throw resolved.
- `trajectory` — `[(ball_x, ball_y, t_ms), ...]` sampled at ~30 fps for ~2.8 s after the click.
- `rim_trajectory` — same shape, for the rim.

Outcome classification (`strategies.classify_outcome`) derives `make` / `undershoot` / `overshoot` / `no_launch` / `unknown` from the trajectory + `scored`.

## Coordinate conventions

- Screen y grows **downward**. A positive `dy = ball_y - rim_y` means the ball is **below** the rim.
- A negative `dy` means the ball is **above** the rim — the typical launch geometry, since the ball needs to arc down into the hoop.
- "Launch from a higher point" = click when `ball_y` is smaller = `dy` is more negative.

## Resolution / setup assumptions

- Coded against a 27" monitor at **1920×1080** (the comment at the top of `README.md`). All hardcoded screen rectangles in `regions.py` are physical pixels at that resolution.
- Captures via `mss` from the primary monitor. Tesseract OCR (default install path `C:\Program Files\Tesseract-OCR\`) is used as a fallback inside `ScoreReader`.

## Phases of any given game (state machine the bot navigates)

1. **Lobby** — `lobby.start_game()` clicks the basketball item, then "you know it" to enter.
2. **In-game stationary** — `score < 10`. `SimpleRimStrategy` is in charge (the `OscillationStrategy` delegates to it for this phase).
3. **In-game moving** — `score >= 10` and rim observed moving for `LATCH_CONFIRM_FRAMES` consecutive frames. `OscillationStrategy` takes over, never falls back.
4. **Game over** — detected via `is_game_over(frame, ...)`. Bot waits 2 s, flushes pending throws, clicks the exit button, then returns to step 1.
5. **Stuck state** — if no throw fires for `FORCE_THROW_AFTER_S` (120 s), the watchdog in `game.py` forces a best-effort click. If `MAX_CONSECUTIVE_DROPS` (3) throws fail to resolve a score, lobby recovery is triggered.

## Why the sub-10 phase still produces ~50% accuracy

The rim being "stationary" is necessary but not sufficient for high accuracy:

- The rim's **fixed position varies enormously across games and across makes within a game**: `rim_y` has been observed across the full 475–957 px range (a 482 px spread). Each new rim position is essentially a fresh shooting problem.
- Make data is **sparse per rim position bucket**: ~2200 stationary makes spread over ~600 distinct rim locations = ~3.7 makes per bucket. The strategy's "pick the nearest make" only works when nearby data exists.
- After a miss, the bot has another shot at *the same rim*, so corrections within a game should theoretically help — but the data shows attempt #2's make rate (40.5%) is actually *lower* than attempt #1 (48.1%). The dy correction streaks are not translating misses into better next-shot decisions; this is an active bug area.
