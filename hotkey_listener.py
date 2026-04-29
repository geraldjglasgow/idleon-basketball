"""F1/F2 hotkeys for toggling auto-throwing.

F1 enables auto-throwing, F2 disables it. Both keys are pure toggles —
they don't start or stop the program. The lobby + game loop runs
continuously regardless; the strategy just won't issue clicks while
auto-throwing is disabled. Use Ctrl+C in the terminal to quit.

Listens globally — the bot's window doesn't need keyboard focus, the
user can stay focused on the game while toggling.
"""

from __future__ import annotations

import threading

from pynput import keyboard


class HotkeyListener:
    def __init__(self) -> None:
        self._listener: keyboard.Listener | None = None
        self._auto_enabled = False
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._listener is not None:
            return
        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.daemon = True
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    @property
    def auto_enabled(self) -> bool:
        with self._lock:
            return self._auto_enabled

    def _on_press(self, key) -> None:
        if key == keyboard.Key.f1:
            with self._lock:
                already = self._auto_enabled
                self._auto_enabled = True
            if not already:
                print("[hotkey] F1 — auto-throwing ENABLED")
        elif key == keyboard.Key.f2:
            with self._lock:
                already = self._auto_enabled
                self._auto_enabled = False
            if already:
                print("[hotkey] F2 — auto-throwing DISABLED")
