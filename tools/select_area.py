"""Drag-select a screen region and print its top/left/width/height.

Press and drag with the left mouse button to draw a rectangle.
Release to print the region. Press Escape to cancel.
"""

import ctypes
import sys
import tkinter as tk

# Make Windows hand us physical pixel coordinates so the values match
# what other tools (PIL.ImageGrab, pyautogui, win32 APIs) will expect.
if sys.platform == "win32":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PER_MONITOR_AWARE_V2
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


class AreaSelector:
    def __init__(self):
        self.root = tk.Tk()
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-alpha", 0.3)
        self.root.attributes("-topmost", True)
        self.root.configure(bg="black")
        self.root.config(cursor="cross")

        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.start_screen = None
        self.start_canvas = None
        self.rect_id = None
        self.result = None

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.root.bind("<Escape>", lambda _e: self.root.destroy())

    def _on_press(self, event):
        self.start_screen = (event.x_root, event.y_root)
        self.start_canvas = (event.x, event.y)
        self.rect_id = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline="red", width=2
        )

    def _on_drag(self, event):
        if self.rect_id is None:
            return
        sx, sy = self.start_canvas
        self.canvas.coords(self.rect_id, sx, sy, event.x, event.y)

    def _on_release(self, event):
        if self.start_screen is None:
            return
        sx, sy = self.start_screen
        x1, x2 = sorted([sx, event.x_root])
        y1, y2 = sorted([sy, event.y_root])
        self.result = {
            "left": x1,
            "top": y1,
            "width": x2 - x1,
            "height": y2 - y1,
        }
        self.root.destroy()

    def select(self):
        self.root.mainloop()
        return self.result


def main():
    region = AreaSelector().select()
    if region is None or region["width"] == 0 or region["height"] == 0:
        print("Cancelled (no region selected)")
        return 1
    print(
        f"left={region['left']}, top={region['top']}, "
        f"width={region['width']}, height={region['height']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
