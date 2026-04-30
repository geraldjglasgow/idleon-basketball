import argparse

import game
from lobby import start_game
from preview_window import PreviewWindow


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Idleon basketball bot")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--debug",
        action="store_true",
        help="open the full tracker preview window during the game loop",
    )
    group.add_argument(
        "--light",
        action="store_true",
        help="open a small debug-text-only window during the game loop",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.debug:
        preview_mode = "full"
    elif args.light:
        preview_mode = "light"
    else:
        preview_mode = "off"

    print("Idleon basketball bot — Ctrl+C to quit")
    preview: PreviewWindow | None = None
    if preview_mode != "off":
        preview = PreviewWindow(lightweight=(preview_mode == "light"))
    try:
        start_game(preview=preview)
        game.run(preview=preview)
    except KeyboardInterrupt:
        print("[main] stopped")
    finally:
        if preview is not None:
            preview.close()


if __name__ == "__main__":
    main()
