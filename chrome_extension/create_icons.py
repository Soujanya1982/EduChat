"""
Generate EduChat extension icons (navy + red, graduation cap style).
Requires Pillow:  pip install pillow
Run once:         python create_icons.py
"""
from pathlib import Path
from PIL import Image, ImageDraw

NAVY = (0, 33, 71, 255)
RED  = (200, 16, 46, 255)
WHITE = (255, 255, 255, 255)

def make_icon(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d   = ImageDraw.Draw(img)

    # Rounded-square background (navy)
    r = max(2, size // 6)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=r, fill=NAVY)

    # Red bottom stripe
    stripe = max(2, size // 8)
    d.rounded_rectangle(
        [0, size - stripe, size - 1, size - 1],
        radius=r, fill=RED
    )

    # Mortarboard cap — board (top flat square rotated 45°)
    cx, cy = size / 2, size * 0.42
    cap_w  = size * 0.54
    half   = cap_w / 2

    # Diamond / rhombus for the cap top
    top_h = cap_w * 0.28
    board = [
        (cx,          cy - top_h),   # top
        (cx + half,   cy),            # right
        (cx,          cy + top_h),   # bottom
        (cx - half,   cy),            # left
    ]
    d.polygon(board, fill=WHITE)

    # Tassel knob (small red square center)
    knob = max(2, size // 14)
    d.ellipse(
        [cx - knob, cy - knob, cx + knob, cy + knob],
        fill=RED
    )

    # Cap body (trapezoid below the board)
    body_top = cy + top_h * 0.3
    body_btm = cy + cap_w * 0.36
    body_w_t = cap_w * 0.46
    body_w_b = cap_w * 0.38
    body = [
        (cx - body_w_t / 2, body_top),
        (cx + body_w_t / 2, body_top),
        (cx + body_w_b / 2, body_btm),
        (cx - body_w_b / 2, body_btm),
    ]
    d.polygon(body, fill=WHITE)

    return img


if __name__ == "__main__":
    out = Path(__file__).parent / "icons"
    out.mkdir(exist_ok=True)

    for size in (16, 48, 128):
        img = make_icon(size)
        path = out / f"icon{size}.png"
        img.save(path, "PNG")
        print(f"  OK  icons/icon{size}.png")

    print("\nDone - reload the extension in chrome://extensions")
