"""Generate nodewatch brand assets: a wordmark logo and a GitHub social-preview card.

Deliberately self-contained on a dark card background: a transparent logo with
light text breaks on GitHub's light theme, and a dark-text one breaks on dark
theme. A card reads correctly on both, and on PyPI.
"""
from PIL import Image, ImageDraw, ImageFont

FIRA_BOLD = "/usr/share/fonts/opentype/fira/FiraMono-Bold.otf"
FIRA_REG = "/usr/share/fonts/opentype/fira/FiraMono-Regular.otf"
SANS = "/usr/share/fonts/opentype/fira/FiraSans-Book.otf"

BG = (13, 17, 23)          # #0d1117
BORDER = (48, 54, 61)      # #30363d
EDGE = (72, 79, 88)        # #484f58
DIM = (139, 148, 158)      # #8b949e
FG = (230, 237, 243)       # #e6edf3
ACCENT = (88, 166, 255)    # #58a6ff
GREEN = (63, 185, 80)      # #3fb950


def rounded_card(w, h, radius, scale=1):
    img = Image.new("RGBA", (w * scale, h * scale), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle(
        [0, 0, w * scale - 1, h * scale - 1],
        radius=radius * scale, fill=BG, outline=BORDER, width=max(1, 2 * scale),
    )
    return img, d


def draw_graph_mark(d, cx, cy, s, watched_color=ACCENT, node_color=DIM, edge_color=EDGE):
    """A branching 3-node graph with one node 'watched' (ringed).

    Evokes a LangGraph fan-out rather than a generic network: one entry node
    splitting into two, with the measured branch highlighted.
    """
    ax, ay = cx - 1.35 * s, cy
    bx, by = cx + 0.75 * s, cy - 1.05 * s
    ccx, ccy = cx + 0.75 * s, cy + 1.05 * s

    lw = max(2, int(0.17 * s))
    d.line([ax, ay, bx, by], fill=edge_color, width=lw)
    d.line([ax, ay, ccx, ccy], fill=edge_color, width=lw)

    r_small = 0.34 * s
    for (x, y) in ((ax, ay), (bx, by)):
        d.ellipse([x - r_small, y - r_small, x + r_small, y + r_small], fill=node_color)

    # The watched node: filled, plus a concentric ring = "under observation".
    r_big = 0.46 * s
    ring = 0.78 * s
    d.ellipse(
        [ccx - ring, ccy - ring, ccx + ring, ccy + ring],
        outline=watched_color, width=max(2, int(0.10 * s)),
    )
    d.ellipse([ccx - r_big, ccy - r_big, ccx + r_big, ccy + r_big], fill=watched_color)


def make_logo(path, scale=2):
    s = scale
    H = 200
    text_x = 206          # where the wordmark starts
    pad_right = 44

    f = ImageFont.truetype(FIRA_BOLD, 74 * s)
    w_node = f.getlength("node")
    w_watch = f.getlength("watch")
    W = text_x + int((w_node + w_watch) / s) + pad_right

    img, d = rounded_card(W, H, 28, s)

    draw_graph_mark(d, 116 * s, (H // 2) * s, 34 * s)

    # anchor="lm" = left/middle, so the wordmark is optically centred on the
    # card rather than sitting on a guessed baseline.
    x, y = text_x * s, (H // 2) * s
    d.text((x, y), "node", font=f, fill=FG, anchor="lm")
    d.text((x + w_node, y), "watch", font=f, fill=ACCENT, anchor="lm")

    img.save(path)
    return img.size


def make_social(path, scale=1):
    W, H = 1280, 640
    img = Image.new("RGBA", (W, H), BG)
    d = ImageDraw.Draw(img)

    # Oversized watermark of the same mark, only a shade above the background so
    # it reads as texture rather than as a second element competing for
    # attention. Placed far right so it clears the longest text line.
    faint = (25, 31, 39)
    draw_graph_mark(d, 1140, 400, 165, watched_color=faint, node_color=faint, edge_color=faint)

    draw_graph_mark(d, 150, 205, 46)

    f_word = ImageFont.truetype(FIRA_BOLD, 96)
    x, y = 268, 152
    d.text((x, y), "node", font=f_word, fill=FG)
    x += d.textlength("node", font=f_word)
    d.text((x, y), "watch", font=f_word, fill=ACCENT)

    f_tag = ImageFont.truetype(SANS, 44)
    d.text((92, 330), "Per-node token, cost and latency tracking", font=f_tag, fill=FG)
    d.text((92, 388), "for LangGraph agents.", font=f_tag, fill=FG)

    f_sub = ImageFont.truetype(FIRA_REG, 32)
    d.text((92, 486), "one SQLite file  ·  no infrastructure  ·  MIT", font=f_sub, fill=DIM)

    # Accent rule separating the wordmark block from the tagline. Sits below the
    # mark's lowest extent (~y=300) so it cannot collide with it.
    d.rounded_rectangle([92, 302, 148, 308], radius=3, fill=GREEN)

    img.convert("RGB").save(path)
    return img.size


if __name__ == "__main__":
    print("logo  ", make_logo("/tmp/logo.png"))
    print("social", make_social("/tmp/social-preview.png"))
