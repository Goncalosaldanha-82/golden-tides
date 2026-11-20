"""Golden Tides channel logo: FLUX sunset artwork + elegant serif wordmark (800x800, safe in the circular crop)."""
import cairo
S = 800
bg = cairo.ImageSurface.create_from_png("gt_bg_11.png")
surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, S, S)
ctx = cairo.Context(surf)
k = S / bg.get_width()
ctx.save(); ctx.scale(k, k); ctx.set_source_surface(bg, 0, 0); ctx.paint(); ctx.restore()
# darken the water so the letters read clearly
g = cairo.LinearGradient(0, S * 0.5, 0, S)
g.add_color_stop_rgba(0, 0, 0, 0, 0); g.add_color_stop_rgba(0.35, 0, 0.02, 0.05, 0.55); g.add_color_stop_rgba(1, 0, 0.02, 0.05, 0.8)
ctx.set_source(g); ctx.paint()

def spaced(s, y, size, spacing, col, font="Georgia", weight=cairo.FONT_WEIGHT_NORMAL):
    ctx.select_font_face(font, cairo.FONT_SLANT_NORMAL, weight); ctx.set_font_size(size)
    w = [ctx.text_extents(ch).x_advance for ch in s]
    x = S / 2 - (sum(w) + spacing * (len(s) - 1)) / 2
    for dx, dy, c in ((3, 4, (0, 0, 0, 0.6)), (0, 0, col)):
        xx = x
        for ch, ww in zip(s, w):
            ctx.move_to(xx + dx, y + dy); ctx.set_source_rgba(*c); ctx.show_text(ch); xx += ww + spacing

gold = cairo.LinearGradient(0, S * 0.6, 0, S * 0.75)
spaced("GOLDEN", S * 0.665, 96, 14, (1.0, 0.86, 0.55, 1), weight=cairo.FONT_WEIGHT_BOLD)
spaced("TIDES", S * 0.775, 74, 26, (1.0, 0.95, 0.86, 1))
# thin gold rules either side of a small diamond
ctx.set_source_rgba(1, 0.82, 0.5, 0.9); ctx.set_line_width(2)
y = S * 0.825
ctx.move_to(S * 0.33, y); ctx.line_to(S * 0.47, y); ctx.move_to(S * 0.53, y); ctx.line_to(S * 0.67, y); ctx.stroke()
ctx.move_to(S / 2, y - 7); ctx.line_to(S / 2 + 7, y); ctx.line_to(S / 2, y + 7); ctx.line_to(S / 2 - 7, y); ctx.close_path(); ctx.fill()
surf.write_to_png("golden_tides_logo.png")
print("ok")
