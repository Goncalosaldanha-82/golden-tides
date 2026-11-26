"""Golden Tides videoclip: a story told in scenes, cut to the music, with animated lyrics.

    python clip.py --audio song.mp3 --scenes out/scenes --title "RISE LIKE THE TIDE" --subtitle "AFRO HOUSE" \
                   --out out/ [--lrc song.lrc] [--hook "I RISE LIKE THE TIDE"]

--scenes holds scene_01.png, scene_02.png ... (16:9, the story in order) and optionally v_01.png ... (9:16 versions
used by the Short, so people are never cropped). Without an LRC file the lyrics are not shown (only title/hook).

How it is cut (from the viral analysis): long dreamy holds with slow camera moves in the calm parts; in the
high-energy parts (chorus/drop) hard cuts on the bar with a light flash and a kick "punch" on every beat.
Outputs video.mp4 (1920x1080), short.mp4 (1080x1920, from the strongest 30 s) and thumbnail.jpg.
"""
import argparse
import math
import os
import re
import subprocess
from pathlib import Path

import cairo
import imageio_ffmpeg
import numpy as np

FPS = 30
FF = imageio_ffmpeg.get_ffmpeg_exe()
DISPLAY = os.environ.get("CLIP_DISPLAY_FONT", "Impact")      # cloud: Anton (free); Windows: Impact
TEXT = os.environ.get("CLIP_TEXT_FONT", "Arial Black")      # cloud: Liberation Sans bold


# ----------------------------------------------------------------------------- audio analysis
def analyse(wav):
    import librosa
    y, sr = librosa.load(str(wav), sr=22050, mono=True)
    dur = len(y) / sr
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    beats = np.asarray(beats, float)
    if len(beats) < 8:                                         # no clear beat: fake a 120 BPM grid
        beats = np.arange(0, dur, 0.5)
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    ot = librosa.times_like(onset, sr=sr)
    strength = np.interp(beats, ot, onset)
    phase = int(np.argmax([strength[k::4].mean() for k in range(4)]))   # which beat of 4 is the downbeat
    bars = beats[phase::4]
    rms = librosa.feature.rms(y=y)[0]
    rt = librosa.times_like(rms, sr=sr)
    k = max(1, int(2.0 * sr / 512))
    sm = np.convolve(rms, np.ones(k) / k, "same")
    lo, hi = np.percentile(sm, 10), np.percentile(sm, 95)
    energy = np.clip((sm - lo) / (hi - lo + 1e-9), 0, 1)
    return {"dur": dur, "beats": beats, "bars": bars, "energy_t": rt, "energy": energy,
            "tempo": float(np.atleast_1d(tempo)[0])}


def energy_at(a, t):
    return float(np.interp(t, a["energy_t"], a["energy"]))


def schedule(a, n_scenes):
    """List of shots (start, end, scene_index, hot). Scenes follow the story in order across the song;
    hot (high-energy) stretches are cut every 2 bars, alternating with neighbouring scenes."""
    bars = list(a["bars"]) + [a["dur"]]
    if bars[0] > 0.05:
        bars = [0.0] + bars
    blocks = [bars[i] for i in range(0, len(bars), 4)]        # 4-bar phrases
    if blocks[-1] < a["dur"]:
        blocks.append(a["dur"])
    nblk = len(blocks) - 1
    shots = []
    for b in range(nblk):
        s, e = blocks[b], blocks[b + 1]
        story = min(n_scenes - 1, int(b * n_scenes / nblk))
        hot = energy_at(a, (s + e) / 2) > 0.62
        if not hot:
            shots.append((s, e, story, False))
            continue
        sub = [t for t in bars if s <= t < e] + [e]
        for j in range(0, len(sub) - 1, 2):                    # cut every 2 bars
            idx = story if (j // 2) % 2 == 0 else (story + 1 + (b + j) % (n_scenes - 1)) % n_scenes
            shots.append((sub[j], sub[min(j + 2, len(sub) - 1)], idx, True))
    return [x for x in shots if x[1] - x[0] > 0.05]


def strongest_window(a, seconds=30):
    t = np.arange(0, max(0.0, a["dur"] - seconds), 0.5)
    if not len(t):
        return 0.0
    score = [np.mean([energy_at(a, x) for x in np.arange(s, s + seconds, 1.0)]) for s in t]
    s = float(t[int(np.argmax(score))])
    near = [b for b in a["bars"] if abs(b - s) < 4]            # start on a bar
    return float(near[0]) if near else s


# ----------------------------------------------------------------------------- lyrics
def read_lrc(path, dur):
    lines = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        for m in re.finditer(r"\[(\d+):(\d+(?:\.\d+)?)\]", raw):
            text = re.sub(r"\[[^\]]*\]", "", raw).strip()
            if text and not text.startswith("("):
                lines.append((int(m[1]) * 60 + float(m[2]), text))
    lines.sort()
    out = []
    for i, (t, text) in enumerate(lines):
        end = lines[i + 1][0] if i + 1 < len(lines) else min(dur, t + 5)
        out.append((t, min(end, t + 7), text))
    return out


# ----------------------------------------------------------------------------- drawing helpers
def load(path):
    return cairo.ImageSurface.create_from_png(str(path))


def paint_cover(ctx, img, W, H, zoom, dx, dy):
    iw, ih = img.get_width(), img.get_height()
    s = max(W / iw, H / ih) * zoom
    ctx.save()
    ctx.translate(W / 2 + dx * W, H / 2 + dy * H)
    ctx.scale(s, s)
    ctx.set_source_surface(img, -iw / 2, -ih / 2)
    ctx.get_source().set_filter(cairo.FILTER_GOOD)
    ctx.paint()
    ctx.restore()


def text_w(ctx, s, font, size, weight=cairo.FONT_WEIGHT_BOLD, spacing=0.0):
    ctx.select_font_face(font, cairo.FONT_SLANT_NORMAL, weight)
    ctx.set_font_size(size)
    return sum(ctx.text_extents(ch).x_advance for ch in s) + spacing * max(0, len(s) - 1)


def draw_text(ctx, s, cx, y, font, size, rgb, alpha, spacing=0.0, glow=0.0, max_w=None,
              weight=cairo.FONT_WEIGHT_BOLD, colors=None):
    w = text_w(ctx, s, font, size, weight, spacing)
    if max_w and w > max_w:
        k = max_w / w
        size, spacing = size * k, spacing * k
        w = text_w(ctx, s, font, size, weight, spacing)
    x0 = cx - w / 2
    passes = [(0, size * 0.05, (0, 0, 0), 0.65)]
    if glow:
        passes = [(r, 0, (1, 0.75, 0.4), glow / (1 + r)) for r in (6, 3)] + passes
    passes.append((0, 0, None, 1.0))
    for spread, off, col, a in passes:
        offsets = [(0, 0)] if not spread else [(math.cos(k) * spread, math.sin(k) * spread) for k in np.linspace(0, 6.28, 8)]
        for ox, oy in offsets:
            x = x0
            for i, ch in enumerate(s):
                c = col or (colors[i] if colors else rgb)
                ctx.set_source_rgba(*c, alpha * a)
                ctx.move_to(x + ox + off, y + oy + off)
                ctx.show_text(ch)
                x += ctx.text_extents(ch).x_advance + spacing
    ctx.new_path()


def wrap(ctx, s, font, size, max_w):
    words, lines, cur = s.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if cur and text_w(ctx, t, font, size) > max_w:
            lines.append(cur)
            cur = w
        else:
            cur = t
    if cur:
        lines.append(cur)
    return lines


# ----------------------------------------------------------------------------- render
def render(a, shots, imgs, wav, out, W, H, title, subtitle, lyrics, hook=None, start=0.0, dur=None, short=False):
    dur = dur or a["dur"] - start
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    ctx = cairo.Context(surf)
    rng = np.random.default_rng(7)
    grains = []
    for _ in range(6):                                         # film grain textures
        g = cairo.ImageSurface(cairo.FORMAT_ARGB32, W // 3, H // 3)
        arr = np.ndarray((H // 3, g.get_stride() // 4), np.uint32, g.get_data())
        v = np.clip(rng.normal(128, 40, (H // 3, W // 3)), 0, 255).astype(np.uint32)
        arr[:, :W // 3] = (255 << 24) | (v << 16) | (v << 8) | v
        g.mark_dirty()
        grains.append(g)
    cmd = [FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(wav),
           "-af", f"afade=t=in:st=0:d=0.4,afade=t=out:st={max(0, dur - 2):.2f}:d=2",
           "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k",
           "-shortest", "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    beats = a["beats"]
    bar_bound = 0.11 if not short else 0.0                     # cinematic letterbox (long video only)
    for i in range(int(dur * FPS)):
        t = start + i / FPS
        k = next((j for j, s in enumerate(shots) if s[0] <= t < s[1]), len(shots) - 1)
        s0, s1, idx, hot = shots[k]
        p = (t - s0) / max(1e-6, s1 - s0)
        # camera: slow push / drift, direction alternates per shot; beat punch when hot
        bprev = beats[beats <= t]
        since = t - bprev[-1] if len(bprev) else 9
        punch = 0.035 * math.exp(-since / 0.12) if hot else 0.0
        zoom = 1.06 + 0.10 * p * (1 if k % 2 == 0 else -1) * 0.6 + punch + (0.06 if k % 2 else 0)
        dx = (p - 0.5) * 0.04 * (1 if k % 3 == 0 else -1)
        dy = (0.5 - p) * 0.02 * (1 if k % 2 else -1)
        ctx.set_source_rgb(0, 0, 0)
        ctx.paint()
        paint_cover(ctx, imgs[idx % len(imgs)], W, H, zoom, dx, dy)
        # crossfade from the previous shot in calm parts
        if k > 0 and not hot and t - s0 < 0.8:
            pv = shots[k - 1]
            ctx.push_group()
            paint_cover(ctx, imgs[pv[2] % len(imgs)], W, H, 1.06 + 0.06, 0, 0)
            ctx.pop_group_to_source()
            ctx.paint_with_alpha(1 - (t - s0) / 0.8)
        # warm grade + energy glow + vignette
        e = energy_at(a, t)
        glow = cairo.RadialGradient(W / 2, H * 0.45, 10, W / 2, H * 0.45, max(W, H) * 0.75)
        glow.add_color_stop_rgba(0, 1.0, 0.72, 0.42, 0.05 + 0.12 * e)
        glow.add_color_stop_rgba(1, 1.0, 0.72, 0.42, 0.0)
        ctx.set_source(glow)
        ctx.paint()
        vig = cairo.RadialGradient(W / 2, H / 2, min(W, H) * 0.3, W / 2, H / 2, max(W, H) * 0.78)
        vig.add_color_stop_rgba(0, 0, 0, 0, 0)
        vig.add_color_stop_rgba(1, 0, 0, 0, 0.62)
        ctx.set_source(vig)
        ctx.paint()
        # flash on a hard cut
        if hot and t - s0 < 0.2:
            ctx.set_source_rgba(1, 0.95, 0.88, 0.35 * (1 - (t - s0) / 0.2))
            ctx.paint()
        # grain
        ctx.save()
        ctx.scale(3, 3)
        ctx.set_source_surface(grains[i % len(grains)], 0, 0)
        ctx.set_operator(cairo.OPERATOR_OVERLAY)
        ctx.paint_with_alpha(0.10)
        ctx.set_operator(cairo.OPERATOR_OVER)
        ctx.restore()
        if bar_bound:
            ctx.set_source_rgb(0, 0, 0)
            ctx.rectangle(0, 0, W, H * bar_bound)
            ctx.rectangle(0, H * (1 - bar_bound), W, H * bar_bound)
            ctx.fill()
        # ---------- text
        lt = t - start
        if not short and lt < 6:                                # big title intro
            al = min(1, lt / 0.8) * min(1, (6 - lt) / 0.8)
            draw_text(ctx, title, W / 2, H * 0.52, DISPLAY, H * 0.15, (1, 0.97, 0.9), al, spacing=H * 0.012,
                      glow=0.18, max_w=W * 0.86, weight=cairo.FONT_WEIGHT_NORMAL)
            draw_text(ctx, subtitle, W / 2, H * 0.60, TEXT, H * 0.03, (1, 0.82, 0.5), al * 0.95, spacing=H * 0.012)
        if short and hook and lt < dur:                         # Short: hook on screen all the way
            al = min(1, lt / 0.4)
            lines = wrap(ctx, hook, DISPLAY, W * 0.12, W * 0.86)
            for n, ln in enumerate(lines[:3]):
                draw_text(ctx, ln, W / 2, H * 0.16 + n * W * 0.13, DISPLAY, W * 0.12, (1, 0.97, 0.9), al,
                          spacing=W * 0.006, glow=0.15, max_w=W * 0.9, weight=cairo.FONT_WEIGHT_NORMAL)
            draw_text(ctx, title + "  ·  " + subtitle, W / 2, H * 0.93, TEXT, W * 0.032, (1, 0.85, 0.55), 0.9,
                      spacing=W * 0.004, max_w=W * 0.9)
        cur = next(((s, e, txt) for s, e, txt in lyrics if s <= t < e), None)
        if cur and (short or lt >= 5.0):
            s, e, txt = cur
            al = min(1, (t - s) / 0.15, (e - t) / 0.2)
            size = (H * 0.062) if not short else W * 0.075
            lines = wrap(ctx, txt.upper() if short else txt, TEXT, size, W * 0.84)
            words_total = max(1, len(txt.split()))
            spoken = int(words_total * min(1, (t - s) / max(0.5, (e - s) * 0.8))) + 1
            base_y = H * (0.80 if not short else 0.70) - (len(lines) - 1) * size * 1.25
            wcount = 0
            for n, ln in enumerate(lines):
                cols, wi = [], wcount
                for ch in ln:
                    cols.append((1, 0.8, 0.35) if wi < spoken else (1, 1, 1))
                    if ch == " ":
                        wi += 1
                wcount += len(ln.split())
                draw_text(ctx, ln, W / 2, base_y + n * size * 1.25, TEXT, size, (1, 1, 1), al, colors=cols,
                          max_w=W * 0.9)
        # fade in/out
        f = min(1.0, (i / FPS) / 0.6, (dur - i / FPS) / 1.5)
        if f < 1:
            ctx.set_source_rgba(0, 0, 0, 1 - max(0.0, f))
            ctx.paint()
        surf.flush()
        proc.stdin.write(bytes(surf.get_data()))
    proc.stdin.close()
    proc.wait()


def thumbnail(imgs, idx, title, subtitle, out):
    W, H = 1280, 720
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    ctx = cairo.Context(surf)
    paint_cover(ctx, imgs[idx], W, H, 1.04, 0, 0)
    g = cairo.LinearGradient(0, H * 0.45, 0, H)
    g.add_color_stop_rgba(0, 0, 0, 0, 0)
    g.add_color_stop_rgba(1, 0, 0, 0, 0.75)
    ctx.set_source(g)
    ctx.paint()
    draw_text(ctx, title, W / 2, H * 0.80, DISPLAY, H * 0.2, (1, 0.97, 0.9), 1, spacing=H * 0.01, glow=0.25,
              max_w=W * 0.92, weight=cairo.FONT_WEIGHT_NORMAL)
    draw_text(ctx, subtitle, W / 2, H * 0.91, TEXT, H * 0.045, (1, 0.82, 0.5), 1, spacing=H * 0.012)
    png = Path(out).with_suffix(".png")
    surf.write_to_png(str(png))
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", str(png), "-q:v", "2", str(out)], check=True)
    png.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--subtitle", default="AFRO HOUSE")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lrc")
    ap.add_argument("--hook")
    ap.add_argument("--plan", action="store_true", help="only print the shot plan and the Short's scenes")
    ap.add_argument("--short-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wav = out / "audio.wav"
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", a.audio, "-ac", "2", "-ar", "44100", str(wav)], check=True)
    an = analyse(wav)
    sc = Path(a.scenes)
    h_imgs = sorted(sc.glob("scene_*.png"))
    v_imgs = sorted(sc.glob("v_*.png"))
    shots = schedule(an, len(h_imgs))
    s0 = strongest_window(an, 30)
    short_dur = min(30.0, an["dur"] - s0)
    used = sorted({s[2] for s in shots if s[1] > s0 and s[0] < s0 + short_dur})
    print(f"tempo {an['tempo']:.0f} BPM, {len(shots)} shots, short from {s0:.1f}s uses scenes {[u + 1 for u in used]}")
    if a.plan:
        return
    lyrics = read_lrc(a.lrc, an["dur"]) if a.lrc and Path(a.lrc).exists() else []
    title, sub = a.title.upper(), a.subtitle.upper()
    H = [load(p) for p in h_imgs]
    if not a.short_only:
        render(an, shots, H, wav, out / "video.mp4", 1920, 1080, title, sub, lyrics)
    V = [load(p) for p in v_imgs] if len(v_imgs) == len(h_imgs) else H
    render(an, shots, V, wav, out / "short.mp4", 1080, 1920, title, sub, lyrics, hook=(a.hook or title).upper(),
           start=s0, dur=short_dur, short=True)
    hot_scene = max(range(len(H)), key=lambda j: sum(e - s for s, e, x, hot in shots if x == j and hot) or 0)
    thumbnail(H, hot_scene, title, sub, out / "thumbnail.jpg")
    print("ready:", out / "video.mp4", out / "short.mp4", out / "thumbnail.jpg")


if __name__ == "__main__":
    main()
