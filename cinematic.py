"""Cinematic music video in the style of the reference channels:
one photographic cover image, very slow zoom, elegant serif title, soft light breathing with the bass,
fade in/out. Also a vertical 1080x1920 Short (~30 s) cut from the strongest part of the song.

    python cinematic.py --audio ace/song.wav --image covers/song.png --title "HOLD ON TO THE LIGHT" --out output/afro_001
"""
import argparse
import math
import os
import subprocess
import wave
from pathlib import Path

import cairo
import imageio_ffmpeg
import numpy as np

FPS = 30
SERIF = os.environ.get("CINEMATIC_SERIF", "Georgia")   # cloud runs use Gelasio (free, Georgia-compatible)
FF = imageio_ffmpeg.get_ffmpeg_exe()


def to_wav(src, dst):
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", str(src), "-ac", "2", "-ar", "44100", str(dst)], check=True)


def load_mono(path):
    with wave.open(str(path)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").reshape(-1, w.getnchannels()) / 32768.0
    return x.mean(axis=1), 44100


def bass_env(mono, sr, nframes):
    """per-frame low-frequency energy 0..1, smoothed: drives the soft light pulse."""
    from scipy import signal
    lo = signal.sosfilt(signal.butter(2, 150, "lowpass", fs=sr, output="sos"), mono)
    hop = sr // FPS
    e = np.array([np.sqrt(np.mean(lo[i * hop:(i + 1) * hop] ** 2) + 1e-12) for i in range(nframes)])
    e = (e - np.percentile(e, 10)) / (np.percentile(e, 98) - np.percentile(e, 10) + 1e-9)
    e = np.clip(e, 0, 1)
    out = np.zeros_like(e)
    for i in range(nframes):
        out[i] = e[i] if i == 0 or e[i] > out[i - 1] else out[i - 1] * 0.88
    return out


def loudest_window(mono, sr, seconds=30):
    hop = sr // 2
    rms = np.array([np.sqrt(np.mean(mono[i:i + hop] ** 2)) for i in range(0, len(mono) - hop, hop)])
    win = int(seconds * 2)
    if len(rms) <= win:
        return 0.0
    sums = np.convolve(rms, np.ones(win), "valid")
    return float(np.argmax(sums) / 2)


def spaced(ctx, s, cx, y, size, color, alpha, spacing, font=None, weight=cairo.FONT_WEIGHT_NORMAL,
           max_w=None):
    ctx.select_font_face(font or SERIF, cairo.FONT_SLANT_NORMAL, weight)
    ctx.set_font_size(size)
    widths = [ctx.text_extents(ch).x_advance for ch in s]
    total = sum(widths) + spacing * (len(s) - 1)
    if max_w and total > max_w:                                   # long titles shrink to fit the frame
        k = max_w / total
        size, spacing = size * k, spacing * k
        ctx.set_font_size(size)
        widths = [ctx.text_extents(ch).x_advance for ch in s]
        total = sum(widths) + spacing * (len(s) - 1)
    x = cx - total / 2
    for dx, dy, a in ((3, 4, 0.45), (0, 0, 1.0)):
        xx = x
        for ch, w in zip(s, widths):
            ctx.move_to(xx + dx, y + dy)
            ctx.set_source_rgba(*(color if dy == 0 else (0, 0, 0)), alpha * a)
            ctx.show_text(ch)
            xx += w + spacing
    ctx.new_path()


def render(image, wav, out, W, H, title, subtitle, start=0.0, dur=None, focus_x=0.5, title_y=None):
    mono, sr = load_mono(wav)
    total = len(mono) / sr
    dur = dur or (total - start)
    nfr_all = int(total * FPS) + 1
    env = bass_env(mono, sr, nfr_all)
    img = cairo.ImageSurface.create_from_png(str(image))
    iw, ih = img.get_width(), img.get_height()
    cover = max(W / iw, H / ih)                                      # fill the frame (crop)
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    ctx = cairo.Context(surf)
    fade_a = ["-af", f"afade=t=in:st=0:d=0.6,afade=t=out:st={max(0, dur - 2):.2f}:d=2"]
    cmd = [FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-ss", f"{start:.2f}", "-t", f"{dur:.2f}", "-i", str(wav), *fade_a,
           "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k",
           "-shortest", "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = int(dur * FPS)
    vertical = H > W
    for i in range(n):
        t = i / FPS
        f = int((start + t) * FPS)
        p = t / max(dur, 1e-6)
        z = cover * (1.0 + 0.08 * p)                                  # very slow push-in
        ctx.save()
        ctx.translate(W / 2 + (-12 + 24 * p) + (0.5 - focus_x) * iw * z, H / 2)
        ctx.scale(z, z)
        ctx.set_source_surface(img, -iw / 2, -ih / 2)
        ctx.get_source().set_filter(cairo.FILTER_GOOD)
        ctx.paint()
        ctx.restore()
        # warm breathing light + vignette
        k = env[min(f, nfr_all - 1)]
        glow = cairo.RadialGradient(W / 2, H * 0.55, 10, W / 2, H * 0.55, max(W, H) * 0.7)
        glow.add_color_stop_rgba(0, 1.0, 0.78, 0.5, 0.06 + 0.10 * k)
        glow.add_color_stop_rgba(1, 1.0, 0.78, 0.5, 0.0)
        ctx.set_source(glow)
        ctx.paint()
        vig = cairo.RadialGradient(W / 2, H / 2, min(W, H) * 0.35, W / 2, H / 2, max(W, H) * 0.75)
        vig.add_color_stop_rgba(0, 0, 0, 0, 0)
        vig.add_color_stop_rgba(1, 0, 0, 0, 0.55)
        ctx.set_source(vig)
        ctx.paint()
        # title (fades in during the first 2 s)
        a = min(1.0, t / 2.0)
        if vertical:
            ty = title_y if title_y is not None else 0.17
            spaced(ctx, title, W / 2, H * ty, W * 0.075, (1, 0.97, 0.92), a, W * 0.008, max_w=W * 0.86)
            spaced(ctx, subtitle, W / 2, H * (ty + 0.035), W * 0.03, (1, 0.88, 0.7), a * 0.95, W * 0.012, "Arial")
        else:
            ty = title_y if title_y is not None else 0.20
            spaced(ctx, title, W / 2, H * ty, H * 0.085, (1, 0.97, 0.92), a, H * 0.012, max_w=W * 0.86)
            spaced(ctx, subtitle, W / 2, H * (ty + 0.055), H * 0.028, (1, 0.88, 0.7), a * 0.95, H * 0.014, "Arial")
        # fade from/to black
        fade = min(1.0, t / 0.8, (dur - t) / 2.0)
        if fade < 1:
            ctx.set_source_rgba(0, 0, 0, 1 - max(0.0, fade))
            ctx.paint()
        surf.flush()
        proc.stdin.write(bytes(surf.get_data()))
    proc.stdin.close()
    proc.wait()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--subtitle", default="AFRO HOUSE")
    ap.add_argument("--out", required=True)
    ap.add_argument("--focus-x", type=float, default=0.5, help="horizontal subject position (0..1) for the vertical Short crop")
    ap.add_argument("--short-only", action="store_true")
    ap.add_argument("--title-y", type=float, help="title height as a fraction of the frame (default: top)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wav = out / "audio.wav"
    to_wav(a.audio, wav)
    img = Path(a.image)
    if img.suffix.lower() != ".png":                                  # cairo reads PNG
        png = out / "cover_src.png"
        subprocess.run([FF, "-y", "-loglevel", "error", "-i", str(img), str(png)], check=True)
        img = png
    if not a.short_only:
        render(img, wav, out / "video.mp4", 1920, 1080, a.title, a.subtitle, title_y=a.title_y)
    mono, sr = load_mono(wav)
    s0 = loudest_window(mono, sr, 30)
    render(img, wav, out / "short.mp4", 1080, 1920, a.title, a.subtitle, start=s0, dur=min(30.0, len(mono) / sr - s0), focus_x=a.focus_x, title_y=a.title_y)
    print("ready:", out / "video.mp4", out / "short.mp4", f"(short from {s0:.0f}s)")


if __name__ == "__main__":
    main()
