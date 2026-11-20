"""Publish one Golden Tides release (full video + Short) to the channel in config.json.

    python publish_afro.py output/afro_001 --title "Hold On To The Light" [--dry-run]

Refuses to upload when the same audio was already published (sha256 recorded in uploaded.json files).
Music generated with ElevenLabs Music (free plan: non-commercial, credit in the description; no monetization).
"""
import argparse
import datetime
import json
from pathlib import Path

from upload_music import OUT, config, insert, service, sha256

ROOT = Path(__file__).resolve().parent


ELEVEN = "Music generated with ElevenLabs Music (elevenlabs.io)."
MORPHIX = "Cover image generated with AI (Morphix)."


def meta(title, genre="Afro House", music=ELEVEN, cover=MORPHIX,
         blurb="Warm Afro House with a soulful voice, tribal drums and an ocean-at-sunset feeling. 🌅",
         hashtags="#afrohouse #deephouse #organichouse #soulfulhouse #sunsetmusic"):
    desc = "\n".join([
        f"{title} ✦ Golden Tides ({genre})",
        "",
        blurb,
        "A new song to lift your soul — let it move you.",
        "",
        "🌍 Comment where you're listening from 👇",
        "🔔 Subscribe for more Golden Tides.",
        "",
        f"Lyrics and concept: Golden Tides. {music}",
        cover,
        hashtags,
    ])
    return {"title": f"{title} ✦ Golden Tides ({genre})"[:100], "description": desc,
            "tags": ["afro house", "deep house", "organic house", "soulful house", "melodic house",
                     "sunset music", "emotional house", "golden tides"], "categoryId": "10"}


def short_meta(title, long_url, genre="Afro House", music=ELEVEN):
    tag = "#" + genre.lower().replace(" ", "")
    return {"title": f"{title} 🌅 {genre} #shorts"[:100],
            "description": f"Full song: {long_url}\n{music}\n{tag} #shorts",
            "tags": ["shorts", "afro house", "deep house", "golden tides"], "categoryId": "10"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--title", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--genre", default="Afro House")
    ap.add_argument("--music-credit", default=ELEVEN)
    ap.add_argument("--cover-credit", default=MORPHIX)
    ap.add_argument("--blurb")
    ap.add_argument("--hashtags")
    a = ap.parse_args()
    folder = Path(a.folder)
    cfg = config()
    yt, me = service(interactive=False)
    if me["id"] != cfg["channel_id"]:
        raise SystemExit(f"token is for {me['snippet']['title']}, expected Golden Tides")
    digest = sha256(folder / "audio.wav")
    for up in list(OUT.glob("*/uploaded.json")):
        d = json.loads(up.read_text(encoding="utf-8"))
        if d.get("audio_sha256") == digest:
            raise SystemExit(f"SKIPPED DUPLICATE: same audio already published as {d.get('url')}")
    extra = {k: v for k, v in (("blurb", a.blurb), ("hashtags", a.hashtags)) if v}
    m = meta(a.title, a.genre, a.music_credit, a.cover_credit, **extra)
    print(("WOULD UPLOAD" if a.dry_run else "uploading"), m["title"], "->", me["snippet"]["title"])
    if a.dry_run:
        return
    partial = folder / "partial_upload.json"          # survives a failure between the two uploads
    state = json.loads(partial.read_text(encoding="utf-8")) if partial.exists() else {}
    if state.get("video_id"):
        vid = state["video_id"]
        print("full video already uploaded earlier:", vid, "- uploading only the Short")
    else:
        vid = insert(yt, folder / "video.mp4", m)
        partial.write_text(json.dumps({"video_id": vid}), encoding="utf-8")
    url = f"https://youtu.be/{vid}"
    sid = insert(yt, folder / "short.mp4", short_meta(a.title, url, a.genre, a.music_credit))
    partial.unlink(missing_ok=True)
    rec = {"video_id": vid, "url": url, "short_id": sid, "short_url": f"https://youtube.com/shorts/{sid}",
           "audio_sha256": digest, "channel": me["id"], "uploaded": datetime.datetime.now().isoformat()}
    (folder / "uploaded.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print("published:", url, "| short:", rec["short_url"])


if __name__ == "__main__":
    main()
