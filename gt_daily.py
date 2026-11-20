"""Golden Tides daily release for GitHub Actions (free): one new song per day, public at 18:00 Lisbon time.

Steps: next song from songs_bank.json (original lyrics) -> music with YuE (open model, HF free GPU quota)
-> cover with FLUX.1-schnell -> cinematic video + Short (cinematic.py) -> upload both to Golden Tides,
private with publishAt = 18:00 Europe/Lisbon (published immediately if the run is late).

State lives in the repository (committed by the workflow): gt_published.json.
Guards: one release per Lisbon day; a song title is never used twice; same audio hash never re-uploaded.

    python gt_daily.py [--dry-run]
Needs: hf_token.txt (HF read token) and token_music.json (YouTube token of Golden Tides).
"""
import argparse
import datetime
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import imageio_ffmpeg
from gradio_client import Client
from googleapiclient.http import MediaFileUpload

import publish_afro as pa
from upload_music import config, service, sha256

ROOT = Path(__file__).resolve().parent
BANK = ROOT / "songs_bank.json"
PUBLISHED = ROOT / "gt_published.json"
LISBON = ZoneInfo("Europe/Lisbon")
HOUR = 18
FF = imageio_ffmpeg.get_ffmpeg_exe()
MUSIC_CREDIT = "Music generated with YuE (open-source AI model) on Hugging Face."
COVER_CREDIT = "Cover image generated with FLUX.1-schnell (Apache-2.0)."
FALLBACK_STYLE = {
    "Afro House": "Emotional Afro House, 122 BPM, tribal percussion, kalimba, deep bass, soulful female vocal",
    "Afro Latin": "Afro-Latin pop, 104 BPM, afrobeats and reggaeton groove, nylon guitar, warm bass, sensual female vocal",
    "Emotional Dance": "Emotional melodic dance, 124 BPM, euphoric piano, driving kick, warm lead vocal, anthemic",
}
FALLBACK_COVER = ("Cinematic photo, {concept}, golden sunset over the ocean, warm light, subject in upper half, "
                  "calm lower third, film grain, ultra realistic, no text")


def hf():
    return (ROOT / "hf_token.txt").read_text(encoding="utf-8").strip()


def slugify(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def next_song(pub):
    used = {d["title"] for d in pub.values()}
    bank = json.loads(BANK.read_text(encoding="utf-8"))
    for s in bank["songs"]:
        if s["title"] not in used:
            return dict(s, lyrics="\n".join(s["lyrics"]))
    for f in bank["fallback"]:
        if f["title"] not in used:
            print("song bank used up: writing lyrics with YuE for", f["title"])
            lang = "Spanish" if f["genre"] == "Afro Latin" else "English"
            style = FALLBACK_STYLE[f["genre"]]
            lyrics = Client("mrfakename/yue2-3b", token=hf(), verbose=False).predict(
                concept=f["concept"], style=style, language=lang,
                structure="Verse – Pre-Chorus – Chorus – Verse – Pre-Chorus – Chorus – Bridge – Chorus",
                seed=len(used), api_name="/write_lyrics")
            return {"title": f["title"], "genre": f["genre"], "style": style, "lyrics": lyrics,
                    "cover": FALLBACK_COVER.format(concept=f["concept"])}
    raise SystemExit("NO SONGS LEFT: songs_bank.json is used up — add new songs to keep the daily releases going.")


def audio_ok(path):
    r = subprocess.run([FF, "-hide_banner", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
                       capture_output=True, text=True)
    dur = re.search(r"Duration: (\d+):(\d+):([\d.]+)", r.stderr)
    vol = re.search(r"mean_volume: (-?[\d.]+) dB", r.stderr)
    secs = int(dur[1]) * 3600 + int(dur[2]) * 60 + float(dur[3]) if dur else 0
    return secs >= 60 and vol and float(vol[1]) > -30, secs


def make_music(song, dst):
    c = Client("mrfakename/yue2-3b", token=hf(), verbose=False)
    for seed in (42, 7, 1234):
        try:
            r = c.predict(style=song["style"], lyrics=song["lyrics"], planning_mode="full", render_quality=32,
                          seed=seed, api_name="/generate_song")
        except Exception as exc:
            print(f"  music attempt (seed {seed}) failed: {str(exc)[:200]}")
            continue
        shutil.copy(r[0], dst)
        ok, secs = audio_ok(dst)
        print(f"  music seed {seed}: {secs:.0f}s {'OK' if ok else 'rejected (too short or silent)'}")
        if ok:
            return
    raise SystemExit("music generation failed 3 times (HF quota or Space down) — nothing published, retry next run")


def make_cover(prompt, dst):
    c = Client("black-forest-labs/FLUX.1-schnell", token=hf(), verbose=False)
    r = c.predict(prompt=prompt, seed=7, randomize_seed=False, width=1344, height=768,
                  num_inference_steps=4, api_name="/infer")
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", r[0] if isinstance(r, (list, tuple)) else r, str(dst)],
                   check=True)


def upload(yt, path, meta, status):
    body = {"snippet": {k: meta[k] for k in ("title", "description", "tags", "categoryId")}, "status": status}
    res = yt.videos().insert(part="snippet,status", body=body,
                             media_body=MediaFileUpload(str(path), mimetype="video/mp4", resumable=True)).execute()
    return res["id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    now = datetime.datetime.now(datetime.timezone.utc)
    slot = now.astimezone(LISBON).replace(hour=HOUR, minute=0, second=0, microsecond=0)
    if now.astimezone(LISBON).hour >= HOUR + 2:        # an evening run prepares tomorrow's release
        slot += datetime.timedelta(days=1)
    day = slot.date().isoformat()
    pub = json.loads(PUBLISHED.read_text(encoding="utf-8")) if PUBLISHED.exists() else {}
    if any(d.get("publish_date") == day for d in pub.values()):
        print(f"nothing to do: a Golden Tides release is already set for {day}")
        return
    yt, me = service(interactive=False)
    if me["id"] != config()["channel_id"]:
        raise SystemExit(f"STOP: token is for {me['snippet']['title']}, not Golden Tides — nothing uploaded")
    song = next_song(pub)
    print(f"today's song: {song['title']} ({song['genre']}) -> public {day} {HOUR}:00 Lisbon")
    if a.dry_run:
        return

    folder = ROOT / "output" / f"gt_{day}_{slugify(song['title'])}"
    folder.mkdir(parents=True, exist_ok=True)
    make_music(song, folder / "song.mp3")
    make_cover(song["cover"], folder / "cover.png")
    subprocess.run([sys.executable, str(ROOT / "cinematic.py"), "--audio", str(folder / "song.mp3"),
                    "--image", str(folder / "cover.png"), "--title", song["title"].upper(),
                    "--subtitle", song["genre"].upper(), "--out", str(folder), "--title-y", "0.86"], check=True)
    digest = sha256(folder / "audio.wav")
    for d in pub.values():
        if d.get("audio_sha256") == digest:
            raise SystemExit(f"SKIPPED DUPLICATE: same audio already published as {d.get('url')}")

    status = {"privacyStatus": "public", "selfDeclaredMadeForKids": False, "containsSyntheticMedia": True}
    if slot > now + datetime.timedelta(minutes=15):
        status.update(privacyStatus="private",
                      publishAt=slot.astimezone(datetime.timezone.utc).isoformat().replace("+00:00", "Z"))
    extra = {k: song[k] for k in ("blurb", "hashtags") if song.get(k)}
    rec = {"title": song["title"], "genre": song["genre"], "publish_date": day,
           "publishAt": status.get("publishAt") or "immediately", "audio_sha256": digest,
           "uploaded": now.isoformat()}
    vid = upload(yt, folder / "video.mp4", pa.meta(song["title"], song["genre"], MUSIC_CREDIT, COVER_CREDIT, **extra),
                 status)
    rec.update(video_id=vid, url=f"https://youtu.be/{vid}")
    pub[day] = rec
    PUBLISHED.write_text(json.dumps(pub, indent=2, ensure_ascii=False), encoding="utf-8")   # saved before the Short
    sid = upload(yt, folder / "short.mp4", pa.short_meta(song["title"], rec["url"], song["genre"], MUSIC_CREDIT),
                 status)
    rec.update(short_id=sid, short_url=f"https://youtube.com/shorts/{sid}")
    PUBLISHED.write_text(json.dumps(pub, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"uploaded: {rec['url']} | short: {rec['short_url']} -> public at "
          f"{status.get('publishAt') and slot.strftime('%Y-%m-%d %H:%M Lisbon') or 'now'}")


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        _s.reconfigure(encoding="utf-8")
    main()
