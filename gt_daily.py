"""Golden Tides daily release for GitHub Actions (free): one new song per day, public at 06:00 Lisbon time.

The song is made in the afternoon (when the free Hugging Face GPU quota is available) and uploaded
scheduled for the NEXT 06:00 Lisbon, so YouTube publishes it at exactly 06:00.

Steps: next song from songs_bank.json (original lyrics) -> music with YuE (open model, HF free GPU quota)
-> cover with FLUX.1-schnell -> cinematic video + Short (cinematic.py) -> upload both to Golden Tides,
private with publishAt = next 06:00 Europe/Lisbon (published immediately if the run is late).

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
HOUR = 6
FF = imageio_ffmpeg.get_ffmpeg_exe()
MUSIC_CREDIT = "Music generated with YuE (open-source AI model) on Hugging Face."
COVER_CREDIT = "Cover image generated with FLUX.1-schnell (Apache-2.0)."
FALLBACK_STYLE = {
    "Afro House": "Emotional Afro House, 122 BPM, tribal percussion, kalimba, deep bass, soulful female vocal",
    "Afro Latin": "Afro-Latin pop, 104 BPM, afrobeats and reggaeton groove, nylon guitar, warm bass, sensual female vocal",
    "Emotional Dance": "Emotional melodic dance, 124 BPM, euphoric piano, driving kick, warm lead vocal, anthemic",
}
FALLBACK_COVER = "Cinematic photo of a beautiful woman, {concept}, golden sunset over the ocean, warm light"
# Channel look (user feedback): natural photo, the woman FULL BODY and never cropped, sensual; title goes over the sky.
COVER_STYLE = ("{base}. Wide full-body shot: the whole person visible head to toe with a little space below the feet, "
               "never cropped, standing in the lower part of the frame with lots of open sunset sky above. "
               "Natural candid photography, realistic skin and natural proportions, elegant and sensual, "
               "shot on 35mm film, soft golden light, sharp focus, no text, no watermark")


def cover_prompt(base):
    base = re.sub(r",?\s*subject in upper half,[^,]*in lower third", "", base)
    base = re.sub(r",?\s*(no text|film grain|ultra realistic|35mm film look)", "", base).rstrip(" ,.")
    if "no people" in base:                            # landscape covers: no person to frame
        return base + ". Natural cinematic photography, wide shot with open sky above, sharp focus, no text, no watermark"
    return COVER_STYLE.format(base=base)


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


class QuotaWait(Exception):
    """The free Hugging Face GPU quota is not available yet: not an error, the next hourly run retries."""


def quota_check(exc):
    msg = str(exc)
    if any(k in msg for k in ("ZeroGPU quota", "runs limit", "GPU duration")):
        raise QuotaWait(msg[:160]) from exc


def make_music(song, dst):
    c = Client("mrfakename/yue2-3b", token=hf(), verbose=False)
    for seed in (42, 7, 1234):
        try:
            r = c.predict(style=song["style"], lyrics=song["lyrics"], planning_mode="full", render_quality=32,
                          seed=seed, api_name="/generate_song")
        except Exception as exc:
            quota_check(exc)
            print(f"  music attempt (seed {seed}) failed: {str(exc)[:200]}")
            continue
        shutil.copy(r[0], dst)
        ok, secs = audio_ok(dst)
        print(f"  music seed {seed}: {secs:.0f}s {'OK' if ok else 'rejected (too short or silent)'}")
        if ok:
            return
    raise SystemExit("music generation failed 3 times (HF quota or Space down) — nothing published, retry next run")


def make_cover(prompt, dst, width=1344, height=768):
    c = Client("black-forest-labs/FLUX.1-schnell", token=hf(), verbose=False)
    try:
        r = c.predict(prompt=cover_prompt(prompt), seed=7, randomize_seed=False, width=width, height=height,
                      num_inference_steps=4, api_name="/infer")
    except Exception as exc:
        quota_check(exc)
        raise
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
    ap.add_argument("--today", action="store_true", help="publish right now as today's release (manual catch-up)")
    a = ap.parse_args()
    now = datetime.datetime.now(datetime.timezone.utc)
    slot = now.astimezone(LISBON).replace(hour=HOUR, minute=0, second=0, microsecond=0)
    pub = json.loads(PUBLISHED.read_text(encoding="utf-8")) if PUBLISHED.exists() else {}
    days = {d.get("publish_date") for d in pub.values()}
    if slot.date().isoformat() in days or now.astimezone(LISBON).hour >= 12:
        slot += datetime.timedelta(days=1)            # prepare tomorrow's 06:00 release
    # (a missed day is only filled in the morning; later in the day the song goes to the next 06:00, so
    #  releases stay at 06:00 even though the free GPU quota renews on a rolling 24 h window)
    if a.today:                                       # manual catch-up: publish now as today's release
        slot = now.astimezone(LISBON)
    day = slot.date().isoformat()
    if day in days:
        print(f"nothing to do: a Golden Tides release is already set for {day}")
        return
    yt, me = service(interactive=False)
    if me["id"] != config()["channel_id"]:
        raise SystemExit(f"STOP: token is for {me['snippet']['title']}, not Golden Tides — nothing uploaded")
    song = next_song(pub)
    print(f"today's song: {song['title']} ({song['genre']}) -> public {day} {HOUR:02d}:00 Lisbon")
    if a.dry_run:
        return

    folder = ROOT / "output" / f"gt_{day}_{slugify(song['title'])}"
    folder.mkdir(parents=True, exist_ok=True)
    if song.get("audio"):                              # song already made (e.g. by hand), stored in the repo
        shutil.copy(ROOT / song["audio"], folder / "song.mp3")
    else:
        make_music(song, folder / "song.mp3")
    make_cover(song["cover"], folder / "cover.png")
    make_cover(song["cover"], folder / "cover_short.png", width=768, height=1344)
    subprocess.run([sys.executable, str(ROOT / "cinematic.py"), "--audio", str(folder / "song.mp3"),
                    "--image", str(folder / "cover.png"), "--title", song["title"].upper(),
                    "--subtitle", song["genre"].upper(), "--out", str(folder),
                    "--short-image", str(folder / "cover_short.png"), "--title-y", "0.88"], check=True)
    # title low, over the sand: the covers show the woman full body, so the top is often her face
    digest = sha256(folder / "audio.wav")
    for d in pub.values():
        if d.get("audio_sha256") == digest:
            raise SystemExit(f"SKIPPED DUPLICATE: same audio already published as {d.get('url')}")

    status = {"privacyStatus": "public", "selfDeclaredMadeForKids": False, "containsSyntheticMedia": True}
    if slot > now + datetime.timedelta(minutes=15):
        status.update(privacyStatus="private",
                      publishAt=slot.astimezone(datetime.timezone.utc).isoformat().replace("+00:00", "Z"))
    extra = {k: song[k] for k in ("blurb", "hashtags") if song.get(k)}
    music = song.get("music_credit") or MUSIC_CREDIT
    rec = {"title": song["title"], "genre": song["genre"], "publish_date": day,
           "publishAt": status.get("publishAt") or "immediately", "audio_sha256": digest,
           "uploaded": now.isoformat()}
    vid = upload(yt, folder / "video.mp4", pa.meta(song["title"], song["genre"], music, COVER_CREDIT, **extra),
                 status)
    rec.update(video_id=vid, url=f"https://youtu.be/{vid}")
    pub[day] = rec
    PUBLISHED.write_text(json.dumps(pub, indent=2, ensure_ascii=False), encoding="utf-8")   # saved before the Short
    sid = upload(yt, folder / "short.mp4", pa.short_meta(song["title"], rec["url"], song["genre"], music),
                 status)
    rec.update(short_id=sid, short_url=f"https://youtube.com/shorts/{sid}")
    PUBLISHED.write_text(json.dumps(pub, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"uploaded: {rec['url']} | short: {rec['short_url']} -> public at "
          f"{status.get('publishAt') and slot.strftime('%Y-%m-%d %H:%M Lisbon') or 'now'}")


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        _s.reconfigure(encoding="utf-8")
    import os, time
    for attempt in range(2):
        try:
            main()
            break
        except QuotaWait as w:
            m = re.search(r"Try again in (\d+):(\d+):(\d+)", str(w))
            wait = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) + 60 if m else None
            # in the cloud, wait inside the run when the quota is back within ~5.5 h (GitHub's hourly
            # schedule is not reliable); otherwise stop quietly and let a later run try again
            if attempt == 0 and wait and wait <= 5.5 * 3600 and os.environ.get("GT_WAIT_FOR_QUOTA"):
                print(f"free GPU quota back in {wait // 60} min: waiting inside this run")
                time.sleep(wait)
                continue
            print(f"waiting for the free GPU quota, a later run will retry: {w}")
            break
