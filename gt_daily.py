"""Golden Tides daily release for GitHub Actions (free): one new song per day, public at 06:00 Lisbon time.

The song is made in the afternoon (when the free Hugging Face GPU quota is available) and uploaded
scheduled for the NEXT 06:00 Lisbon, so YouTube publishes it at exactly 06:00.

Steps: next song from songs_bank.json (original lyrics + a storyboard of scenes) -> music with ACE-Step 1.5
(open model, HF free GPU quota; YuE as backup) -> one FLUX.1-schnell image per scene -> videoclip cut to the
music with animated lyrics (clip.py) + Short + thumbnail -> upload to Golden Tides, private with
publishAt = next 06:00 Europe/Lisbon. Work in progress is kept in pending/<song>/ (committed by the workflow),
so when the free quota runs out halfway the next run carries on from there.

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
ACE_CREDIT = "Music generated with ACE-Step 1.5 (open-source AI model, Apache-2.0) on Hugging Face."
COVER_CREDIT = "Images generated with FLUX.1-schnell (Apache-2.0)."
PENDING = ROOT / "pending"
ACE_SPACES = ["Kansasfan/Ace-Step-v1.5", "ACE-Step/Ace-Step-v1.5"]
SCENE_STYLE = ("Cinematic music video still, {scene}. {character}. Natural candid photography, realistic skin and "
               "natural proportions, dramatic cinematic lighting, shallow depth of field, rich colour grade, "
               "anamorphic film look, sharp focus, no text, no watermark")
SHOT_VARIANTS = ["wide establishing shot, {cover}", "full-body shot dancing, {cover}",
                 "full-body shot from behind looking at the horizon, {cover}",
                 "atmospheric detail shot of the place, no people, {cover}",
                 "full-body shot at night under dramatic light, {cover}", "golden hour wide shot, {cover}"]
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


AUDIO_EXT = (".mp3", ".flac", ".wav", ".ogg", ".m4a")


def find_files(x, exts):
    """Every local file path or URL with one of exts inside a Gradio result (dicts, lists, update objects)."""
    found = []
    if isinstance(x, str):
        if x.lower().split("?")[0].endswith(exts):
            found.append(x)
    elif isinstance(x, dict):
        for k in ("path", "value", "name", "url", "orig_name"):
            if k in x:
                found += find_files(x[k], exts)
    elif isinstance(x, (list, tuple)):
        for y in x:
            found += find_files(y, exts)
    return found


def fetch(src, dst):
    if str(src).startswith("http"):
        import urllib.request
        with urllib.request.urlopen(src, timeout=120) as r, open(dst, "wb") as f:
            f.write(r.read())
    else:
        shutil.copy(src, dst)


def ace_music(song, folder):
    """ACE-Step 1.5 XL (the best free open model). Saves song.mp3 and the lyrics timing (song.lrc) when given.
    The Space returns the audio as 'Sample 1' and/or in the 'All Generated Files' list; the lyrics timestamps
    come in 'Lyrics Timestamps (Sample 1)' (output 28)."""
    for space in ACE_SPACES:
        for model in ("acestep-v15-xl-turbo", "acestep-v15-turbo"):
            try:
                c = Client(space, token=hf(), verbose=False)
                ep = c.view_api(return_format="dict", print_info=False)["named_endpoints"]["/generation_wrapper"]
                kw = {p["parameter_name"]: p.get("parameter_default") for p in ep["parameters"]}
                kw.update(selected_model=model, generation_mode="custom", simple_query_input="",
                          simple_vocal_language=song.get("lang", "en"), param_4=song["style"],
                          param_5=song["lyrics"], param_6=float(song.get("bpm", 0)), param_9=song.get("lang", "en"),
                          param_14=None, param_15=float(song.get("seconds", 150)), param_16=1, param_17=None,
                          param_18="", param_19=0.0, param_20=-1.0, param_44=True, param_46=1.0,
                          param_47="vocals", param_48=[])
                r = c.predict(**kw, api_name="/generation_wrapper")
            except Exception as exc:
                quota_check(exc)
                print(f"  ACE-Step {model} on {space} failed: {str(exc)[:200]}")
                if "duration" in str(exc).lower():
                    continue                            # too long for the free tier: try the smaller model
                break
            (folder / "ace_result.txt").write_text(repr(r)[:20000], encoding="utf-8")   # for diagnosis
            audio = find_files(list(r[:9]), AUDIO_EXT)
            if not audio:
                print(f"  ACE-Step {model} on {space}: no audio file in the answer ({str(r[10])[:160]})")
                break
            fetch(audio[0], folder / "song.mp3")
            ok, secs = audio_ok(folder / "song.mp3")
            print(f"  ACE-Step {model} ({space}): {secs:.0f}s {'OK' if ok else 'rejected'}")
            if not ok:
                break
            for x in r[28:29] + r[9:12]:                # lyrics timestamps (LRC-style) if present
                txt = x if isinstance(x, str) else ""
                if len(txt) < 400 and txt.endswith(".lrc") and Path(txt).exists():
                    txt = Path(txt).read_text(encoding="utf-8")
                if re.search(r"\[\d+:\d+(\.\d+)?\]|\d+(\.\d+)?\s*[-–>]+\s*\d+(\.\d+)?|\"start\"", txt):
                    (folder / "song.lrc").write_text(txt, encoding="utf-8")
                    print("  lyrics timing saved")
                    break
            return ACE_CREDIT.replace("ACE-Step 1.5", "ACE-Step 1.5 XL" if "xl" in model else "ACE-Step 1.5")
    return None


def make_image(prompt, dst, width=1344, height=768, seed=7):
    c = Client("black-forest-labs/FLUX.1-schnell", token=hf(), verbose=False)
    try:
        r = c.predict(prompt=prompt, seed=seed, randomize_seed=False, width=width, height=height,
                      num_inference_steps=4, api_name="/infer")
    except Exception as exc:
        quota_check(exc)
        raise
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", r[0] if isinstance(r, (list, tuple)) else r, str(dst)],
                   check=True)


def scene_prompts(song):
    scenes = song.get("scenes") or [v.format(cover=song["cover"]) for v in SHOT_VARIANTS]
    character = song.get("character", "")
    return [SCENE_STYLE.format(scene=sc, character=character if "no people" not in sc else "") for sc in scenes]


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
    if song.get("publish_when_ready"):                 # one-off request: publish as soon as it is made
        slot, day = now.astimezone(LISBON), now.astimezone(LISBON).date().isoformat()
    print(f"today's song: {song['title']} ({song['genre']}) -> public {day} {HOUR:02d}:00 Lisbon")
    if a.dry_run:
        return

    folder = PENDING / slugify(song["title"])          # committed between runs: resumes after a quota stop
    folder.mkdir(parents=True, exist_ok=True)
    credit_file = folder / "music_credit.txt"
    if not (folder / "song.mp3").exists():
        if song.get("audio"):                          # song already made (e.g. by hand), stored in the repo
            shutil.copy(ROOT / song["audio"], folder / "song.mp3")
            credit_file.write_text(song.get("music_credit") or MUSIC_CREDIT, encoding="utf-8")
        else:
            credit = ace_music(song, folder)
            if not credit:                              # quality first: no weak fallback, retry later
                raise SystemExit("ACE-Step did not deliver a song this time; nothing published, next run retries")
            credit_file.write_text(credit, encoding="utf-8")
    prompts = scene_prompts(song)
    for n, pr in enumerate(prompts, 1):                 # the story, one image per scene
        if not (folder / f"scene_{n:02d}.png").exists():
            make_image(pr, folder / f"scene_{n:02d}.png", seed=10 + n)
            print(f"  scene {n}/{len(prompts)} ready")
    plan = subprocess.run([sys.executable, str(ROOT / "clip.py"), "--audio", str(folder / "song.mp3"),
                           "--scenes", str(folder), "--title", song["title"], "--out", str(ROOT / "output" / "plan"),
                           "--plan"], capture_output=True, text=True, check=True).stdout
    print(" ", plan.strip())
    used = [int(x) for x in re.findall(r"\d+", plan.split("uses scenes")[-1])]
    for n in used:                                      # vertical versions only for the Short's scenes
        if not (folder / f"v_{n:02d}.png").exists():
            make_image(prompts[n - 1], folder / f"v_{n:02d}.png", width=768, height=1344, seed=10 + n)
    for n in range(1, len(prompts) + 1):
        if not (folder / f"v_{n:02d}.png").exists():
            shutil.copy(folder / f"scene_{n:02d}.png", folder / f"v_{n:02d}.png")
    out = ROOT / "output" / f"gt_{day}_{slugify(song['title'])}"
    cmd = [sys.executable, str(ROOT / "clip.py"), "--audio", str(folder / "song.mp3"), "--scenes", str(folder),
           "--title", song["title"], "--subtitle", song["genre"], "--out", str(out),
           "--hook", song.get("hook") or song["title"]]
    if (folder / "song.lrc").exists():
        cmd += ["--lrc", str(folder / "song.lrc")]
    subprocess.run(cmd, check=True)
    music_credit = credit_file.read_text(encoding="utf-8")
    pending_dir, folder = folder, out
    digest = sha256(folder / "audio.wav")
    for d in pub.values():
        if d.get("audio_sha256") == digest:
            raise SystemExit(f"SKIPPED DUPLICATE: same audio already published as {d.get('url')}")

    status = {"privacyStatus": "public", "selfDeclaredMadeForKids": False, "containsSyntheticMedia": True}
    if slot > now + datetime.timedelta(minutes=15):
        status.update(privacyStatus="private",
                      publishAt=slot.astimezone(datetime.timezone.utc).isoformat().replace("+00:00", "Z"))
    extra = {k: song[k] for k in ("blurb", "hashtags") if song.get(k)}
    music = music_credit
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
    try:                                                # custom thumbnail (needs a verified channel)
        yt.thumbnails().set(videoId=vid, media_body=MediaFileUpload(str(folder / "thumbnail.jpg"))).execute()
    except Exception as exc:
        print(f"  thumbnail not set: {str(exc)[:120]}")
    shutil.rmtree(pending_dir, ignore_errors=True)
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
