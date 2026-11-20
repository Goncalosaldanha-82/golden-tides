"""Upload generated tracks (full video + Short) to the MUSIC channel via the free YouTube Data API.

Reuses the Google Cloud OAuth client of the Ball League project (../ball-league/client_secret.json)
but keeps its OWN token (token_music.json) and its OWN channel id (config.json), so the two
channels never mix.

    python upload_music.py --dry-run      # show what would be uploaded, upload nothing
    python upload_music.py --unattended   # scheduled runs: never opens a browser
    python upload_music.py --auth         # one-time sign-in (choose the music channel)

Never uploads a duplicate: same audio file hash, same composition seed, or a track number
already on the channel => skipped.
"""
import argparse
import datetime
import hashlib
import json
import re
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
CONFIG = ROOT / "config.json"
TOKEN = ROOT / "token_music.json"
CLIENT = ROOT.parent / "ball-league" / "client_secret.json"
SCOPES = ["https://www.googleapis.com/auth/youtube.upload", "https://www.googleapis.com/auth/youtube.readonly"]
REAUTH = "MUSIC CHANNEL AUTH MISSING/EXPIRED: run 'python upload_music.py --auth' and pick the music channel."


def config():
    return json.loads(CONFIG.read_text(encoding="utf-8")) if CONFIG.exists() else {}


def service(interactive):
    creds = Credentials.from_authorized_user_file(str(TOKEN)) if TOKEN.exists() else None
    if not creds or not creds.valid:
        ok = False
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                ok = True
            except Exception as exc:
                print(f"refresh failed: {exc}")
        if not ok:
            if not interactive:
                raise SystemExit(REAUTH)
            creds = InstalledAppFlow.from_client_secrets_file(str(CLIENT), SCOPES).run_local_server(port=0)
    yt = build("youtube", "v3", credentials=creds)
    me = yt.channels().list(part="id,snippet", mine=True).execute()["items"][0]
    cfg = config()
    if cfg.get("channel_id") and me["id"] != cfg["channel_id"]:
        TOKEN.unlink(missing_ok=True)                      # never keep a token for the wrong channel
        raise SystemExit(f"Signed in to {me['snippet']['title']} ({me['id']}), expected {cfg['channel_id']}. "
                         "Delete token_music.json and run --auth again choosing the music channel.")
    TOKEN.write_text(creds.to_json(), encoding="utf-8")   # saved only after the channel check passed
    return yt, me


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def metadata(info, cfg):
    g = info["genre"]
    mood = {"GYM PHONK": "Aggressive Gym Phonk", "BRAZILIAN PHONK": "Brazilian Gym Phonk",
            "HARD PHONK": "Hard Phonk", "EDM": "High Energy EDM", "EPIC TRAP": "Epic Motivational Trap",
            "DRUM & BASS": "Fast Drum & Bass"}[g]
    emoji = {"GYM PHONK": "🔥", "BRAZILIAN PHONK": "🇧🇷", "HARD PHONK": "💀", "EDM": "⚡", "EPIC TRAP": "👑",
             "DRUM & BASS": "🏃"}[g]
    title = f"{info['title']} {emoji} {mood} | Gym Phonk Workout Music"
    desc = "\n".join([
        f"{info['title']} — {mood} ({info['bpm']} BPM). Original track #{info['track']}.",
        "A brand-new original track every day to push your workout, run or focus session. 💪",
        "",
        "Turn it up and hit subscribe for tomorrow's track.",
        "",
        "100% original: composed and produced from scratch (no samples, no copied melodies).",
        "#workoutmusic #gymmotivation #" + g.lower().replace(" & ", "").replace(" ", ""),
    ])
    tags = ["gym phonk", "phonk", "workout music", "gym music", "gym motivation", "phonk music",
            "original music"] + {
        "GYM PHONK": ["aggressive phonk", "drift phonk"], "BRAZILIAN PHONK": ["brazilian phonk", "funk phonk"],
        "HARD PHONK": ["hard phonk", "aggressive phonk"], "EDM": ["edm", "big room", "festival"],
        "EPIC TRAP": ["epic trap", "hype music", "trap"], "DRUM & BASS": ["drum and bass", "dnb", "running music"]}[g]
    return {"title": title[:100], "description": desc, "tags": tags, "categoryId": "10"}


def short_metadata(info, long_url):
    g = info["genre"].lower().replace(" & ", "").replace(" ", "")
    return {"title": f"{info['title']} 🔥 wait for the drop #shorts"[:100],
            "description": f"Full track: {long_url}\n#{g} #gymmotivation #workoutmusic #shorts",
            "tags": ["shorts", "workout music", "gym motivation", info["genre"].lower()], "categoryId": "10"}


def channel_track_numbers(yt, channel_id):
    ch = yt.channels().list(part="contentDetails", id=channel_id).execute()["items"][0]
    pl = ch["contentDetails"]["relatedPlaylists"]["uploads"]
    found, tok = {}, None
    while True:
        r = yt.playlistItems().list(part="snippet", playlistId=pl, maxResults=50, pageToken=tok).execute()
        for it in r.get("items", []):
            m = re.search(r"Original track #(\d+)\b", it["snippet"].get("description", ""))
            if m:
                found[int(m.group(1))] = it["snippet"]["resourceId"]["videoId"]
        tok = r.get("nextPageToken")
        if not tok:
            return found


def insert(yt, path, meta):
    body = {"snippet": {k: meta[k] for k in ("title", "description", "tags", "categoryId")},
            "status": {"privacyStatus": "public", "selfDeclaredMadeForKids": False, "containsSyntheticMedia": False}}
    res = yt.videos().insert(part="snippet,status", body=body,
                             media_body=MediaFileUpload(str(path), mimetype="video/mp4", resumable=True)).execute()
    return res["id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auth", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--unattended", action="store_true")
    a = ap.parse_args()
    yt, me = service(interactive=not a.unattended)
    cfg = config()
    if a.auth or not cfg.get("channel_id"):
        print(f"signed in as channel: {me['snippet']['title']} ({me['id']})")
        if not cfg.get("channel_id"):
            print("config.json has no channel_id yet: nothing uploaded. Set it to the music channel id.")
        return                                            # --auth only authorizes; it never uploads
    done = {}
    for up in OUT.glob("track_*/uploaded.json"):
        d = json.loads(up.read_text(encoding="utf-8"))
        done[up.parent.name] = d
    on_channel = channel_track_numbers(yt, cfg["channel_id"])
    queue = [f for f in sorted(OUT.glob("track_*")) if (f / "video.mp4").exists() and (f / "short.mp4").exists()
             and (f / "song.json").exists() and not (f / "uploaded.json").exists()]
    print(f"published: {sorted(done) or 'none'} | on channel: tracks {sorted(on_channel) or 'none'} | waiting: {[f.name for f in queue] or 'none'}")
    for folder in queue[:1]:                                  # at most ONE new track per run
        info = json.loads((folder / "song.json").read_text(encoding="utf-8"))
        digest = sha256(folder / "audio.wav")
        why = None
        for name, d in done.items():
            if d.get("audio_sha256") == digest:
                why = f"same audio as {name}"
            elif d.get("seed") == info["seed"]:
                why = f"same composition seed as {name}"
        if info["track"] in on_channel:
            why = f"track #{info['track']} already on the channel (youtu.be/{on_channel[info['track']]})"
        if why:
            print(f"SKIPPED DUPLICATE {folder.name}: {why}. Not uploaded.")
            continue
        meta = metadata(info, cfg)
        print(f"{'WOULD UPLOAD' if a.dry_run else 'uploading'} {folder.name}: \"{meta['title']}\" + Short (unique audio, seed and number)")
        if a.dry_run:
            continue
        vid = insert(yt, folder / "video.mp4", meta)
        url = f"https://youtu.be/{vid}"
        sid = insert(yt, folder / "short.mp4", short_metadata(info, url))
        rec = {"video_id": vid, "short_id": sid, "url": url, "short_url": f"https://youtube.com/shorts/{sid}",
               "audio_sha256": digest, "seed": info["seed"], "uploaded": datetime.datetime.now().isoformat()}
        (folder / "uploaded.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(f"{folder.name} -> {url} | short {rec['short_url']}")


if __name__ == "__main__":
    main()
