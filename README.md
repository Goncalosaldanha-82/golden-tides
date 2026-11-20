# Golden Tides — daily release

Every day GitHub Actions makes one new song for the Golden Tides YouTube channel and schedules it for 18:00 (Lisbon time):
original lyrics from `songs_bank.json` → music with YuE (open model, Hugging Face free GPU quota) → cover with FLUX.1-schnell →
cinematic video + Short (`cinematic.py`) → upload (`gt_daily.py`). Published releases are recorded in `gt_published.json`.
Secrets (`HF_TOKEN`, `GT_TOKEN_JSON`) are stored as encrypted repository secrets, never in the code.
