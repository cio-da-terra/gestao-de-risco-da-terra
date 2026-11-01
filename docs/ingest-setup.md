# Turning on scheduled ingestion

`.github/workflows/ingest.yml` is in the repo but does nothing on its own:
scheduled runs are skipped until the repo variable `INGEST_ENABLED` is `true`.

## How it works

Each run starts on a fresh GitHub VM, so the database lives in Google Drive
through DVC. Git only holds `data.dvc`, a small pointer to the current version.

    dvc pull  ->  python -m ingest <cadence>  ->  dvc add data + dvc push  ->  commit data.dvc

| Trigger | When (UTC) | Runs |
|---|---|---|
| Daily cron | 23:17, Mon–Fri | `daily` (BCB, NASA POWER) |
| Monthly cron | 12:37 on day 5 | `monthly` (+ ONI, Pink Sheet, IPCA, CEPEA files) |
| Manual ("Run workflow") | any time | the cadence you pick; `yearly` adds PAM |

## Setup

1. **Drive folder.** Create a folder in a **Shared Drive** and copy its id
   (the last part of the folder URL). Service accounts have no storage quota
   of their own, so uploads to a folder in someone's personal "My Drive"
   fail. Shared Drives need a Google Workspace account (most university
   accounts are); a plain @gmail.com account can't create one.
2. **Service account.** In Google Cloud Console: create a project, enable the
   Google Drive API, create a service account, and download a JSON key. Add
   the service account's email as a *Content manager* of the Shared Drive.
3. **Point DVC at the folder.**
   `dvc remote modify gdrive url gdrive://<FOLDER_ID>` and commit `.dvc/config`.
   (The id isn't secret; the key is.)
4. **Secret.** Repo Settings -> Secrets and variables -> Actions -> *Secrets*:
   `GDRIVE_CREDENTIALS_DATA` = the whole JSON key file content.
   Never commit the key: the repo is public.
5. **Test by hand.** Actions -> ingest -> Run workflow, cadence `daily`.
   Then `yearly` once, to backfill PAM.
6. **Switch on.** Same settings page, *Variables* tab: `INGEST_ENABLED` = `true`.

Before switching on, set the real municipalities in `ingest/__main__.py`
(`LOCATIONS`).

## Things to know

- Cron times are UTC and GitHub may start runs late or, rarely, skip one.
  BCB re-fetches the last 30 days, so a missed day fills itself in.
- Public repos pause scheduled workflows after 60 days without commits. The
  bot's commits keep it alive while it runs.
- If any source fails, the run shows red (GitHub emails the person who
  last edited the workflow) but data from the other sources is still saved.
  Details are in the log and in the `fetch_log` table.
- If a Brazilian site blocks GitHub's US runners, run that source locally
  and `dvc push`, or move it to a self-hosted runner.
- Turn it off: delete `INGEST_ENABLED` or set it to anything other than `true`.
