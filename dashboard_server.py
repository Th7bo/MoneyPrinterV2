"""
MoneyPrinterV2 Dashboard — single-file server.

A web control panel over the upstream MoneyPrinterV2 CLI. Wraps the existing
src/classes (YouTube, Twitter, AffiliateMarketing, Outreach) in an HTTP API +
single-page UI, runs them as background jobs with live logs, and persists
schedules. Designed to be dropped into a fork as ONE file and started with:

    uvicorn dashboard_server:app --host 0.0.0.0 --port 8080

No upstream code is modified.
"""
import os
import io
import re
import sys
import json
import time
import queue
import shutil
import threading
import traceback
from uuid import uuid4
from typing import Optional
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Bootstrap: make the upstream ./src importable and prepare runtime files.
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(REPO_ROOT, "src")
MP_DIR = os.path.join(REPO_ROOT, ".mp")
CONFIG_PATH = os.path.join(REPO_ROOT, "config.json")
CONFIG_EXAMPLE = os.path.join(REPO_ROOT, "config.example.json")
MP_CONFIG = os.path.join(MP_DIR, "config.json")

# Upstream config.py computes ROOT_DIR = os.path.dirname(sys.path[0]),
# so src/ must be sys.path[0] for it to resolve to the repo root.
if SRC_DIR in sys.path:
    sys.path.remove(SRC_DIR)
sys.path.insert(0, SRC_DIR)
os.chdir(REPO_ROOT)


def _prepare_runtime():
    os.makedirs(MP_DIR, exist_ok=True)
    try:
        os.makedirs("/profiles", exist_ok=True)
    except OSError:
        pass

    # Persist config inside .mp and expose it at ./config.json via a symlink,
    # so a single named volume (.mp) survives redeploys.
    if not os.path.exists(MP_CONFIG):
        if os.path.isfile(CONFIG_PATH) and not os.path.islink(CONFIG_PATH):
            shutil.move(CONFIG_PATH, MP_CONFIG)
        elif os.path.exists(CONFIG_EXAMPLE):
            shutil.copyfile(CONFIG_EXAMPLE, MP_CONFIG)
        else:
            json.dump({"verbose": True}, open(MP_CONFIG, "w"), indent=2)
    try:
        if os.path.islink(CONFIG_PATH) or os.path.exists(CONFIG_PATH):
            os.remove(CONFIG_PATH)
        os.symlink(MP_CONFIG, CONFIG_PATH)
    except OSError:
        # Fall back to a copy if the filesystem disallows symlinks.
        if not os.path.exists(CONFIG_PATH):
            shutil.copyfile(MP_CONFIG, CONFIG_PATH)

    # Container-correct defaults, without clobbering user edits.
    try:
        cfg = json.load(open(CONFIG_PATH))
    except Exception:
        cfg = {}
    magick = shutil.which("convert") or "/usr/bin/convert"
    if not cfg.get("imagemagick_path") or "Path to" in str(cfg.get("imagemagick_path", "")):
        cfg["imagemagick_path"] = magick
    cfg.setdefault("headless", False)
    url = os.environ.get("OLLAMA_BASE_URL", "").strip()
    if url and str(cfg.get("ollama_base_url", "")).startswith("http://127.0.0.1"):
        cfg["ollama_base_url"] = url
    json.dump(cfg, open(CONFIG_PATH, "w"), indent=2)


_prepare_runtime()

# ---------------------------------------------------------------------------
# Per-thread stdout/stderr routing so each job captures its own output.
# ---------------------------------------------------------------------------
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_thread_local = threading.local()
_real_stdout, _real_stderr = sys.stdout, sys.stderr


class _Router(io.TextIOBase):
    def __init__(self, real):
        self._real = real

    def write(self, s):
        buf = getattr(_thread_local, "buffer", None)
        if buf is not None:
            for line in _ANSI.sub("", s).splitlines(keepends=True):
                buf.append(line)
        return self._real.write(s)

    def flush(self):
        try:
            self._real.flush()
        except Exception:
            pass


sys.stdout = _Router(_real_stdout)
sys.stderr = _Router(_real_stderr)


def _now():
    return datetime.now(timezone.utc).isoformat()


class Job:
    def __init__(self, kind, label, fn):
        self.id = uuid4().hex[:12]
        self.kind, self.label, self.fn = kind, label, fn
        self.status = "queued"
        self.created_at = _now()
        self.started_at = self.ended_at = self.error = None
        self.lines = []

    def to_dict(self, include_logs=False):
        d = {"id": self.id, "kind": self.kind, "label": self.label,
             "status": self.status, "created_at": self.created_at,
             "started_at": self.started_at, "ended_at": self.ended_at,
             "error": self.error, "line_count": len(self.lines)}
        if include_logs:
            d["log"] = "".join(self.lines)
        return d


_JOBS, _ORDER, _LOCK = {}, [], threading.Lock()
_Q: "queue.Queue[Job]" = queue.Queue()


def _worker():
    while True:
        job = _Q.get()
        job.status, job.started_at = "running", _now()
        _thread_local.buffer = job.lines
        job.lines.append(f"[{datetime.now():%H:%M:%S}] \u25b6 {job.label}\n")
        try:
            job.fn()
            job.status = "done"
            job.lines.append(f"[{datetime.now():%H:%M:%S}] \u2713 finished\n")
        except Exception as e:
            job.status, job.error = "error", str(e)
            job.lines.append("".join(traceback.format_exc()))
            job.lines.append(f"[{datetime.now():%H:%M:%S}] \u2717 failed: {e}\n")
        finally:
            job.ended_at = _now()
            _thread_local.buffer = None
            _Q.task_done()


threading.Thread(target=_worker, daemon=True, name="job-worker").start()


def enqueue(kind, label, fn):
    job = Job(kind, label, fn)
    with _LOCK:
        _JOBS[job.id] = job
        _ORDER.append(job.id)
        while len(_ORDER) > 200:
            _JOBS.pop(_ORDER.pop(0), None)
    _Q.put(job)
    return job


def list_jobs(limit=50):
    with _LOCK:
        return [_JOBS[i].to_dict() for i in _ORDER[-limit:][::-1] if i in _JOBS]


# --- helpers into upstream (imported lazily so heavy deps never break the app) --
def _select_model_or_raise():
    from config import get_ollama_model
    from llm_provider import select_model
    model = (get_ollama_model() or "").strip()
    if not model:
        raise RuntimeError("No Ollama model set. Open Settings and set 'ollama_model' "
                           "(e.g. llama3.2:3b) with Ollama reachable at 'ollama_base_url'.")
    select_model(model)
    return model


def _find_account(provider, account_id):
    from cache import get_accounts
    for acc in get_accounts(provider):
        if acc.get("id") == account_id:
            return acc
    raise RuntimeError(f"No {provider} account with id {account_id!r} found.")


def action_youtube_generate(account_id, upload):
    acc = _find_account("youtube", account_id)

    def run():
        model = _select_model_or_raise(); print(f"Using model: {model}")
        from config import assert_folder_structure
        from utils import fetch_songs
        from classes.Tts import TTS
        from classes.YouTube import YouTube
        assert_folder_structure()
        try:
            fetch_songs()
        except Exception as e:
            print(f"(warning) could not fetch songs: {e}")
        yt = YouTube(acc["id"], acc["nickname"], acc["firefox_profile"], acc["niche"], acc["language"])
        yt.generate_video(TTS())
        if upload:
            print("Uploading to YouTube...")
            print("Upload succeeded." if yt.upload_video() else "Upload failed.")

    return enqueue("youtube", f"YouTube \u00b7 {acc['nickname']} \u00b7 {'generate + upload' if upload else 'generate'}", run)


def action_twitter_post(account_id, text):
    acc = _find_account("twitter", account_id)

    def run():
        _select_model_or_raise()
        from classes.Twitter import Twitter
        Twitter(acc["id"], acc["nickname"], acc["firefox_profile"], acc["topic"]).post(text or None)

    return enqueue("twitter", f"Twitter \u00b7 {acc['nickname']} \u00b7 post", run)


def action_afm_share(product_id):
    from cache import get_products
    product = next((p for p in get_products() if p.get("id") == product_id), None)
    if not product:
        raise RuntimeError(f"No product with id {product_id!r}.")
    acc = _find_account("twitter", product["twitter_uuid"])

    def run():
        _select_model_or_raise()
        from classes.AFM import AffiliateMarketing
        afm = AffiliateMarketing(product["affiliate_link"], acc["firefox_profile"],
                                 acc["id"], acc["nickname"], acc["topic"])
        afm.generate_pitch(); afm.share_pitch("twitter")

    return enqueue("afm", f"Affiliate \u00b7 {acc['nickname']} \u00b7 pitch + share", run)


def action_outreach_run():
    def run():
        from classes.Outreach import Outreach
        Outreach().start()
    return enqueue("outreach", "Outreach \u00b7 scrape + email", run)


def run_scheduled(kind, account_id="", upload=True):
    """Referenced by APScheduler as 'dashboard_server:run_scheduled'."""
    if kind == "youtube":
        action_youtube_generate(account_id, upload)
    elif kind == "twitter":
        action_twitter_post(account_id, None)
    elif kind == "afm":
        action_afm_share(account_id)
    elif kind == "outreach":
        action_outreach_run()
    else:
        raise ValueError(f"Unknown scheduled kind: {kind}")

# ---------------------------------------------------------------------------
# Embedded single-page UI
# ---------------------------------------------------------------------------
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>MoneyPrinter · Console</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root{
    --ink:#0e1218; --panel:#151b24; --raised:#1b222d; --line:#28303c;
    --text:#e2e7ee; --muted:#8b95a6; --dim:#5c6675;
    --mint:#4ea88b; --mint-soft:#2b4a44; --amber:#d9a441; --red:#d96a5a;
    --gold:#b0894f;
    --radius:10px;
    --mono:'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
    --sans:'Space Grotesk',Inter,system-ui,-apple-system,sans-serif;
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%}
  body{
    background:var(--ink);color:var(--text);font-family:var(--sans);
    font-size:15px;line-height:1.5;
    background-image:radial-gradient(circle at 100% 0,rgba(78,168,139,.06),transparent 40%);
  }
  a{color:inherit}
  button{font-family:inherit;cursor:pointer}
  input,select,textarea{font-family:var(--sans);font-size:14px}

  /* ---- shell ---- */
  .app{display:grid;grid-template-columns:230px 1fr;min-height:100vh}
  .side{border-right:1px solid var(--line);padding:22px 16px;position:sticky;top:0;height:100vh;display:flex;flex-direction:column;gap:6px}
  .brand{display:flex;align-items:center;gap:10px;padding:0 8px 18px;margin-bottom:8px;border-bottom:1px solid var(--line)}
  .brand .mark{width:30px;height:30px;border-radius:7px;background:linear-gradient(150deg,var(--mint),#2f6f5c);display:grid;place-items:center;font-weight:700;color:#08120e;font-size:16px;box-shadow:0 0 0 1px rgba(255,255,255,.06) inset}
  .brand b{font-weight:600;letter-spacing:.3px}
  .brand span{display:block;font-size:11px;color:var(--dim);font-family:var(--mono);letter-spacing:1px;text-transform:uppercase}
  .nav{display:flex;flex-direction:column;gap:2px;margin-top:6px}
  .nav button{background:none;border:0;color:var(--muted);text-align:left;padding:9px 12px;border-radius:8px;font-size:14px;display:flex;justify-content:space-between;align-items:center;transition:.15s}
  .nav button:hover{background:var(--panel);color:var(--text)}
  .nav button.active{background:var(--raised);color:var(--text)}
  .nav button.active::before{content:"";position:absolute;left:8px;width:3px;height:16px;background:var(--mint);border-radius:2px}
  .nav button{position:relative}
  .nav .count{font-family:var(--mono);font-size:11px;color:var(--dim)}
  .side .foot{margin-top:auto;font-family:var(--mono);font-size:11px;color:var(--dim);padding:8px}

  /* ---- ticker (signature) ---- */
  .ticker{border-bottom:1px solid var(--line);background:linear-gradient(180deg,var(--panel),transparent);overflow:hidden;position:relative}
  .ticker::after{content:"";position:absolute;left:0;right:0;bottom:0;height:1px;background:repeating-linear-gradient(90deg,var(--gold) 0 6px,transparent 6px 12px);opacity:.35}
  .ticker .rail{display:flex;gap:44px;white-space:nowrap;font-family:var(--mono);font-size:12px;color:var(--muted);padding:9px 0;animation:scroll 38s linear infinite;width:max-content}
  .ticker .rail b{color:var(--text);font-weight:600}
  .ticker .rail .g{color:var(--mint)}
  .ticker:hover .rail{animation-play-state:paused}
  @keyframes scroll{from{transform:translateX(0)}to{transform:translateX(-50%)}}

  main{padding:26px 30px 120px;max-width:1080px}
  .head{display:flex;align-items:baseline;justify-content:space-between;margin-bottom:20px;gap:16px;flex-wrap:wrap}
  .head h1{font-size:22px;font-weight:600;margin:0;letter-spacing:.2px}
  .head p{margin:4px 0 0;color:var(--muted);font-size:13.5px}
  .eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:2px;text-transform:uppercase;color:var(--gold);margin:0 0 4px}

  .grid{display:grid;gap:14px}
  .cards{grid-template-columns:repeat(auto-fit,minmax(150px,1fr))}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:16px}
  .stat{font-family:var(--mono);font-size:30px;font-weight:600;letter-spacing:-1px}
  .stat .lbl{display:block;font-family:var(--sans);font-size:12px;color:var(--muted);letter-spacing:.4px;margin-top:2px;font-weight:400}

  .panel{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);overflow:hidden}
  .panel h3{margin:0;padding:14px 16px;font-size:13px;font-weight:600;letter-spacing:.4px;border-bottom:1px solid var(--line);color:var(--text);display:flex;justify-content:space-between;align-items:center}
  .panel h3 .hint{font-family:var(--mono);font-size:11px;color:var(--dim);font-weight:400}
  .panel .body{padding:16px}
  .stack{display:flex;flex-direction:column;gap:14px}

  .row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
  .item{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:12px 14px;border:1px solid var(--line);border-radius:9px;background:var(--raised)}
  .item + .item{margin-top:8px}
  .item .meta b{font-weight:600}
  .item .meta small{display:block;font-family:var(--mono);font-size:11px;color:var(--dim);margin-top:2px;word-break:break-all}

  label{display:block;font-size:12px;color:var(--muted);margin:0 0 5px;letter-spacing:.3px}
  input[type=text],input[type=number],input[type=password],select,textarea{
    width:100%;background:var(--ink);border:1px solid var(--line);color:var(--text);
    border-radius:8px;padding:9px 11px;outline:none;transition:.15s
  }
  input:focus,select:focus,textarea:focus{border-color:var(--mint);box-shadow:0 0 0 3px rgba(78,168,139,.12)}
  textarea{font-family:var(--mono);font-size:12.5px;resize:vertical;min-height:74px}
  .field{margin-bottom:12px}
  .fields{display:grid;grid-template-columns:1fr 1fr;gap:12px}

  .btn{background:var(--raised);border:1px solid var(--line);color:var(--text);padding:8px 14px;border-radius:8px;font-size:13.5px;transition:.15s;display:inline-flex;align-items:center;gap:7px}
  .btn:hover{border-color:var(--dim);background:#232b36}
  .btn.go{background:var(--mint);color:#08130f;border-color:transparent;font-weight:600}
  .btn.go:hover{background:#5cbb9a}
  .btn.ghost{background:none}
  .btn.danger{color:var(--red);border-color:transparent;background:none;padding:6px 8px}
  .btn.danger:hover{background:rgba(217,106,90,.12)}
  .btn:disabled{opacity:.5;cursor:not-allowed}

  .pill{font-family:var(--mono);font-size:11px;padding:3px 8px;border-radius:20px;border:1px solid var(--line);color:var(--muted);text-transform:uppercase;letter-spacing:.5px}
  .pill.running{color:var(--mint);border-color:var(--mint-soft);background:rgba(78,168,139,.08)}
  .pill.queued{color:var(--amber);border-color:rgba(217,164,65,.3)}
  .pill.done{color:var(--dim)}
  .pill.error{color:var(--red);border-color:rgba(217,106,90,.3)}
  .dot{width:7px;height:7px;border-radius:50%;display:inline-block;background:var(--dim)}
  .dot.running{background:var(--mint);box-shadow:0 0 0 0 rgba(78,168,139,.6);animation:pulse 1.6s infinite}
  .dot.queued{background:var(--amber)} .dot.error{background:var(--red)} .dot.done{background:var(--dim)}
  @keyframes pulse{0%{box-shadow:0 0 0 0 rgba(78,168,139,.5)}70%{box-shadow:0 0 0 7px rgba(78,168,139,0)}100%{box-shadow:0 0 0 0 rgba(78,168,139,0)}}

  .empty{color:var(--dim);font-size:13.5px;padding:18px;text-align:center;border:1px dashed var(--line);border-radius:9px}
  .note{font-size:12.5px;color:var(--muted);background:rgba(217,164,65,.07);border:1px solid rgba(217,164,65,.2);border-radius:8px;padding:11px 13px;margin-top:12px}
  .note b{color:var(--amber)}

  .view{display:none;animation:fade .25s ease}
  .view.active{display:block}
  @keyframes fade{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}

  /* ---- console drawer ---- */
  .console{position:fixed;left:0;right:0;bottom:0;height:0;background:#0a0d12;border-top:1px solid var(--line);transition:height .28s cubic-bezier(.4,0,.2,1);z-index:40;display:flex;flex-direction:column;box-shadow:0 -20px 40px rgba(0,0,0,.4)}
  .console.open{height:46vh}
  .console .bar{display:flex;align-items:center;gap:12px;padding:9px 16px;border-bottom:1px solid var(--line);font-family:var(--mono);font-size:12px}
  .console .bar .title{color:var(--text)}
  .console .bar .x{margin-left:auto}
  .console pre{margin:0;flex:1;overflow:auto;padding:14px 16px;font-family:var(--mono);font-size:12.5px;line-height:1.55;color:#c9d2de;white-space:pre-wrap;word-break:break-word}
  .toast{position:fixed;bottom:16px;right:16px;background:var(--raised);border:1px solid var(--line);border-left:3px solid var(--mint);padding:12px 16px;border-radius:8px;font-size:13.5px;z-index:60;max-width:320px;animation:fade .2s}
  .toast.err{border-left-color:var(--red)}

  @media(max-width:760px){
    .app{grid-template-columns:1fr}
    .side{position:static;height:auto;flex-direction:row;flex-wrap:wrap;align-items:center}
    .side .foot,.brand{display:none}
    .nav{flex-direction:row;flex-wrap:wrap}
    .nav button.active::before{display:none}
    main{padding:18px 16px 120px}
    .fields{grid-template-columns:1fr}
  }
  @media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>
</head>
<body>
<div class="app">
  <aside class="side">
    <div class="brand">
      <div class="mark">$</div>
      <div><b>MoneyPrinter</b><span>console</span></div>
    </div>
    <nav class="nav" id="nav">
      <button data-view="overview" class="active">Overview</button>
      <button data-view="youtube">YouTube <span class="count" id="c-yt">·</span></button>
      <button data-view="twitter">Twitter <span class="count" id="c-tw">·</span></button>
      <button data-view="affiliate">Affiliate <span class="count" id="c-af">·</span></button>
      <button data-view="outreach">Outreach</button>
      <button data-view="schedule">Schedule <span class="count" id="c-sc">·</span></button>
      <button data-view="settings">Settings</button>
      <button data-view="logs">Job log</button>
    </nav>
    <div class="foot" id="foot">idle</div>
  </aside>

  <div>
    <div class="ticker"><div class="rail" id="ticker">loading…</div></div>
    <main>
      <!-- OVERVIEW -->
      <section class="view active" id="v-overview">
        <div class="head"><div><p class="eyebrow">Mission control</p><h1>Overview</h1><p>Everything the printer is running, at a glance.</p></div></div>
        <div class="grid cards" id="stat-cards"></div>
        <div class="panel" style="margin-top:16px">
          <h3>Recent runs <span class="hint">newest first</span></h3>
          <div class="body" id="recent"></div>
        </div>
      </section>

      <!-- YOUTUBE -->
      <section class="view" id="v-youtube">
        <div class="head"><div><p class="eyebrow">Shorts automation</p><h1>YouTube</h1><p>Generate and upload Shorts per account. Uploads drive a logged-in Firefox profile.</p></div></div>
        <div class="stack">
          <div class="panel"><h3>Accounts</h3><div class="body" id="yt-list"></div></div>
          <div class="panel"><h3>Add account</h3><div class="body">
            <div class="fields">
              <div class="field"><label>Nickname</label><input id="yt-nick" type="text" placeholder="finance-shorts"></div>
              <div class="field"><label>Niche</label><input id="yt-niche" type="text" placeholder="personal finance"></div>
              <div class="field"><label>Firefox profile path</label><input id="yt-fp" type="text" placeholder="/profiles/youtube"></div>
              <div class="field"><label>Language</label><input id="yt-lang" type="text" value="English"></div>
            </div>
            <button class="btn go" onclick="addYouTube()">Add account</button>
          </div></div>
        </div>
      </section>

      <!-- TWITTER -->
      <section class="view" id="v-twitter">
        <div class="head"><div><p class="eyebrow">Posting bot</p><h1>Twitter / X</h1><p>Post generated content, or a specific message, per account.</p></div></div>
        <div class="stack">
          <div class="panel"><h3>Accounts</h3><div class="body" id="tw-list"></div></div>
          <div class="panel"><h3>Add account</h3><div class="body">
            <div class="fields">
              <div class="field"><label>Nickname</label><input id="tw-nick" type="text" placeholder="build-in-public"></div>
              <div class="field"><label>Topic</label><input id="tw-topic" type="text" placeholder="indie hacking"></div>
              <div class="field" style="grid-column:1/-1"><label>Firefox profile path</label><input id="tw-fp" type="text" placeholder="/profiles/twitter"></div>
            </div>
            <button class="btn go" onclick="addTwitter()">Add account</button>
          </div></div>
        </div>
      </section>

      <!-- AFFILIATE -->
      <section class="view" id="v-affiliate">
        <div class="head"><div><p class="eyebrow">Affiliate marketing</p><h1>Affiliate</h1><p>Turn an affiliate link into a pitch and share it from a Twitter account.</p></div></div>
        <div class="stack">
          <div class="panel"><h3>Products</h3><div class="body" id="af-list"></div></div>
          <div class="panel"><h3>Add product</h3><div class="body">
            <div class="field"><label>Affiliate link</label><input id="af-link" type="text" placeholder="https://amzn.to/..."></div>
            <div class="field"><label>Share from Twitter account</label><select id="af-acc"></select></div>
            <button class="btn go" onclick="addProduct()">Add product</button>
          </div></div>
        </div>
      </section>

      <!-- OUTREACH -->
      <section class="view" id="v-outreach">
        <div class="head"><div><p class="eyebrow">Cold outreach</p><h1>Outreach</h1><p>Scrape local businesses for a niche and email them via your SMTP account.</p></div></div>
        <div class="panel"><h3>Run outreach</h3><div class="body">
          <p style="margin-top:0;color:var(--muted);font-size:13.5px">Uses the niche, SMTP credentials and message template from <b style="color:var(--text)">Settings</b>. Requires Go installed in the container (the image includes it).</p>
          <button class="btn go" onclick="runOutreach()">Start scrape &amp; email</button>
          <div class="note"><b>Heads up</b> — sending unsolicited bulk email can violate anti-spam laws (CAN-SPAM, GDPR) and your SMTP provider's terms. Make sure you have a lawful basis and an unsubscribe path before running this at volume.</div>
        </div></div>
      </section>

      <!-- SCHEDULE -->
      <section class="view" id="v-schedule">
        <div class="head"><div><p class="eyebrow">Cron</p><h1>Schedule</h1><p>Recurring runs, persisted across restarts. Times are UTC.</p></div></div>
        <div class="stack">
          <div class="panel"><h3>Active schedules</h3><div class="body" id="sc-list"></div></div>
          <div class="panel"><h3>New schedule</h3><div class="body">
            <div class="fields">
              <div class="field"><label>What to run</label><select id="sc-kind" onchange="scKindChange()">
                <option value="youtube">YouTube — generate + upload</option>
                <option value="twitter">Twitter — post</option>
                <option value="afm">Affiliate — pitch + share</option>
                <option value="outreach">Outreach — scrape + email</option>
              </select></div>
              <div class="field" id="sc-target-wrap"><label>Target</label><select id="sc-target"></select></div>
            </div>
            <div class="field"><label>Cron (min hour day month weekday, UTC)</label><input id="sc-cron" type="text" value="0 10 * * *"></div>
            <div class="row" style="margin-bottom:12px">
              <button class="btn ghost" onclick="setCron('0 10 * * *')">Once/day</button>
              <button class="btn ghost" onclick="setCron('0 10,16 * * *')">Twice/day</button>
              <button class="btn ghost" onclick="setCron('0 8,12,18 * * *')">3×/day</button>
            </div>
            <button class="btn go" onclick="addSchedule()">Create schedule</button>
          </div></div>
        </div>
      </section>

      <!-- SETTINGS -->
      <section class="view" id="v-settings">
        <div class="head"><div><p class="eyebrow">config.json</p><h1>Settings</h1><p>Edit the live config. Objects and arrays are edited as JSON.</p></div>
          <button class="btn go" onclick="saveConfig()">Save config</button></div>
        <div class="panel"><h3>Configuration</h3><div class="body" id="cfg-form"></div></div>
      </section>

      <!-- LOGS -->
      <section class="view" id="v-logs">
        <div class="head"><div><p class="eyebrow">History</p><h1>Job log</h1><p>Tap a run to stream its output.</p></div>
          <button class="btn ghost" onclick="loadLogs()">Refresh</button></div>
        <div class="panel"><h3>Runs</h3><div class="body" id="log-list"></div></div>
      </section>
    </main>
  </div>
</div>

<!-- console drawer -->
<div class="console" id="console">
  <div class="bar">
    <span class="dot" id="con-dot"></span>
    <span class="title" id="con-title">—</span>
    <span class="pill" id="con-status">—</span>
    <button class="btn ghost x" onclick="closeConsole()">close ✕</button>
  </div>
  <pre id="con-log"></pre>
</div>

<script>
const $ = s => document.querySelector(s);
const api = async (path, opts={}) => {
  const r = await fetch('/api'+path, {headers:{'Content-Type':'application/json'}, ...opts});
  if(!r.ok){ let m; try{m=(await r.json()).detail}catch{m=r.statusText} throw new Error(m||'request failed'); }
  return r.status===204?null:r.json();
};
const toast = (msg,err=false)=>{const t=document.createElement('div');t.className='toast'+(err?' err':'');t.textContent=msg;document.body.appendChild(t);setTimeout(()=>t.remove(),3800)};
const esc = s => (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

/* nav */
document.querySelectorAll('#nav button').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('#nav button').forEach(x=>x.classList.remove('active'));
  b.classList.add('active');
  document.querySelectorAll('.view').forEach(v=>v.classList.remove('active'));
  $('#v-'+b.dataset.view).classList.add('active');
  const v=b.dataset.view;
  if(v==='overview')loadOverview(); if(v==='youtube')loadYouTube();
  if(v==='twitter')loadTwitter(); if(v==='affiliate')loadAffiliate();
  if(v==='schedule')loadSchedule(); if(v==='settings')loadConfig(); if(v==='logs')loadLogs();
});

/* ---------- overview + ticker ---------- */
async function loadOverview(){
  const s = await api('/status');
  $('#c-yt').textContent=s.youtube_accounts; $('#c-tw').textContent=s.twitter_accounts;
  $('#c-af').textContent=s.products; $('#c-sc').textContent=s.schedules;
  $('#stat-cards').innerHTML = [
    ['Queue', s.queue], ['YouTube', s.youtube_accounts], ['Twitter', s.twitter_accounts],
    ['Products', s.products], ['Schedules', s.schedules]
  ].map(([l,n])=>`<div class="card"><div class="stat">${n}<span class="lbl">${l}</span></div></div>`).join('');
  $('#recent').innerHTML = s.recent_jobs.length ? s.recent_jobs.map(jobRow).join('')
    : '<div class="empty">No runs yet. Trigger one from any section.</div>';
  const running = s.recent_jobs.find(j=>j.status==='running');
  $('#foot').textContent = s.queue ? `${s.queue} in queue` : 'idle';
  const parts = [
    `QUEUE <b>${s.queue}</b>`, `YOUTUBE <b>${s.youtube_accounts}</b>`,
    `TWITTER <b>${s.twitter_accounts}</b>`, `PRODUCTS <b>${s.products}</b>`,
    `SCHEDULES <b>${s.schedules}</b>`,
    running?`<span class="g">▶ RUNNING ${esc(running.label)}</span>`:`<span class="g">● IDLE</span>`
  ];
  $('#ticker').innerHTML = (parts.join(' &nbsp;·&nbsp; ')+' &nbsp;·&nbsp; ').repeat(2);
}
function jobRow(j){
  return `<div class="item"><div class="meta"><b>${esc(j.label)}</b><small>${j.id} · ${j.created_at?.replace('T',' ').slice(0,19)}</small></div>
    <div class="row"><span class="pill ${j.status}">${j.status}</span>
    <button class="btn ghost" onclick="openConsole('${j.id}')">view</button></div></div>`;
}

/* ---------- youtube ---------- */
async function loadYouTube(){
  const a = await api('/accounts/youtube');
  $('#yt-list').innerHTML = a.length ? a.map(x=>`
    <div class="item"><div class="meta"><b>${esc(x.nickname)}</b><small>${esc(x.niche)} · ${esc(x.language)} · ${x.id}</small></div>
    <div class="row">
      <button class="btn" onclick="ytRun('${x.id}',false)">Generate</button>
      <button class="btn go" onclick="ytRun('${x.id}',true)">Generate + Upload</button>
      <button class="btn danger" onclick="delAcc('youtube','${x.id}',loadYouTube)">✕</button>
    </div></div>`).join('') : '<div class="empty">No YouTube accounts yet.</div>';
}
async function addYouTube(){
  const b={nickname:$('#yt-nick').value.trim(),niche:$('#yt-niche').value.trim(),firefox_profile:$('#yt-fp').value.trim(),language:$('#yt-lang').value.trim()||'English'};
  if(!b.nickname||!b.niche)return toast('Nickname and niche are required',true);
  try{await api('/accounts/youtube',{method:'POST',body:JSON.stringify(b)});toast('Account added');['#yt-nick','#yt-niche','#yt-fp'].forEach(s=>$(s).value='');loadYouTube();}catch(e){toast(e.message,true)}
}
async function ytRun(id,upload){try{const j=await api('/actions/youtube',{method:'POST',body:JSON.stringify({account_id:id,upload})});toast('Run queued');openConsole(j.id);}catch(e){toast(e.message,true)}}

/* ---------- twitter ---------- */
async function loadTwitter(){
  const a = await api('/accounts/twitter');
  $('#tw-list').innerHTML = a.length ? a.map(x=>`
    <div class="item"><div class="meta"><b>${esc(x.nickname)}</b><small>${esc(x.topic)} · ${x.id}</small></div>
    <div class="row">
      <button class="btn go" onclick="twRun('${x.id}')">Post</button>
      <button class="btn danger" onclick="delAcc('twitter','${x.id}',loadTwitter)">✕</button>
    </div></div>`).join('') : '<div class="empty">No Twitter accounts yet.</div>';
}
async function addTwitter(){
  const b={nickname:$('#tw-nick').value.trim(),topic:$('#tw-topic').value.trim(),firefox_profile:$('#tw-fp').value.trim()};
  if(!b.nickname||!b.topic)return toast('Nickname and topic are required',true);
  try{await api('/accounts/twitter',{method:'POST',body:JSON.stringify(b)});toast('Account added');['#tw-nick','#tw-topic','#tw-fp'].forEach(s=>$(s).value='');loadTwitter();}catch(e){toast(e.message,true)}
}
async function twRun(id){
  const text = prompt('Optional: exact text to post. Leave blank to auto-generate.');
  if(text===null)return;
  try{const j=await api('/actions/twitter',{method:'POST',body:JSON.stringify({account_id:id,text:text||null})});toast('Post queued');openConsole(j.id);}catch(e){toast(e.message,true)}
}

/* ---------- affiliate ---------- */
async function loadAffiliate(){
  const [p,tw] = await Promise.all([api('/products'),api('/accounts/twitter')]);
  const name = id => (tw.find(t=>t.id===id)||{}).nickname || id;
  $('#af-acc').innerHTML = tw.map(t=>`<option value="${t.id}">${esc(t.nickname)}</option>`).join('') || '<option value="">(add a Twitter account first)</option>';
  $('#af-list').innerHTML = p.length ? p.map(x=>`
    <div class="item"><div class="meta"><b>${esc(x.affiliate_link)}</b><small>via ${esc(name(x.twitter_uuid))} · ${x.id}</small></div>
    <div class="row">
      <button class="btn go" onclick="afRun('${x.id}')">Pitch + Share</button>
      <button class="btn danger" onclick="delProduct('${x.id}')">✕</button>
    </div></div>`).join('') : '<div class="empty">No products yet.</div>';
}
async function addProduct(){
  const b={affiliate_link:$('#af-link').value.trim(),twitter_uuid:$('#af-acc').value};
  if(!b.affiliate_link||!b.twitter_uuid)return toast('Link and a Twitter account are required',true);
  try{await api('/products',{method:'POST',body:JSON.stringify(b)});toast('Product added');$('#af-link').value='';loadAffiliate();}catch(e){toast(e.message,true)}
}
async function afRun(id){try{const j=await api('/actions/afm',{method:'POST',body:JSON.stringify({product_id:id})});toast('Pitch queued');openConsole(j.id);}catch(e){toast(e.message,true)}}
async function delProduct(id){if(!confirm('Delete this product?'))return;await api('/products/'+id,{method:'DELETE'});loadAffiliate();}

/* ---------- outreach ---------- */
async function runOutreach(){try{const j=await api('/actions/outreach',{method:'POST'});toast('Outreach queued');openConsole(j.id);}catch(e){toast(e.message,true)}}

/* ---------- schedule ---------- */
async function loadSchedule(){
  const [list] = await Promise.all([api('/schedule')]);
  await scKindChange();
  $('#sc-list').innerHTML = list.length ? list.map(x=>`
    <div class="item"><div class="meta"><b>${x.kind}${x.account_id?' · '+x.account_id.slice(0,8):''}</b><small>${esc(x.trigger)} · next ${x.next_run?x.next_run.replace('T',' ').slice(0,16):'—'}</small></div>
    <button class="btn danger" onclick="delSchedule('${x.id}')">✕</button></div>`).join('') : '<div class="empty">No schedules.</div>';
}
function setCron(v){$('#sc-cron').value=v;}
async function scKindChange(){
  const kind=$('#sc-kind').value, wrap=$('#sc-target-wrap'), sel=$('#sc-target');
  if(kind==='outreach'){wrap.style.display='none';return;} wrap.style.display='';
  let opts=[];
  if(kind==='youtube'){opts=(await api('/accounts/youtube')).map(a=>[a.id,a.nickname]);}
  else if(kind==='twitter'){opts=(await api('/accounts/twitter')).map(a=>[a.id,a.nickname]);}
  else if(kind==='afm'){opts=(await api('/products')).map(p=>[p.id,p.affiliate_link]);}
  sel.innerHTML=opts.map(([v,l])=>`<option value="${v}">${esc(l)}</option>`).join('')||'<option value="">(none available)</option>';
}
async function addSchedule(){
  const kind=$('#sc-kind').value;
  const b={kind,cron:$('#sc-cron').value.trim(),account_id:kind==='outreach'?'':$('#sc-target').value,upload:true};
  try{await api('/schedule',{method:'POST',body:JSON.stringify(b)});toast('Schedule created');loadSchedule();}catch(e){toast(e.message,true)}
}
async function delSchedule(id){await api('/schedule/'+id,{method:'DELETE'});loadSchedule();}

/* ---------- settings (generic config editor) ---------- */
let CFG={};
async function loadConfig(){
  CFG = await api('/config');
  const rows = Object.entries(CFG).map(([k,v])=>{
    const id='cf_'+k;
    if(typeof v==='boolean')
      return `<div class="field"><label>${k}</label><select id="${id}"><option value="true"${v?' selected':''}>true</option><option value="false"${!v?' selected':''}>false</option></select></div>`;
    if(typeof v==='number')
      return `<div class="field"><label>${k}</label><input id="${id}" type="number" value="${v}"></div>`;
    if(typeof v==='object'&&v!==null)
      return `<div class="field" style="grid-column:1/-1"><label>${k} <span style="color:var(--dim)">(json)</span></label><textarea id="${id}">${esc(JSON.stringify(v,null,2))}</textarea></div>`;
    const secret=/key|password|token/i.test(k);
    return `<div class="field"><label>${k}</label><input id="${id}" type="${secret?'password':'text'}" value="${esc(String(v))}"></div>`;
  });
  $('#cfg-form').innerHTML = `<div class="fields">${rows.join('')}</div>`;
}
async function saveConfig(){
  const out={};
  for(const [k,v] of Object.entries(CFG)){
    const el=$('#cf_'+k); if(!el){out[k]=v;continue;}
    if(typeof v==='boolean')out[k]=el.value==='true';
    else if(typeof v==='number')out[k]=el.value===''?v:Number(el.value);
    else if(typeof v==='object'&&v!==null){try{out[k]=JSON.parse(el.value)}catch{return toast(`Invalid JSON in "${k}"`,true)}}
    else out[k]=el.value;
  }
  try{await api('/config',{method:'PUT',body:JSON.stringify(out)});CFG=out;toast('Config saved');}catch(e){toast(e.message,true)}
}

/* ---------- logs + console ---------- */
async function loadLogs(){
  const j = await api('/jobs?limit=60');
  $('#log-list').innerHTML = j.length ? j.map(jobRow).join('') : '<div class="empty">No runs yet.</div>';
}
let conTimer=null, conId=null;
function openConsole(id){
  conId=id; $('#console').classList.add('open'); $('#con-log').textContent='connecting…';
  pollConsole(); clearInterval(conTimer); conTimer=setInterval(pollConsole,1500);
}
function closeConsole(){$('#console').classList.remove('open');clearInterval(conTimer);conId=null;}
async function pollConsole(){
  if(!conId)return;
  try{
    const j = await api('/jobs/'+conId);
    $('#con-title').textContent=j.label;
    $('#con-status').textContent=j.status; $('#con-status').className='pill '+j.status;
    $('#con-dot').className='dot '+j.status;
    const pre=$('#con-log'); const atBottom = pre.scrollHeight-pre.scrollTop-pre.clientHeight < 40;
    pre.textContent = j.log || '(no output yet)';
    if(atBottom)pre.scrollTop=pre.scrollHeight;
    if(j.status==='done'||j.status==='error'){clearInterval(conTimer); if(['overview'].includes(current()))loadOverview();}
  }catch(e){$('#con-log').textContent='error: '+e.message;clearInterval(conTimer);}
}
const current = ()=>document.querySelector('#nav button.active').dataset.view;

/* ---------- shared ---------- */
async function delAcc(provider,id,after){if(!confirm('Delete this account?'))return;await api(`/accounts/${provider}/${id}`,{method:'DELETE'});after();}

loadOverview();
setInterval(()=>{if(current()==='overview')loadOverview();},6000);
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.triggers.cron import CronTrigger

_scheduler = BackgroundScheduler(
    jobstores={"default": SQLAlchemyJobStore(url=f"sqlite:///{os.path.join(MP_DIR, 'scheduler.sqlite')}")},
    job_defaults={"coalesce": True, "misfire_grace_time": 3600, "max_instances": 1},
    timezone="UTC")
_scheduler.start()

app = FastAPI(title="MoneyPrinterV2 Dashboard")


def _read_config():
    return json.load(open(CONFIG_PATH))


def _write_config(data):
    with open(CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2); f.flush(); os.fsync(f.fileno())


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.get("/api/health")
def health():
    return {"ok": True, "queue": _Q.unfinished_tasks}


@app.get("/api/status")
def status():
    from cache import get_accounts, get_products
    return {"youtube_accounts": len(get_accounts("youtube")),
            "twitter_accounts": len(get_accounts("twitter")),
            "products": len(get_products()), "schedules": len(_scheduler.get_jobs()),
            "queue": _Q.unfinished_tasks, "recent_jobs": list_jobs(8)}


@app.get("/api/config")
def get_config():
    return _read_config()


@app.put("/api/config")
def put_config(data: dict = Body(...)):
    if not isinstance(data, dict):
        raise HTTPException(400, "Config must be a JSON object.")
    _write_config(data); return {"ok": True}


class YouTubeAccount(BaseModel):
    nickname: str; firefox_profile: str; niche: str; language: str = "English"


class TwitterAccount(BaseModel):
    nickname: str; firefox_profile: str; topic: str


@app.get("/api/accounts/{provider}")
def accounts(provider: str):
    from cache import get_accounts
    if provider not in ("youtube", "twitter"):
        raise HTTPException(404, "provider must be 'youtube' or 'twitter'")
    return get_accounts(provider)


@app.post("/api/accounts/youtube")
def add_youtube(acc: YouTubeAccount):
    from cache import add_account
    data = acc.model_dump(); data.update({"id": str(uuid4()), "videos": []})
    add_account("youtube", data); return data


@app.post("/api/accounts/twitter")
def add_twitter(acc: TwitterAccount):
    from cache import add_account
    data = acc.model_dump(); data.update({"id": str(uuid4()), "posts": []})
    add_account("twitter", data); return data


@app.delete("/api/accounts/{provider}/{account_id}")
def delete_account(provider: str, account_id: str):
    from cache import remove_account
    if provider not in ("youtube", "twitter"):
        raise HTTPException(404, "unknown provider")
    remove_account(provider, account_id)
    for job in _scheduler.get_jobs():
        if job.kwargs.get("account_id") == account_id:
            job.remove()
    return {"ok": True}


class Product(BaseModel):
    affiliate_link: str; twitter_uuid: str


@app.get("/api/products")
def products():
    from cache import get_products
    return get_products()


@app.post("/api/products")
def add_product(p: Product):
    from cache import add_product as _add
    data = p.model_dump(); data["id"] = str(uuid4()); _add(data); return data


@app.delete("/api/products/{product_id}")
def delete_product(product_id: str):
    path = os.path.join(MP_DIR, "afm.json")
    if os.path.exists(path):
        parsed = json.load(open(path))
        parsed["products"] = [p for p in parsed.get("products", []) if p.get("id") != product_id]
        json.dump(parsed, open(path, "w"), indent=4)
    return {"ok": True}


class YouTubeRun(BaseModel):
    account_id: str; upload: bool = False


class TwitterRun(BaseModel):
    account_id: str; text: Optional[str] = None


class AfmRun(BaseModel):
    product_id: str


@app.post("/api/actions/youtube")
def run_youtube(body: YouTubeRun):
    try:
        return action_youtube_generate(body.account_id, body.upload).to_dict()
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/actions/twitter")
def run_twitter(body: TwitterRun):
    try:
        return action_twitter_post(body.account_id, body.text).to_dict()
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/actions/afm")
def run_afm(body: AfmRun):
    try:
        return action_afm_share(body.product_id).to_dict()
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/actions/outreach")
def run_outreach():
    return action_outreach_run().to_dict()


@app.get("/api/jobs")
def get_jobs(limit: int = 50):
    return list_jobs(limit)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job.to_dict(include_logs=True)


class Schedule(BaseModel):
    kind: str; account_id: str = ""; cron: str; upload: bool = True


def _serialize_job(job):
    return {"id": job.id, "kind": job.kwargs.get("kind"),
            "account_id": job.kwargs.get("account_id"), "upload": job.kwargs.get("upload"),
            "trigger": str(job.trigger),
            "next_run": job.next_run_time.isoformat() if job.next_run_time else None}


@app.get("/api/schedule")
def list_schedule():
    return [_serialize_job(j) for j in _scheduler.get_jobs()]


@app.post("/api/schedule")
def add_schedule(s: Schedule):
    if s.kind not in ("youtube", "twitter", "afm", "outreach"):
        raise HTTPException(400, "invalid kind")
    try:
        trigger = CronTrigger.from_crontab(s.cron, timezone="UTC")
    except Exception:
        raise HTTPException(400, "cron must be a 5-field crontab, e.g. '0 10 * * *'")
    job = _scheduler.add_job("dashboard_server:run_scheduled", trigger=trigger,
                             kwargs={"kind": s.kind, "account_id": s.account_id, "upload": s.upload})
    return _serialize_job(job)


@app.delete("/api/schedule/{job_id}")
def delete_schedule(job_id: str):
    job = _scheduler.get_job(job_id)
    if not job:
        raise HTTPException(404, "schedule not found")
    job.remove(); return {"ok": True}