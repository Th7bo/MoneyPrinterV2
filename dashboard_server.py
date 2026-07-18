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
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
<meta name="theme-color" content="#0e1218" />
<meta name="apple-mobile-web-app-capable" content="yes" />
<title>MoneyPrinter · Console</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root{
    --ink:#0e1218; --panel:#151b24; --raised:#1b222d; --line:#28303c;
    --text:#e2e7ee; --muted:#8b95a6; --dim:#5c6675;
    --mint:#4ea88b; --amber:#d9a441; --red:#d96a5a; --gold:#b0894f;
    --mono:'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
    --sans:'Space Grotesk',Inter,system-ui,-apple-system,sans-serif;
    --tab-h:calc(58px + env(safe-area-inset-bottom));
    --tap:46px;
  }
  *{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
  html,body{margin:0;padding:0}
  body{
    background:var(--ink);color:var(--text);font-family:var(--sans);
    font-size:16px;line-height:1.5;overscroll-behavior-y:none;
    background-image:radial-gradient(circle at 100% 0,rgba(78,168,139,.06),transparent 45%);
  }
  button{font-family:inherit;cursor:pointer}
  /* 16px inputs stop iOS from zooming the viewport on focus */
  input,select,textarea{font-family:var(--sans);font-size:16px}

  /* ---------- top bar ---------- */
  .top{
    position:sticky;top:0;z-index:30;background:rgba(14,18,24,.92);
    backdrop-filter:blur(12px);border-bottom:1px solid var(--line);
    padding:calc(env(safe-area-inset-top) + 10px) 16px 0;
  }
  .top .r1{display:flex;align-items:center;gap:10px}
  .mark{width:28px;height:28px;border-radius:7px;background:linear-gradient(150deg,var(--mint),#2f6f5c);
        display:grid;place-items:center;font-weight:700;color:#08120e;font-size:15px;flex:0 0 auto}
  .top h1{font-size:17px;font-weight:600;margin:0;flex:1;letter-spacing:.2px}
  .top .live{font-family:var(--mono);font-size:11px;color:var(--muted);display:flex;align-items:center;gap:6px}
  .ticker{overflow:hidden;margin-top:8px;position:relative}
  .ticker::after{content:"";position:absolute;left:0;right:0;bottom:0;height:1px;
    background:repeating-linear-gradient(90deg,var(--gold) 0 6px,transparent 6px 12px);opacity:.3}
  .ticker .rail{display:flex;white-space:nowrap;font-family:var(--mono);font-size:11px;color:var(--muted);
    padding:7px 0 8px;animation:scroll 30s linear infinite;width:max-content;gap:0}
  .ticker .rail b{color:var(--text);font-weight:600}
  .ticker .rail .g{color:var(--mint)}
  @keyframes scroll{from{transform:translateX(0)}to{transform:translateX(-50%)}}

  /* ---------- layout ---------- */
  main{padding:16px 14px calc(var(--tab-h) + 26px);max-width:920px;margin:0 auto}
  .head{margin-bottom:14px}
  .eyebrow{font-family:var(--mono);font-size:10.5px;letter-spacing:2px;text-transform:uppercase;color:var(--gold);margin:0 0 3px}
  .head h2{font-size:21px;font-weight:600;margin:0;letter-spacing:.2px}
  .head p{margin:5px 0 0;color:var(--muted);font-size:14px}

  .panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden;margin-bottom:14px}
  .panel h3{margin:0;padding:13px 14px;font-size:13px;font-weight:600;letter-spacing:.3px;
    border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:center;gap:8px}
  .panel h3 .hint{font-family:var(--mono);font-size:10.5px;color:var(--dim);font-weight:400}
  .panel .body{padding:14px}

  /* stat strip: horizontal scroll beats a cramped grid on a phone */
  .stats{display:flex;gap:10px;overflow-x:auto;padding:2px 14px 6px;margin:0 -14px 14px;scrollbar-width:none;scroll-snap-type:x mandatory}
  .stats::-webkit-scrollbar{display:none}
  .stats .card{flex:0 0 auto;min-width:104px;background:var(--panel);border:1px solid var(--line);
    border-radius:12px;padding:13px 15px;scroll-snap-align:start}
  .stats .n{font-family:var(--mono);font-size:26px;font-weight:600;letter-spacing:-1px;line-height:1.1}
  .stats .l{font-size:12px;color:var(--muted);margin-top:3px}

  /* list rows stack on mobile so buttons get full width */
  .item{border:1px solid var(--line);border-radius:11px;background:var(--raised);padding:13px 14px;margin-bottom:10px}
  .item:last-child{margin-bottom:0}
  .item .meta b{font-weight:600;font-size:15px;display:block;word-break:break-word}
  .item .meta small{display:block;font-family:var(--mono);font-size:11px;color:var(--dim);margin-top:3px;word-break:break-all;line-height:1.45}
  .item .acts{display:flex;gap:8px;margin-top:12px}
  .item .acts .btn{flex:1;justify-content:center}
  .item .acts .btn.icon{flex:0 0 var(--tap);max-width:var(--tap)}

  label{display:block;font-size:12.5px;color:var(--muted);margin:0 0 6px;letter-spacing:.2px}
  input[type=text],input[type=number],input[type=password],select,textarea{
    width:100%;background:var(--ink);border:1px solid var(--line);color:var(--text);
    border-radius:10px;padding:12px;outline:none;min-height:var(--tap);appearance:none}
  select{background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='12' height='8'><path d='M1 1l5 5 5-5' stroke='%238b95a6' stroke-width='1.6' fill='none' stroke-linecap='round'/></svg>");
    background-repeat:no-repeat;background-position:right 13px center;padding-right:34px}
  input:focus,select:focus,textarea:focus{border-color:var(--mint);box-shadow:0 0 0 3px rgba(78,168,139,.13)}
  textarea{font-family:var(--mono);font-size:13px;resize:vertical;min-height:96px;line-height:1.5}
  .field{margin-bottom:12px}

  .btn{background:var(--raised);border:1px solid var(--line);color:var(--text);padding:0 16px;
    border-radius:10px;font-size:14.5px;min-height:var(--tap);display:inline-flex;align-items:center;
    justify-content:center;gap:7px;transition:.12s;font-weight:500}
  .btn:active{transform:scale(.98)}
  .btn.go{background:var(--mint);color:#08130f;border-color:transparent;font-weight:600}
  .btn.wide{display:flex;width:100%}
  .btn.ghost{background:none}
  .btn.icon{padding:0;color:var(--red);border-color:transparent;background:rgba(217,106,90,.1);font-size:17px}
  .btn:disabled{opacity:.5}

  .pill{font-family:var(--mono);font-size:10.5px;padding:4px 9px;border-radius:20px;border:1px solid var(--line);
    color:var(--muted);text-transform:uppercase;letter-spacing:.5px;white-space:nowrap}
  .pill.running{color:var(--mint);border-color:rgba(78,168,139,.35);background:rgba(78,168,139,.08)}
  .pill.queued{color:var(--amber);border-color:rgba(217,164,65,.3)}
  .pill.done{color:var(--dim)} .pill.error{color:var(--red);border-color:rgba(217,106,90,.3)}
  .dot{width:7px;height:7px;border-radius:50%;display:inline-block;background:var(--dim);flex:0 0 auto}
  .dot.running{background:var(--mint);animation:pulse 1.6s infinite}
  .dot.queued{background:var(--amber)} .dot.error{background:var(--red)}
  @keyframes pulse{0%{box-shadow:0 0 0 0 rgba(78,168,139,.5)}70%{box-shadow:0 0 0 7px rgba(78,168,139,0)}100%{box-shadow:0 0 0 0 rgba(78,168,139,0)}}

  .rowline{display:flex;align-items:center;gap:8px;justify-content:space-between}
  .empty{color:var(--dim);font-size:14px;padding:22px 14px;text-align:center;border:1px dashed var(--line);border-radius:11px}
  .note{font-size:13px;color:var(--muted);background:rgba(217,164,65,.07);border:1px solid rgba(217,164,65,.2);
    border-radius:10px;padding:12px;margin-top:14px;line-height:1.55}
  .note b{color:var(--amber)}
  .chips{display:flex;gap:8px;overflow-x:auto;padding-bottom:4px;margin-bottom:12px;scrollbar-width:none}
  .chips::-webkit-scrollbar{display:none}
  .chips .btn{flex:0 0 auto;min-height:40px;font-size:13.5px}

  .view{display:none} .view.active{display:block;animation:fade .2s ease}
  @keyframes fade{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:none}}

  /* ---------- bottom tab bar (thumb reach) ---------- */
  .tabs{
    position:fixed;left:0;right:0;bottom:0;z-index:35;display:flex;
    background:rgba(19,24,32,.96);backdrop-filter:blur(14px);border-top:1px solid var(--line);
    padding-bottom:env(safe-area-inset-bottom);overflow-x:auto;scrollbar-width:none}
  .tabs::-webkit-scrollbar{display:none}
  .tabs button{
    flex:1 0 20%;min-width:66px;background:none;border:0;color:var(--dim);padding:8px 4px 9px;
    display:flex;flex-direction:column;align-items:center;gap:3px;font-size:10.5px;letter-spacing:.2px}
  .tabs button svg{width:21px;height:21px;stroke:currentColor;fill:none;stroke-width:1.7;stroke-linecap:round;stroke-linejoin:round}
  .tabs button.active{color:var(--mint)}
  .tabs button .badge{position:absolute;transform:translate(15px,-7px);background:var(--mint);color:#08130f;
    font-family:var(--mono);font-size:9px;font-weight:600;border-radius:9px;padding:1px 5px;min-width:16px}

  /* ---------- console bottom sheet ---------- */
  .sheet{position:fixed;inset:0;z-index:50;display:none;background:rgba(0,0,0,.55)}
  .sheet.open{display:block}
  .sheet .card{position:absolute;left:0;right:0;bottom:0;height:82vh;background:#0a0d12;
    border-radius:16px 16px 0 0;border-top:1px solid var(--line);display:flex;flex-direction:column;
    animation:up .26s cubic-bezier(.4,0,.2,1);padding-bottom:env(safe-area-inset-bottom)}
  @keyframes up{from{transform:translateY(100%)}to{transform:none}}
  .sheet .grab{width:38px;height:4px;background:var(--line);border-radius:3px;margin:9px auto 4px;flex:0 0 auto}
  .sheet .bar{display:flex;align-items:center;gap:9px;padding:8px 14px 11px;border-bottom:1px solid var(--line)}
  .sheet .bar .t{font-family:var(--mono);font-size:12px;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .sheet pre{margin:0;flex:1;overflow:auto;padding:14px;font-family:var(--mono);font-size:12px;
    line-height:1.55;color:#c9d2de;white-space:pre-wrap;word-break:break-word;-webkit-overflow-scrolling:touch}
  .sheet .close{width:36px;height:36px;border-radius:9px;border:1px solid var(--line);background:none;color:var(--muted);font-size:16px}

  /* modal for text input (replaces prompt()) */
  .modal{position:fixed;inset:0;z-index:60;display:none;background:rgba(0,0,0,.6);align-items:flex-end}
  .modal.open{display:flex}
  .modal .card{width:100%;background:var(--panel);border-radius:16px 16px 0 0;border-top:1px solid var(--line);
    padding:18px 16px calc(18px + env(safe-area-inset-bottom));animation:up .24s cubic-bezier(.4,0,.2,1)}
  .modal h4{margin:0 0 4px;font-size:16px;font-weight:600}
  .modal p{margin:0 0 13px;color:var(--muted);font-size:13.5px}
  .modal .acts{display:flex;gap:9px;margin-top:13px}
  .modal .acts .btn{flex:1}

  .toast{position:fixed;left:14px;right:14px;bottom:calc(var(--tab-h) + 12px);background:var(--raised);
    border:1px solid var(--line);border-left:3px solid var(--mint);padding:13px 15px;border-radius:11px;
    font-size:14px;z-index:70;animation:fade .2s;box-shadow:0 8px 28px rgba(0,0,0,.4)}
  .toast.err{border-left-color:var(--red)}

  /* ---------- desktop: promote tabs to a sidebar ---------- */
  @media(min-width:900px){
    body{display:grid;grid-template-columns:212px 1fr}
    .tabs{position:sticky;top:0;left:auto;height:100vh;flex-direction:column;border-top:0;
      border-right:1px solid var(--line);padding:18px 12px;gap:3px;overflow:visible;align-content:start}
    .tabs button{flex:0 0 auto;flex-direction:row;justify-content:flex-start;gap:11px;font-size:14px;
      padding:11px 13px;border-radius:9px;width:100%}
    .tabs button.active{background:var(--raised);color:var(--text)}
    .tabs button.active svg{color:var(--mint)}
    .tabs button .badge{position:static;transform:none;margin-left:auto}
    .top{padding-top:14px} main{padding:20px 26px 60px}
    .item{display:flex;align-items:center;justify-content:space-between;gap:14px}
    .item .acts{margin-top:0;flex:0 0 auto}
    .item .acts .btn{flex:0 0 auto}
    .fields2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
    .sheet .card{height:62vh}
    .modal{align-items:center;justify-content:center}
    .modal .card{max-width:460px;border-radius:14px;border:1px solid var(--line)}
    .toast{left:auto;right:20px;bottom:20px;max-width:340px}
  }
  @media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>
</head>
<body>

<nav class="tabs" id="tabs">
  <button data-view="overview" class="active"><svg viewBox="0 0 24 24"><path d="M3 12h4l3 8 4-16 3 8h4"/></svg>Overview</button>
  <button data-view="youtube"><svg viewBox="0 0 24 24"><rect x="2" y="5" width="20" height="14" rx="4"/><path d="M10 9.5l5 2.5-5 2.5z"/></svg>YouTube</button>
  <button data-view="twitter"><svg viewBox="0 0 24 24"><path d="M4 4l7 9m9-9l-8.5 10L20 20h-4l-5.5-7L4 20"/></svg>Twitter</button>
  <button data-view="affiliate"><svg viewBox="0 0 24 24"><path d="M9 15l6-6"/><path d="M11 6l1-1a4 4 0 116 6l-1 1"/><path d="M13 18l-1 1a4 4 0 11-6-6l1-1"/></svg>Affiliate</button>
  <button data-view="outreach"><svg viewBox="0 0 24 24"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3 7l9 6 9-6"/></svg>Outreach</button>
  <button data-view="schedule"><svg viewBox="0 0 24 24"><rect x="3" y="5" width="18" height="16" rx="2"/><path d="M3 10h18M8 3v4M16 3v4"/></svg>Schedule</button>
  <button data-view="logs"><svg viewBox="0 0 24 24"><path d="M4 6h16M4 12h16M4 18h10"/></svg>Jobs<span class="badge" id="tab-badge" style="display:none">0</span></button>
  <button data-view="settings"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 00.3 1.9l.1.1a2 2 0 11-2.8 2.8l-.1-.1a1.7 1.7 0 00-2.9 1.2V21a2 2 0 11-4 0v-.1A1.7 1.7 0 007 19.4l-.1.1a2 2 0 11-2.8-2.8l.1-.1a1.7 1.7 0 00-1.2-2.9H3a2 2 0 110-4h.1A1.7 1.7 0 004.6 7l-.1-.1a2 2 0 112.8-2.8l.1.1a1.7 1.7 0 002.9-1.2V3a2 2 0 114 0v.1A1.7 1.7 0 0017 4.6l.1-.1a2 2 0 112.8 2.8l-.1.1a1.7 1.7 0 001.2 2.9H21a2 2 0 110 4h-.1a1.7 1.7 0 00-1.5 1z"/></svg>Settings</button>
</nav>

<div>
  <header class="top">
    <div class="r1">
      <div class="mark">$</div>
      <h1 id="page-title">Overview</h1>
      <span class="live"><span class="dot" id="live-dot"></span><span id="live-txt">idle</span></span>
    </div>
    <div class="ticker"><div class="rail" id="ticker">loading…</div></div>
  </header>

  <main>
    <!-- OVERVIEW -->
    <section class="view active" id="v-overview">
      <div class="head"><p class="eyebrow">Mission control</p><h2>Overview</h2><p>What the printer is doing right now.</p></div>
      <div class="stats" id="stat-cards"></div>
      <div class="panel"><h3>Recent runs <span class="hint">newest first</span></h3><div class="body" id="recent"></div></div>
    </section>

    <!-- YOUTUBE -->
    <section class="view" id="v-youtube">
      <div class="head"><p class="eyebrow">Shorts automation</p><h2>YouTube</h2><p>Generate and upload Shorts. Uploads drive a logged-in Firefox profile.</p></div>
      <div class="panel"><h3>Accounts</h3><div class="body" id="yt-list"></div></div>
      <div class="panel"><h3>Add account</h3><div class="body">
        <div class="fields2">
          <div class="field"><label>Nickname</label><input id="yt-nick" type="text" placeholder="finance-shorts"></div>
          <div class="field"><label>Niche</label><input id="yt-niche" type="text" placeholder="personal finance"></div>
        </div>
        <div class="fields2">
          <div class="field"><label>Firefox profile path</label><input id="yt-fp" type="text" placeholder="/profiles/youtube"></div>
          <div class="field"><label>Language</label><input id="yt-lang" type="text" value="English"></div>
        </div>
        <button class="btn go wide" onclick="addYouTube()">Add account</button>
      </div></div>
    </section>

    <!-- TWITTER -->
    <section class="view" id="v-twitter">
      <div class="head"><p class="eyebrow">Posting bot</p><h2>Twitter / X</h2><p>Post generated content, or your own text, per account.</p></div>
      <div class="panel"><h3>Accounts</h3><div class="body" id="tw-list"></div></div>
      <div class="panel"><h3>Add account</h3><div class="body">
        <div class="fields2">
          <div class="field"><label>Nickname</label><input id="tw-nick" type="text" placeholder="build-in-public"></div>
          <div class="field"><label>Topic</label><input id="tw-topic" type="text" placeholder="indie hacking"></div>
        </div>
        <div class="field"><label>Firefox profile path</label><input id="tw-fp" type="text" placeholder="/profiles/twitter"></div>
        <button class="btn go wide" onclick="addTwitter()">Add account</button>
      </div></div>
    </section>

    <!-- AFFILIATE -->
    <section class="view" id="v-affiliate">
      <div class="head"><p class="eyebrow">Affiliate marketing</p><h2>Affiliate</h2><p>Turn a link into a pitch and share it from a Twitter account.</p></div>
      <div class="panel"><h3>Products</h3><div class="body" id="af-list"></div></div>
      <div class="panel"><h3>Add product</h3><div class="body">
        <div class="field"><label>Affiliate link</label><input id="af-link" type="text" placeholder="https://amzn.to/..."></div>
        <div class="field"><label>Share from</label><select id="af-acc"></select></div>
        <button class="btn go wide" onclick="addProduct()">Add product</button>
      </div></div>
    </section>

    <!-- OUTREACH -->
    <section class="view" id="v-outreach">
      <div class="head"><p class="eyebrow">Cold outreach</p><h2>Outreach</h2><p>Scrape local businesses and email them over SMTP.</p></div>
      <div class="panel"><h3>Run outreach</h3><div class="body">
        <p style="margin:0 0 14px;color:var(--muted);font-size:14px">Uses the niche, SMTP credentials and message template from <b style="color:var(--text)">Settings</b>.</p>
        <button class="btn go wide" onclick="runOutreach()">Start scrape &amp; email</button>
        <div class="note"><b>Heads up</b> — unsolicited bulk email can violate anti-spam laws (CAN-SPAM, GDPR) and your SMTP provider's terms. Make sure you have a lawful basis and an unsubscribe path.</div>
      </div></div>
    </section>

    <!-- SCHEDULE -->
    <section class="view" id="v-schedule">
      <div class="head"><p class="eyebrow">Cron</p><h2>Schedule</h2><p>Recurring runs, persisted across restarts. Times are UTC.</p></div>
      <div class="panel"><h3>Active</h3><div class="body" id="sc-list"></div></div>
      <div class="panel"><h3>New schedule</h3><div class="body">
        <div class="field"><label>What to run</label><select id="sc-kind" onchange="scKindChange()">
          <option value="youtube">YouTube — generate + upload</option>
          <option value="twitter">Twitter — post</option>
          <option value="afm">Affiliate — pitch + share</option>
          <option value="outreach">Outreach — scrape + email</option>
        </select></div>
        <div class="field" id="sc-target-wrap"><label>Target</label><select id="sc-target"></select></div>
        <div class="field"><label>Cron (min hour day month weekday, UTC)</label><input id="sc-cron" type="text" value="0 10 * * *" inputmode="text" autocapitalize="off" autocorrect="off"></div>
        <div class="chips">
          <button class="btn ghost" onclick="setCron('0 10 * * *')">Once/day</button>
          <button class="btn ghost" onclick="setCron('0 10,16 * * *')">Twice/day</button>
          <button class="btn ghost" onclick="setCron('0 8,12,18 * * *')">3×/day</button>
          <button class="btn ghost" onclick="setCron('0 */6 * * *')">Every 6h</button>
        </div>
        <button class="btn go wide" onclick="addSchedule()">Create schedule</button>
      </div></div>
    </section>

    <!-- LOGS -->
    <section class="view" id="v-logs">
      <div class="head"><p class="eyebrow">History</p><h2>Jobs</h2><p>Tap a run to stream its output.</p></div>
      <div class="panel"><h3>Runs <span class="hint" id="log-hint"></span></h3><div class="body" id="log-list"></div></div>
    </section>

    <!-- SETTINGS -->
    <section class="view" id="v-settings">
      <div class="head"><p class="eyebrow">config.json</p><h2>Settings</h2><p>Edit the live config. Objects and arrays are edited as JSON.</p></div>
      <div class="panel"><h3>Configuration</h3><div class="body" id="cfg-form"></div></div>
      <button class="btn go wide" onclick="saveConfig()">Save config</button>
      <div class="panel" style="margin-top:14px"><h3>Session</h3><div class="body">
        <p style="margin:0 0 13px;color:var(--muted);font-size:13.5px">Signed in. Sessions last 14 days per device.</p>
        <button class="btn wide" onclick="logout()">Sign out</button>
      </div></div>
    </section>
  </main>
</div>

<!-- console sheet -->
<div class="sheet" id="sheet" onclick="if(event.target===this)closeConsole()">
  <div class="card">
    <div class="grab"></div>
    <div class="bar">
      <span class="dot" id="con-dot"></span>
      <span class="t" id="con-title">—</span>
      <span class="pill" id="con-status">—</span>
      <button class="close" onclick="closeConsole()">✕</button>
    </div>
    <pre id="con-log"></pre>
  </div>
</div>

<!-- text input sheet -->
<div class="modal" id="modal">
  <div class="card">
    <h4 id="m-title">Post text</h4>
    <p id="m-sub">Leave blank to auto-generate.</p>
    <textarea id="m-input" placeholder="What should it say?"></textarea>
    <div class="acts">
      <button class="btn" onclick="closeModal()">Cancel</button>
      <button class="btn go" id="m-ok">Post</button>
    </div>
  </div>
</div>

<script>
const $ = s => document.querySelector(s);
const api = async (p,o={}) => {
  const r = await fetch('/api'+p,{headers:{'Content-Type':'application/json'},...o});
  if(r.status===401){location.href='/login';throw new Error('Session expired')}
  if(!r.ok){let m;try{m=(await r.json()).detail}catch{m=r.statusText}throw new Error(m||'request failed')}
  return r.status===204?null:r.json();
};
async function logout(){
  try{await fetch('/api/logout',{method:'POST'})}catch{}
  location.href='/login';
}
const toast=(m,e=false)=>{const t=document.createElement('div');t.className='toast'+(e?' err':'');t.textContent=m;document.body.appendChild(t);setTimeout(()=>t.remove(),3600)};
const esc=s=>(s||'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const short=s=>s&&s.length>34?s.slice(0,34)+'…':s;

/* nav */
const TITLES={overview:'Overview',youtube:'YouTube',twitter:'Twitter / X',affiliate:'Affiliate',outreach:'Outreach',schedule:'Schedule',logs:'Jobs',settings:'Settings'};
document.querySelectorAll('#tabs button').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('#tabs button').forEach(x=>x.classList.remove('active'));
  b.classList.add('active');
  document.querySelectorAll('.view').forEach(v=>v.classList.remove('active'));
  const v=b.dataset.view;
  $('#v-'+v).classList.add('active');
  $('#page-title').textContent=TITLES[v];
  window.scrollTo({top:0});
  ({overview:loadOverview,youtube:loadYouTube,twitter:loadTwitter,affiliate:loadAffiliate,
    schedule:loadSchedule,settings:loadConfig,logs:loadLogs}[v]||(()=>{}))();
});
const current=()=>document.querySelector('#tabs button.active').dataset.view;

/* overview */
async function loadOverview(){
  const s=await api('/status');
  $('#stat-cards').innerHTML=[['Queue',s.queue],['YouTube',s.youtube_accounts],['Twitter',s.twitter_accounts],
    ['Products',s.products],['Schedules',s.schedules]]
    .map(([l,n])=>`<div class="card"><div class="n">${n}</div><div class="l">${l}</div></div>`).join('');
  $('#recent').innerHTML=s.recent_jobs.length?s.recent_jobs.map(jobRow).join(''):'<div class="empty">No runs yet. Start one from any tab.</div>';
  const run=s.recent_jobs.find(j=>j.status==='running');
  $('#live-dot').className='dot'+(run?' running':'');
  $('#live-txt').textContent=run?'running':(s.queue?s.queue+' queued':'idle');
  const b=$('#tab-badge'); if(s.queue){b.style.display='';b.textContent=s.queue}else b.style.display='none';
  const parts=[`QUEUE <b>${s.queue}</b>`,`YT <b>${s.youtube_accounts}</b>`,`TW <b>${s.twitter_accounts}</b>`,
    `PRODUCTS <b>${s.products}</b>`,`CRON <b>${s.schedules}</b>`,
    run?`<span class="g">▶ ${esc(run.label)}</span>`:`<span class="g">● IDLE</span>`];
  $('#ticker').innerHTML=(parts.join(' &nbsp;·&nbsp; ')+' &nbsp;·&nbsp; ').repeat(2);
}
function jobRow(j){
  return `<div class="item"><div class="meta"><b>${esc(j.label)}</b>
    <small>${(j.created_at||'').replace('T',' ').slice(0,19)} · ${j.id}</small></div>
    <div class="acts"><span class="pill ${j.status}" style="align-self:center">${j.status}</span>
    <button class="btn" onclick="openConsole('${j.id}')">View log</button></div></div>`;
}

/* youtube */
async function loadYouTube(){
  const a=await api('/accounts/youtube');
  $('#yt-list').innerHTML=a.length?a.map(x=>`<div class="item">
    <div class="meta"><b>${esc(x.nickname)}</b><small>${esc(x.niche)} · ${esc(x.language)}</small></div>
    <div class="acts">
      <button class="btn" onclick="ytRun('${x.id}',false)">Generate</button>
      <button class="btn go" onclick="ytRun('${x.id}',true)">+ Upload</button>
      <button class="btn icon" onclick="delAcc('youtube','${x.id}',loadYouTube)">✕</button>
    </div></div>`).join(''):'<div class="empty">No YouTube accounts yet.</div>';
}
async function addYouTube(){
  const b={nickname:$('#yt-nick').value.trim(),niche:$('#yt-niche').value.trim(),
    firefox_profile:$('#yt-fp').value.trim(),language:$('#yt-lang').value.trim()||'English'};
  if(!b.nickname||!b.niche)return toast('Nickname and niche are required',true);
  try{await api('/accounts/youtube',{method:'POST',body:JSON.stringify(b)});toast('Account added');
    ['#yt-nick','#yt-niche','#yt-fp'].forEach(s=>$(s).value='');loadYouTube()}catch(e){toast(e.message,true)}
}
async function ytRun(id,up){try{const j=await api('/actions/youtube',{method:'POST',body:JSON.stringify({account_id:id,upload:up})});toast('Run queued');openConsole(j.id)}catch(e){toast(e.message,true)}}

/* twitter */
async function loadTwitter(){
  const a=await api('/accounts/twitter');
  $('#tw-list').innerHTML=a.length?a.map(x=>`<div class="item">
    <div class="meta"><b>${esc(x.nickname)}</b><small>${esc(x.topic)}</small></div>
    <div class="acts">
      <button class="btn go" onclick="twRun('${x.id}')">Post</button>
      <button class="btn icon" onclick="delAcc('twitter','${x.id}',loadTwitter)">✕</button>
    </div></div>`).join(''):'<div class="empty">No Twitter accounts yet.</div>';
}
async function addTwitter(){
  const b={nickname:$('#tw-nick').value.trim(),topic:$('#tw-topic').value.trim(),firefox_profile:$('#tw-fp').value.trim()};
  if(!b.nickname||!b.topic)return toast('Nickname and topic are required',true);
  try{await api('/accounts/twitter',{method:'POST',body:JSON.stringify(b)});toast('Account added');
    ['#tw-nick','#tw-topic','#tw-fp'].forEach(s=>$(s).value='');loadTwitter()}catch(e){toast(e.message,true)}
}
function twRun(id){
  openModal('Post text','Leave blank to auto-generate with the LLM.','Post',async val=>{
    try{const j=await api('/actions/twitter',{method:'POST',body:JSON.stringify({account_id:id,text:val||null})});
      toast('Post queued');openConsole(j.id)}catch(e){toast(e.message,true)}
  });
}

/* affiliate */
async function loadAffiliate(){
  const [p,tw]=await Promise.all([api('/products'),api('/accounts/twitter')]);
  const nm=id=>(tw.find(t=>t.id===id)||{}).nickname||id.slice(0,8);
  $('#af-acc').innerHTML=tw.map(t=>`<option value="${t.id}">${esc(t.nickname)}</option>`).join('')||'<option value="">(add a Twitter account first)</option>';
  $('#af-list').innerHTML=p.length?p.map(x=>`<div class="item">
    <div class="meta"><b>${esc(short(x.affiliate_link))}</b><small>via ${esc(nm(x.twitter_uuid))}</small></div>
    <div class="acts">
      <button class="btn go" onclick="afRun('${x.id}')">Pitch + Share</button>
      <button class="btn icon" onclick="delProduct('${x.id}')">✕</button>
    </div></div>`).join(''):'<div class="empty">No products yet.</div>';
}
async function addProduct(){
  const b={affiliate_link:$('#af-link').value.trim(),twitter_uuid:$('#af-acc').value};
  if(!b.affiliate_link||!b.twitter_uuid)return toast('Link and a Twitter account are required',true);
  try{await api('/products',{method:'POST',body:JSON.stringify(b)});toast('Product added');$('#af-link').value='';loadAffiliate()}catch(e){toast(e.message,true)}
}
async function afRun(id){try{const j=await api('/actions/afm',{method:'POST',body:JSON.stringify({product_id:id})});toast('Pitch queued');openConsole(j.id)}catch(e){toast(e.message,true)}}
async function delProduct(id){if(!confirm('Delete this product?'))return;await api('/products/'+id,{method:'DELETE'});loadAffiliate()}

/* outreach */
async function runOutreach(){try{const j=await api('/actions/outreach',{method:'POST'});toast('Outreach queued');openConsole(j.id)}catch(e){toast(e.message,true)}}

/* schedule */
async function loadSchedule(){
  const list=await api('/schedule');
  await scKindChange();
  $('#sc-list').innerHTML=list.length?list.map(x=>`<div class="item">
    <div class="meta"><b>${x.kind}${x.account_id?' · '+x.account_id.slice(0,8):''}</b>
    <small>next: ${x.next_run?x.next_run.replace('T',' ').slice(0,16)+' UTC':'—'}</small></div>
    <div class="acts"><button class="btn icon" onclick="delSchedule('${x.id}')">✕</button></div></div>`).join(''):'<div class="empty">No schedules.</div>';
}
const setCron=v=>$('#sc-cron').value=v;
async function scKindChange(){
  const k=$('#sc-kind').value,w=$('#sc-target-wrap'),s=$('#sc-target');
  if(k==='outreach'){w.style.display='none';return}
  w.style.display='';
  let o=[];
  if(k==='youtube')o=(await api('/accounts/youtube')).map(a=>[a.id,a.nickname]);
  else if(k==='twitter')o=(await api('/accounts/twitter')).map(a=>[a.id,a.nickname]);
  else if(k==='afm')o=(await api('/products')).map(p=>[p.id,short(p.affiliate_link)]);
  s.innerHTML=o.map(([v,l])=>`<option value="${v}">${esc(l)}</option>`).join('')||'<option value="">(none available)</option>';
}
async function addSchedule(){
  const k=$('#sc-kind').value;
  try{await api('/schedule',{method:'POST',body:JSON.stringify({kind:k,cron:$('#sc-cron').value.trim(),
    account_id:k==='outreach'?'':$('#sc-target').value,upload:true})});toast('Schedule created');loadSchedule()}catch(e){toast(e.message,true)}
}
async function delSchedule(id){await api('/schedule/'+id,{method:'DELETE'});loadSchedule()}

/* settings */
let CFG={};
async function loadConfig(){
  CFG=await api('/config');
  $('#cfg-form').innerHTML=Object.entries(CFG).map(([k,v])=>{
    const id='cf_'+k;
    if(typeof v==='boolean')return `<div class="field"><label>${k}</label><select id="${id}"><option value="true"${v?' selected':''}>true</option><option value="false"${!v?' selected':''}>false</option></select></div>`;
    if(typeof v==='number')return `<div class="field"><label>${k}</label><input id="${id}" type="number" inputmode="numeric" value="${v}"></div>`;
    if(typeof v==='object'&&v!==null)return `<div class="field"><label>${k} <span style="color:var(--dim)">(json)</span></label><textarea id="${id}" autocapitalize="off" autocorrect="off" spellcheck="false">${esc(JSON.stringify(v,null,2))}</textarea></div>`;
    const sec=/key|password|token/i.test(k);
    return `<div class="field"><label>${k}</label><input id="${id}" type="${sec?'password':'text'}" autocapitalize="off" autocorrect="off" spellcheck="false" value="${esc(String(v))}"></div>`;
  }).join('');
}
async function saveConfig(){
  const out={};
  for(const [k,v] of Object.entries(CFG)){
    const el=$('#cf_'+k); if(!el){out[k]=v;continue}
    if(typeof v==='boolean')out[k]=el.value==='true';
    else if(typeof v==='number')out[k]=el.value===''?v:Number(el.value);
    else if(typeof v==='object'&&v!==null){try{out[k]=JSON.parse(el.value)}catch{return toast(`Invalid JSON in "${k}"`,true)}}
    else out[k]=el.value;
  }
  try{await api('/config',{method:'PUT',body:JSON.stringify(out)});CFG=out;toast('Config saved')}catch(e){toast(e.message,true)}
}

/* logs + console */
async function loadLogs(){
  const j=await api('/jobs?limit=60');
  $('#log-hint').textContent=j.length?j.length+' total':'';
  $('#log-list').innerHTML=j.length?j.map(jobRow).join(''):'<div class="empty">No runs yet.</div>';
}
let conTimer=null,conId=null;
function openConsole(id){
  conId=id;$('#sheet').classList.add('open');$('#con-log').textContent='connecting…';
  document.body.style.overflow='hidden';
  pollConsole();clearInterval(conTimer);conTimer=setInterval(pollConsole,1500);
}
function closeConsole(){$('#sheet').classList.remove('open');document.body.style.overflow='';clearInterval(conTimer);conId=null}
async function pollConsole(){
  if(!conId)return;
  try{
    const j=await api('/jobs/'+conId);
    $('#con-title').textContent=j.label;
    $('#con-status').textContent=j.status;$('#con-status').className='pill '+j.status;
    $('#con-dot').className='dot '+j.status;
    const pre=$('#con-log'),bottom=pre.scrollHeight-pre.scrollTop-pre.clientHeight<50;
    pre.textContent=j.log||'(no output yet)';
    if(bottom)pre.scrollTop=pre.scrollHeight;
    if(j.status==='done'||j.status==='error'){clearInterval(conTimer);if(current()==='overview')loadOverview()}
  }catch(e){$('#con-log').textContent='error: '+e.message;clearInterval(conTimer)}
}

/* modal */
let mCb=null;
function openModal(title,sub,ok,cb){
  $('#m-title').textContent=title;$('#m-sub').textContent=sub;$('#m-ok').textContent=ok;
  $('#m-input').value='';mCb=cb;$('#modal').classList.add('open');
}
function closeModal(){$('#modal').classList.remove('open');mCb=null}
$('#m-ok').onclick=()=>{const v=$('#m-input').value.trim();const cb=mCb;closeModal();if(cb)cb(v)};

async function delAcc(p,id,after){if(!confirm('Delete this account?'))return;await api(`/accounts/${p}/${id}`,{method:'DELETE'});after()}

loadOverview();
setInterval(()=>{if(current()==='overview'&&!document.hidden)loadOverview()},6000);
</script>
</body>
</html>
"""

LOGIN_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#0e1218">
<title>Sign in · MoneyPrinter</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root{--ink:#0e1218;--panel:#151b24;--line:#28303c;--text:#e2e7ee;--muted:#8b95a6;--dim:#5c6675;
        --mint:#4ea88b;--red:#d96a5a;--gold:#b0894f;
        --mono:'JetBrains Mono',ui-monospace,monospace;--sans:'Space Grotesk',system-ui,sans-serif}
  *{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
  body{margin:0;min-height:100dvh;background:var(--ink);color:var(--text);font-family:var(--sans);
    display:flex;align-items:center;justify-content:center;padding:24px;
    background-image:radial-gradient(circle at 50% 0,rgba(78,168,139,.08),transparent 55%)}
  .box{width:100%;max-width:360px}
  .brand{display:flex;align-items:center;gap:11px;justify-content:center;margin-bottom:26px}
  .mark{width:38px;height:38px;border-radius:9px;background:linear-gradient(150deg,var(--mint),#2f6f5c);
    display:grid;place-items:center;font-weight:700;color:#08120e;font-size:20px}
  .brand b{font-size:18px;font-weight:600;display:block;line-height:1.2}
  .brand span{font-family:var(--mono);font-size:10.5px;color:var(--dim);letter-spacing:1.5px;text-transform:uppercase}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:20px}
  h1{font-size:16px;margin:0 0 4px;font-weight:600}
  p.sub{margin:0 0 18px;color:var(--muted);font-size:13.5px;line-height:1.5}
  label{display:block;font-size:12.5px;color:var(--muted);margin-bottom:6px}
  input{width:100%;background:var(--ink);border:1px solid var(--line);color:var(--text);border-radius:10px;
    padding:13px;font-size:16px;font-family:var(--sans);outline:none;min-height:48px}
  input:focus{border-color:var(--mint);box-shadow:0 0 0 3px rgba(78,168,139,.13)}
  button{width:100%;margin-top:14px;min-height:48px;border:0;border-radius:10px;background:var(--mint);
    color:#08130f;font-weight:600;font-size:15px;font-family:var(--sans);cursor:pointer}
  button:active{transform:scale(.99)} button:disabled{opacity:.6}
  .err{display:none;margin-top:13px;background:rgba(217,106,90,.1);border:1px solid rgba(217,106,90,.3);
    color:#f0a99c;border-radius:9px;padding:11px 13px;font-size:13.5px;line-height:1.5}
  .foot{text-align:center;margin-top:16px;font-family:var(--mono);font-size:10.5px;color:var(--dim);line-height:1.6}
</style>
</head>
<body>
  <div class="box">
    <div class="brand"><div class="mark">$</div><div><b>MoneyPrinter</b><span>console</span></div></div>
    <div class="card">
      <h1>Sign in</h1>
      <p class="sub">This console can post as you, send email and spend API credits. Access is password protected.</p>
      <label for="pw">Password</label>
      <input id="pw" type="password" autocomplete="current-password" autocapitalize="off" autocorrect="off" spellcheck="false">
      <button id="go">Unlock</button>
      <div class="err" id="err"></div>
    </div>
    <div class="foot">Set via DASHBOARD_PASSWORD</div>
  </div>
<script>
  const pw=document.getElementById('pw'),go=document.getElementById('go'),err=document.getElementById('err');
  async function submit(){
    if(!pw.value)return;
    go.disabled=true;go.textContent='Checking…';err.style.display='none';
    try{
      const r=await fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({password:pw.value})});
      if(r.ok){location.href='/';return}
      let m='Incorrect password.';try{m=(await r.json()).detail||m}catch{}
      err.textContent=m;err.style.display='block';pw.value='';pw.focus();
    }catch(e){err.textContent='Network error. Try again.';err.style.display='block'}
    go.disabled=false;go.textContent='Unlock';
  }
  go.onclick=submit;
  pw.addEventListener('keydown',e=>{if(e.key==='Enter')submit()});
  pw.focus();
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
import hmac
import base64
import hashlib
import secrets
from fastapi import FastAPI, HTTPException, Body, Request, Response
from fastapi.responses import RedirectResponse, JSONResponse
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

# ---------------------------------------------------------------------------
# Authentication
#
# The dashboard exposes API keys, SMTP credentials and remote job execution, so
# it must never be reachable unauthenticated. Sessions are stateless: a signed,
# expiring token in an HttpOnly cookie (stdlib HMAC — no extra dependencies).
# ---------------------------------------------------------------------------
SESSION_COOKIE = "mp_session"
SESSION_DAYS = int(os.environ.get("SESSION_DAYS", "14"))
MAX_ATTEMPTS = 8            # failed logins per IP before a lockout
LOCKOUT_SECONDS = 900       # 15 minutes

_SECRET_FILE = os.path.join(MP_DIR, "session.key")


def _session_secret() -> bytes:
    """Persist a random signing key so sessions survive restarts."""
    if os.path.exists(_SECRET_FILE):
        data = open(_SECRET_FILE, "rb").read().strip()
        if data:
            return data
    key = base64.urlsafe_b64encode(secrets.token_bytes(32))
    with open(_SECRET_FILE, "wb") as f:
        f.write(key)
    try:
        os.chmod(_SECRET_FILE, 0o600)
    except OSError:
        pass
    return key


_SECRET = _session_secret()


def _get_password():
    """Password from env. If unset, generate one and print it to the logs."""
    pw = os.environ.get("DASHBOARD_PASSWORD", "").strip()
    if pw:
        return pw
    gen = os.path.join(MP_DIR, "generated_password.txt")
    if os.path.exists(gen):
        pw = open(gen).read().strip()
        if pw:
            return pw
    pw = secrets.token_urlsafe(12)
    with open(gen, "w") as f:
        f.write(pw)
    try:
        os.chmod(gen, 0o600)
    except OSError:
        pass
    print("=" * 62, file=_real_stdout)
    print("  No DASHBOARD_PASSWORD set. Generated a temporary password:", file=_real_stdout)
    print(f"  ->  {pw}", file=_real_stdout)
    print("  Set DASHBOARD_PASSWORD in your environment to control it.", file=_real_stdout)
    print("=" * 62, file=_real_stdout)
    return pw


_PASSWORD = _get_password()


def _sign(payload: str) -> str:
    sig = hmac.new(_SECRET, payload.encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(sig).decode().rstrip("=")


def _make_token() -> str:
    expires = int(time.time()) + SESSION_DAYS * 86400
    payload = f"{expires}.{secrets.token_urlsafe(8)}"
    return f"{payload}.{_sign(payload)}"


def _valid_token(token: str) -> bool:
    if not token or token.count(".") != 2:
        return False
    exp, nonce, sig = token.split(".")
    if not hmac.compare_digest(sig, _sign(f"{exp}.{nonce}")):
        return False
    try:
        return int(exp) > time.time()
    except ValueError:
        return False


# Simple in-memory brute-force throttle, keyed by client IP.
_attempts = {}


def _client_ip(request: "Request") -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _locked_out(ip: str):
    rec = _attempts.get(ip)
    if not rec:
        return 0
    count, first = rec
    if count < MAX_ATTEMPTS:
        return 0
    remaining = int(first + LOCKOUT_SECONDS - time.time())
    if remaining <= 0:
        _attempts.pop(ip, None)
        return 0
    return remaining


def _record_failure(ip: str):
    count, first = _attempts.get(ip, (0, time.time()))
    if time.time() - first > LOCKOUT_SECONDS:
        count, first = 0, time.time()
    _attempts[ip] = (count + 1, first)


app = FastAPI(title="MoneyPrinterV2 Dashboard")

# Paths reachable without a session.
_OPEN_PATHS = {"/login", "/api/login", "/api/health"}


@app.middleware("http")
async def require_auth(request: Request, call_next):
    path = request.url.path
    if path in _OPEN_PATHS:
        return await call_next(request)
    if _valid_token(request.cookies.get(SESSION_COOKIE, "")):
        return await call_next(request)
    if path.startswith("/api/"):
        return JSONResponse({"detail": "Not authenticated"}, status_code=401)
    return RedirectResponse("/login", status_code=302)


def _secure_cookie(request: Request) -> bool:
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    return proto == "https"


@app.get("/login", response_class=HTMLResponse)
def login_page():
    return LOGIN_HTML


@app.post("/api/login")
async def do_login(request: Request, body: dict = Body(...)):
    ip = _client_ip(request)
    wait = _locked_out(ip)
    if wait:
        raise HTTPException(429, f"Too many attempts. Try again in {wait // 60 + 1} min.")
    supplied = str(body.get("password", ""))
    if not hmac.compare_digest(supplied, _PASSWORD):
        _record_failure(ip)
        raise HTTPException(401, "Incorrect password.")
    _attempts.pop(ip, None)
    resp = JSONResponse({"ok": True})
    resp.set_cookie(SESSION_COOKIE, _make_token(), max_age=SESSION_DAYS * 86400,
                    httponly=True, samesite="lax", secure=_secure_cookie(request), path="/")
    return resp


@app.post("/api/logout")
def do_logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(SESSION_COOKIE, path="/")
    return resp


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