"""Minimal local web UI: paste a YouTube link (or upload audio), get piano sheet music.

Run:  yt2sheet-web   then open http://127.0.0.1:5000
"""

from __future__ import annotations

import argparse
import queue
import re
import threading
import traceback
import uuid
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file

from .cli import build_parser, run

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".webm", ".mp4", ".mkv", ".mid", ".midi"}
FILE_KINDS = {"pdf", "musicxml", "score_midi", "transcription_midi"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024
JOBS: dict[str, dict] = {}
WORK: queue.Queue = queue.Queue()
DATA_DIR = Path("yt2sheet-jobs")


def _worker() -> None:
    while True:
        job_id, argv = WORK.get()
        job = JOBS[job_id]
        job["status"] = "running"
        try:
            args = build_parser().parse_args(argv)
            outputs = run(args, log=job["log"].append)
            job["outputs"] = {k: str(v) for k, v in outputs.items()}
            job["status"] = "done"
        except SystemExit as e:
            job["status"], job["error"] = "error", str(e)
        except Exception as e:  # report, keep the worker alive
            job["status"], job["error"] = "error", f"{e}\n{traceback.format_exc(limit=3)}"


@app.post("/api/jobs")
def create_job():
    job_id = uuid.uuid4().hex[:12]
    job_dir = (DATA_DIR / job_id).resolve()
    job_dir.mkdir(parents=True)
    url = (request.form.get("url") or "").strip()
    upload = request.files.get("file")
    if upload and upload.filename:
        ext = Path(upload.filename).suffix.lower()
        if ext not in AUDIO_EXTS:
            return jsonify(error=f"Unsupported file type {ext}"), 400
        src = job_dir / f"input{ext}"
        upload.save(src)
        source = str(src)
        default_title = Path(upload.filename).stem
    elif re.match(r"^https?://", url):
        source, default_title = url, None
    else:
        return jsonify(error="Paste a YouTube link or choose an audio file."), 400

    argv = [source, "-o", str(job_dir)]
    title = (request.form.get("title") or "").strip() or default_title
    if title:
        argv += ["--title", title]
    for field, flag in (("time_signature", "--time-signature"), ("bpm", "--bpm"), ("start", "--start"),
                        ("end", "--end"), ("key", "--key"), ("paper", "--paper"),
                        ("fingering", "--fingering"), ("reach", "--reach")):
        val = (request.form.get(field) or "").strip()
        if val:
            argv += [flag, val]
    if request.form.get("no_triplets"):
        argv.append("--no-triplets")
    try:
        build_parser().parse_args(argv)  # validate before queueing
    except SystemExit:
        return jsonify(error="Invalid options."), 400

    JOBS[job_id] = {"status": "queued", "log": [], "outputs": {}, "error": None}
    WORK.put((job_id, argv))
    return jsonify(id=job_id)


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    job = JOBS.get(job_id) or abort(404)
    return jsonify(status=job["status"], log=job["log"], error=job["error"],
                   files=sorted(k for k in job["outputs"] if k in FILE_KINDS),
                   queue=WORK.qsize())


@app.get("/files/<job_id>/<kind>")
def job_file(job_id: str, kind: str):
    job = JOBS.get(job_id) or abort(404)
    if kind not in FILE_KINDS or kind not in job["outputs"]:
        abort(404)
    path = Path(job["outputs"][kind])
    return send_file(path, as_attachment=request.args.get("dl") == "1", download_name=path.name)


@app.get("/")
def index():
    return PAGE


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>yt2sheet</title>
<style>
:root{--bg:#f7f6f3;--card:#fff;--ink:#1d1c1a;--muted:#6b6862;--line:#e3e0da;--accent:#2f5d50;--err:#a23b2a}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#1f1f1d;--ink:#eceae5;--muted:#9c988f;--line:#33322f;--accent:#7fb8a4;--err:#e07a68}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:860px;margin:0 auto;padding:32px 16px}
h1{font-size:26px;margin:0 0 4px}p.sub{color:var(--muted);margin:0 0 24px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px;margin-bottom:16px}
label{display:block;font-size:13px;color:var(--muted);margin:0 0 4px}
input,select{width:100%;padding:10px 12px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--ink);font:inherit}
.row{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;margin-top:12px}
.or{color:var(--muted);font-size:13px;margin:10px 0}
details{margin-top:14px}summary{cursor:pointer;color:var(--muted);font-size:14px}
button,.btn{display:inline-block;margin-top:16px;padding:10px 18px;border:0;border-radius:8px;background:var(--accent);color:var(--bg);font:600 15px system-ui;cursor:pointer;text-decoration:none}
button:disabled{opacity:.5;cursor:default}.btn.alt{background:transparent;color:var(--accent);border:1px solid var(--accent);margin-right:8px}
pre{white-space:pre-wrap;font:13px/1.5 ui-monospace,monospace;color:var(--muted);margin:0}
.err{color:var(--err)}iframe{width:100%;height:80vh;border:1px solid var(--line);border-radius:8px;margin-top:16px;background:#fff}
.check{display:flex;gap:8px;align-items:center;margin-top:12px}.check input{width:auto}
</style></head><body><main>
<h1>yt2sheet</h1><p class="sub">Piano sheet music from a YouTube link or an audio file.</p>
<form class="card" id="f">
  <label for="url">YouTube link</label><input id="url" name="url" placeholder="https://www.youtube.com/watch?v=...">
  <div class="or">or upload audio (mp3, wav, m4a&hellip;) or MIDI</div><input type="file" name="file" accept="audio/*,video/*,.mid,.midi">
  <details><summary>Options</summary>
    <div class="row">
      <div><label>Title</label><input name="title" placeholder="from video"></div>
      <div><label>Time signature</label><select name="time_signature"><option>4/4</option><option>3/4</option><option>2/4</option><option>6/8</option><option>12/8</option><option>2/2</option></select></div>
      <div><label>Tempo hint (BPM)</label><input name="bpm" type="number" min="30" max="300" placeholder="auto"></div>
      <div><label>Key</label><input name="key" placeholder="auto (e.g. E-, f#)"></div>
      <div><label>Start (s)</label><input name="start" type="number" min="0" step="0.1"></div>
      <div><label>End (s)</label><input name="end" type="number" min="0" step="0.1"></div>
      <div><label>Paper</label><select name="paper"><option>letter</option><option>a4</option></select></div>
      <div><label>Fingering</label><select name="fingering"><option value="smart">where the hand moves</option><option value="all">every note</option><option value="none">none</option></select></div>
      <div><label>Hand reach</label><select name="reach"><option value="12">octave (12)</option><option value="14">9th (14)</option><option value="16">10th (16)</option><option value="10">small (10)</option></select></div>
    </div>
    <label class="check"><input type="checkbox" name="no_triplets" value="1"> Never use triplets</label>
  </details>
  <button id="go">Make sheet music</button>
</form>
<div class="card" id="status" hidden><pre id="log"></pre><div id="links"></div></div>
<script>
const f=document.getElementById('f'),go=document.getElementById('go'),st=document.getElementById('status'),
      log=document.getElementById('log'),links=document.getElementById('links');
const names={pdf:'PDF',musicxml:'MusicXML (edit in MuseScore)',score_midi:'MIDI (score)',transcription_midi:'MIDI (performance)'};
f.onsubmit=async e=>{e.preventDefault();go.disabled=true;st.hidden=false;links.innerHTML='';log.className='';log.textContent='Submitting…';
  const r=await fetch('/api/jobs',{method:'POST',body:new FormData(f)});const j=await r.json();
  if(!r.ok){log.className='err';log.textContent=j.error;go.disabled=false;return}poll(j.id)};
async function poll(id){const r=await fetch('/api/jobs/'+id);const j=await r.json();
  log.textContent=(j.status==='queued'?'Waiting in queue…\\n':'')+j.log.join('\\n');
  if(j.status==='error'){log.className='err';log.textContent+='\\n'+j.error;go.disabled=false;return}
  if(j.status!=='done'){setTimeout(()=>poll(id),1500);return}
  go.disabled=false;
  links.innerHTML=j.files.map(k=>`<a class="btn alt" href="/files/${id}/${k}?dl=1">${names[k]||k}</a>`).join('')+
    (j.files.includes('pdf')?`<iframe src="/files/${id}/pdf"></iframe>`:'')}
</script></main></body></html>"""


def main(argv: list[str] | None = None) -> None:
    global DATA_DIR
    p = argparse.ArgumentParser(prog="yt2sheet-web", description="Local web UI for yt2sheet")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--data-dir", default="yt2sheet-jobs", help="where uploads and results are stored")
    a = p.parse_args(argv)
    DATA_DIR = Path(a.data_dir)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=_worker, daemon=True).start()
    print(f"yt2sheet web UI on http://{a.host}:{a.port}")
    app.run(host=a.host, port=a.port, threaded=True)


if __name__ == "__main__":
    main()
