"""Fetch audio from YouTube (or any site yt-dlp supports)."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

URL_RE = re.compile(r"^https?://", re.I)


def is_url(s: str) -> bool:
    return bool(URL_RE.match(s))


def download_audio(url: str, out_dir: Path, cookies: str | None = None) -> tuple[Path, str]:
    """Download the best audio stream of `url` into `out_dir`.

    Returns (audio_path, title).
    """
    import yt_dlp

    out_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "wav"},
        ],
    }
    if cookies:
        opts["cookiefile"] = cookies
    # YouTube needs a JavaScript runtime (plus the yt-dlp-ejs package) to solve its
    # player challenges. yt-dlp only enables deno by default; enable whatever is installed.
    runtimes = {name: {} for name in ("deno", "node", "bun", "qjs") if shutil.which(name)}
    if runtimes:
        opts["js_runtimes"] = {("quickjs" if k == "qjs" else k): v for k, v in runtimes.items()}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if "entries" in info:  # a playlist slipped through; take the first entry
            info = info["entries"][0]
        path = Path(ydl.prepare_filename(info)).with_suffix(".wav")
    if not path.exists():
        raise RuntimeError(f"yt-dlp finished but {path} is missing")
    return path, info.get("title") or path.stem
