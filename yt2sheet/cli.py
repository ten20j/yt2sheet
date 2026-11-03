"""Command line entry point: yt2sheet <youtube-url | audio file | .mid> [options]"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
import time
from pathlib import Path

from . import __version__


def _slug(s: str) -> str:
    s = re.sub(r"[^\w\s-]", "", s).strip().lower()
    return re.sub(r"[\s_-]+", "-", s)[:80] or "score"


def _log(msg: str) -> None:
    print(f"[yt2sheet] {msg}", file=sys.stderr, flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="yt2sheet",
        description="Generate piano sheet music (PDF + MusicXML + MIDI) from a YouTube link or audio file.",
    )
    p.add_argument("source", help="YouTube URL, audio/video file (mp3, wav, m4a, ...), or a .mid file")
    p.add_argument("-o", "--out-dir", default="output", help="output directory (default: ./output)")
    p.add_argument("--title", help="score title (default: video title or file name)")
    p.add_argument("--start", type=float, help="start time in seconds")
    p.add_argument("--end", type=float, help="end time in seconds")
    p.add_argument("--time-signature", default="4/4", help="e.g. 4/4, 3/4, 6/8 (default 4/4)")
    p.add_argument("--bpm", type=float, help="tempo hint in beats per minute (fixes half/double tempo errors)")
    p.add_argument("--key", help="force the key, e.g. 'C', 'a' (minor), 'E-' (E-flat major)")
    p.add_argument("--grid", type=int, choices=[8, 16, 32], default=16, help="smallest note value (default 16th)")
    p.add_argument("--no-triplets", action="store_true", help="never use triplets")
    p.add_argument("--split-point", type=int, help="fixed hand split MIDI pitch (default: automatic)")
    p.add_argument("--downbeat", type=int, help="override which beat (0-based) of the first bar the music starts on")
    p.add_argument("--min-velocity", type=int, default=12, help="drop transcribed notes quieter than this")
    p.add_argument("--engine", choices=["auto", "lilypond", "verovio"], default="auto", help="PDF engraver")
    p.add_argument("--paper", choices=["letter", "a4"], default="letter")
    p.add_argument("--device", default="cpu", help="torch device for transcription (cpu, cuda, mps)")
    p.add_argument("--cookies", help="cookies.txt for YouTube (if it asks you to sign in)")
    p.add_argument("--version", action="version", version=f"yt2sheet {__version__}")
    return p


def run(args: argparse.Namespace, log=_log) -> dict[str, Path]:
    """Run the full pipeline. `log` receives progress messages."""
    from .download import download_audio, is_url
    from .rhythm import (
        Meter,
        align_to_barlines,
        choose_beat_octave,
        quantize,
        tempo_bpm,
        time_to_beat,
        track_beats,
    )

    meter = Meter.parse(args.time_signature)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    with tempfile.TemporaryDirectory(prefix="yt2sheet-") as tmp:
        source = args.source
        title = args.title
        if is_url(source):
            log(f"Downloading audio from {source} ...")
            try:
                audio_path, vid_title = download_audio(source, Path(tmp), cookies=args.cookies)
            except Exception as e:
                raise SystemExit(
                    f"Could not download {source}: {e}\n"
                    "Tips: update yt-dlp (pip install -U 'yt-dlp[default]'), install a JavaScript runtime "
                    "such as Deno or Node.js, or pass --cookies cookies.txt if YouTube asks you to sign in. "
                    "You can also download the audio yourself and pass the file instead."
                ) from None
            title = title or vid_title
        else:
            audio_path = Path(source)
            if not audio_path.exists():
                raise SystemExit(f"File not found: {audio_path}")
            title = title or audio_path.stem.replace("_", " ")

        base = out_dir / _slug(title)
        outputs: dict[str, Path] = {}

        if audio_path.suffix.lower() in (".mid", ".midi"):
            from .transcribe import midi_to_notes, midi_to_pedal

            log("Reading notes from MIDI (skipping transcription)")
            notes = midi_to_notes(str(audio_path))
            pedal = midi_to_pedal(str(audio_path))
            if args.start is not None or args.end is not None:
                lo, hi = args.start or 0.0, args.end or float("inf")
                notes = [n for n in notes if lo <= n.start < hi]
        else:
            from .audio import load_audio
            from .transcribe import SAMPLE_RATE, notes_to_midi, transcribe

            audio = load_audio(str(audio_path), sr=SAMPLE_RATE, channels=1, start=args.start, end=args.end)
            dur = len(audio) / SAMPLE_RATE
            log(f"Transcribing {dur:.0f}s of audio with Transkun on {args.device} (this is the slow part) ...")
            notes, pedal = transcribe(audio, device=args.device)
            outputs["transcription_midi"] = base.with_name(base.name + "-performance.mid")
            notes_to_midi(notes, str(outputs["transcription_midi"]), pedal)
            log(f"  {len(notes)} notes in {time.time() - t0:.0f}s")

    notes = [n for n in notes if n.velocity >= args.min_velocity and n.end - n.start > 0.02]
    if not notes:
        raise SystemExit("No notes were detected - is this piano music?")

    log("Tracking beats and quantizing rhythm ...")
    beats = track_beats(notes, bpm=args.bpm)
    if not args.bpm:
        beats = choose_beat_octave(notes, beats, meter)
    bpm = tempo_bpm(beats)
    qnotes = quantize(notes, beats, meter, smallest=args.grid, triplets=not args.no_triplets)
    qnotes, origin = align_to_barlines(qnotes, meter, shift=args.downbeat)
    pedal_beats = [
        (float(time_to_beat(a, beats)) - origin, float(time_to_beat(b, beats)) - origin) for a, b in pedal
    ]
    log(f"  tempo ~{bpm:.0f} BPM, {meter}")

    from .hands import split_hands
    from .render import render_pdf
    from .score import build_score

    rh, lh = split_hands(qnotes, split_point=args.split_point)
    score, key = build_score(rh, lh, meter, bpm, title, key_override=args.key, pedal=pedal_beats)
    log(f"  key: {key.tonic.name.replace('-', 'b')} {key.mode}; RH {len(rh)} notes, LH {len(lh)} notes")

    outputs["musicxml"] = base.with_suffix(".musicxml")
    score.write("musicxml", fp=str(outputs["musicxml"]))
    outputs["score_midi"] = base.with_name(base.name + "-score.mid")
    score.write("midi", fp=str(outputs["score_midi"]))

    log("Engraving PDF ...")
    pdf, engine = render_pdf(outputs["musicxml"], base.with_suffix(".pdf"), engine=args.engine, paper=args.paper)
    outputs["pdf"] = pdf
    log(f"Done in {time.time() - t0:.0f}s (engraved with {engine}).")
    return outputs


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    outputs = run(args)
    for kind, path in outputs.items():
        print(f"{kind:>20}: {path}")


if __name__ == "__main__":
    main()
