"""Engrave MusicXML to PDF with LilyPond (preferred) or Verovio (fallback)."""

from __future__ import annotations

import io
import re
import shutil
import subprocess
import sys
from pathlib import Path

PAPER_MM = {"letter": (215.9, 279.4), "a4": (210.0, 297.0)}


def _lilypond_bin() -> Path | None:
    try:
        import lilypond  # PyPI package bundling LilyPond binaries

        return Path(lilypond.executable())
    except Exception:
        found = shutil.which("lilypond")
        return Path(found) if found else None


def _stack_lower_fingerings(src: str) -> str:
    """musicxml2ly writes a chord's fingerings in note order (lowest first). Above the staff
    LilyPond stacks them upwards, which is right; below the staff it stacks them downwards,
    putting the lowest note's finger on top. Reverse those so the stack mirrors the chord."""

    def fix(m: re.Match) -> str:
        digits = re.findall(r"_([1-5])", m.group(0))
        arp = "\\arpeggio " if "\\arpeggio" in m.group(0) else ""
        return arp + " ".join("_" + d for d in reversed(digits)) + " "

    return re.sub(r"(?:(?:\\arpeggio\s+)?_[1-5]\s+){2,}", fix, src)


def render_lilypond(musicxml: Path, pdf: Path, paper: str = "letter") -> Path:
    ly_bin = _lilypond_bin()
    if ly_bin is None:
        raise RuntimeError("LilyPond not found")
    musicxml2ly = ly_bin.parent / "musicxml2ly"
    if not musicxml2ly.exists():
        raise RuntimeError(f"musicxml2ly not found next to {ly_bin}")
    ly_file = pdf.with_suffix(".ly")
    # The pip "lilypond" package ships musicxml2ly as a shell wrapper around its own python;
    # a system install ships it as a plain python script.
    with open(musicxml2ly, "rb") as f:
        is_python = b"python" in f.readline()
    runner = [sys.executable] if is_python else []
    cmd = runner + [str(musicxml2ly), "--no-beaming", "--no-page-layout", "-o", str(ly_file), str(musicxml)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not ly_file.exists():
        raise RuntimeError(f"musicxml2ly failed:\n{proc.stderr[-2000:]}")

    src = ly_file.read_text()
    src = re.sub(r"(\\version\s+\"[^\"]+\")", r'\1\n#(set-default-paper-size "%s")' % paper, src, count=1)
    # music21 writes per-note stem directions; with LilyPond's own beaming these produce
    # crossed beams, so let LilyPond choose stems itself.
    src = re.sub(r"\\stem(Up|Down|Neutral)\s*", "", src)
    # musicxml2ly copies the title (MusicXML movement-title) into a subtitle; drop it, and the tagline.
    src = re.sub(r"^\s*subtitle\s*=.*\n", "", src, flags=re.M)
    src = re.sub(r"(\\header\s*\{)", r"\1\n    tagline = ##f", src, count=1)
    src = _stack_lower_fingerings(src)
    ly_file.write_text(src)

    out_base = pdf.with_suffix("")
    cmd = [str(ly_bin), "-dno-point-and-click", "-dbackend=cairo", "--pdf", "-o", str(out_base), str(ly_file)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not pdf.exists():
        # retry with the default backend in case cairo is unavailable in this build
        cmd = [str(ly_bin), "-dno-point-and-click", "--pdf", "-o", str(out_base), str(ly_file)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not pdf.exists():
        raise RuntimeError(f"lilypond failed:\n{proc.stderr[-3000:]}")
    return pdf


def render_verovio(musicxml: Path, pdf: Path, paper: str = "letter") -> Path:
    import cairosvg
    import verovio
    from pypdf import PdfReader, PdfWriter

    w_mm, h_mm = PAPER_MM[paper]
    tk = verovio.toolkit()
    tk.setOptions(
        {
            "pageWidth": int(w_mm * 10),
            "pageHeight": int(h_mm * 10),
            "pageMarginLeft": 120,
            "pageMarginRight": 120,
            "pageMarginTop": 100,
            "pageMarginBottom": 100,
            "scale": 40,
            "adjustPageHeight": False,
            "breaks": "auto",
            "footer": "none",
            "header": "auto",
        }
    )
    if not tk.loadFile(str(musicxml)):
        raise RuntimeError("Verovio could not load the MusicXML")
    writer = PdfWriter()
    for page in range(1, tk.getPageCount() + 1):
        svg = tk.renderToSVG(page)
        # Give the outer <svg> a physical size so the PDF page is the right paper size.
        svg = re.sub(r'(<svg[^>]*?)\swidth="[^"]+"', rf'\1 width="{w_mm}mm"', svg, count=1)
        svg = re.sub(r'(<svg[^>]*?)\sheight="[^"]+"', rf'\1 height="{h_mm}mm"', svg, count=1)
        page_pdf = cairosvg.svg2pdf(bytestring=svg.encode("utf-8"))
        for p in PdfReader(io.BytesIO(page_pdf)).pages:
            writer.add_page(p)
    with open(pdf, "wb") as f:
        writer.write(f)
    return pdf


def render_pdf(musicxml: Path, pdf: Path, engine: str = "auto", paper: str = "letter") -> tuple[Path, str]:
    """Render and return (pdf_path, engine_used)."""
    errors = []
    order = ["lilypond", "verovio"] if engine == "auto" else [engine]
    for eng in order:
        try:
            fn = render_lilypond if eng == "lilypond" else render_verovio
            return fn(musicxml, pdf, paper), eng
        except Exception as e:  # try the next engine
            errors.append(f"{eng}: {e}")
    raise RuntimeError("PDF rendering failed:\n" + "\n".join(errors))
