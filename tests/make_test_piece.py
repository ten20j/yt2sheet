"""Render a ground-truth test piece to MIDI and (with FluidSynth) to audio, for an
end-to-end audio -> PDF check.

Usage: python tests/make_test_piece.py OUT_DIR [minuet|nocturne|waltz]
Then:  yt2sheet OUT_DIR/performed.wav --time-signature <meter of the piece>
"""

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pieces import PIECES, perform  # noqa: E402

from yt2sheet.transcribe import notes_to_midi  # noqa: E402

SOUNDFONT = Path("/usr/share/sounds/sf2/FluidR3_GM.sf2")  # apt install fluid-soundfont-gm

if __name__ == "__main__":
    out = Path(sys.argv[1])
    name = sys.argv[2] if len(sys.argv) > 2 else "minuet"
    out.mkdir(parents=True, exist_ok=True)
    piece = PIECES[name]
    notes_to_midi(perform(piece, seed=0, jitter=0.0, release=1.0, drift=0.0), str(out / "truth.mid"))
    notes_to_midi(perform(piece, seed=1), str(out / "performed.mid"))
    if shutil.which("fluidsynth") and SOUNDFONT.exists():
        subprocess.run(["fluidsynth", "-ni", "-q", "-g", "0.8", "-r", "44100", "-F", str(out / "performed.wav"),
                        str(SOUNDFONT), str(out / "performed.mid")], check=True)
    print(f"{name} ({piece[0]}):", *sorted(p.name for p in out.iterdir()))
