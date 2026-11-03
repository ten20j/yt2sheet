from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


@dataclass
class Note:
    """A performed note, times in seconds (output of transcription)."""

    start: float
    end: float
    pitch: int  # MIDI pitch
    velocity: int


@dataclass
class QNote:
    """A quantized note, positions measured in beats from the first barline."""

    start: Fraction
    end: Fraction
    pitch: int
    velocity: int
