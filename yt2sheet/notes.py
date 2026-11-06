from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction


@dataclass
class Note:
    """A performed note, times in seconds (output of transcription)."""

    start: float
    end: float
    pitch: int  # MIDI pitch
    velocity: int
    graces: tuple = field(default=())  # pitches of grace notes played just before this note


@dataclass
class QNote:
    """A quantized note, positions measured in beats from the first barline."""

    start: Fraction
    end: Fraction
    pitch: int
    velocity: int
    graces: tuple = field(default=())
