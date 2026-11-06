"""Recognise grace-note ornaments so they are engraved as grace notes, not as rhythm."""

from __future__ import annotations

import bisect
from dataclasses import replace

from .notes import Note

MAX_GRACE = 0.16  # seconds: a grace note is struck and left almost at once
MIN_GAP, MAX_GAP = 0.03, 0.22  # seconds from the grace note to its main note


def extract_graces(notes: list[Note]) -> tuple[list[Note], int]:
    """Find grace notes: a short note a step (1-2 semitones) from a clearly longer note
    struck right after it, as in Yiruma-style D-E ornaments. They are removed from the note
    list and recorded on their main note's `graces`. Fast scale runs are left alone (there
    the next note is no longer than the short one, and the short note follows another step).

    Returns (notes, number of grace notes found). The input notes are not modified.
    """
    notes = sorted((replace(n) for n in notes), key=lambda n: (n.start, n.pitch))
    starts = [n.start for n in notes]
    graces: set[int] = set()
    for i, n in enumerate(notes):
        dur = n.end - n.start
        if dur >= MAX_GRACE:
            continue
        lo = bisect.bisect_right(starts, n.start + MIN_GAP)
        hi = bisect.bisect_right(starts, n.start + MAX_GAP)
        targets = [
            m for m in notes[lo:hi]
            if id(m) not in graces and 1 <= abs(m.pitch - n.pitch) <= 2 and (m.end - m.start) >= 1.5 * dur
        ]
        if not targets:
            continue
        # part of a run rather than an ornament: a short step-wise note just before it
        before = notes[max(0, i - 8):i]
        if any(0 < n.start - p.start <= MAX_GAP and abs(p.pitch - n.pitch) <= 2 and p.end - p.start < MAX_GRACE
               for p in before):
            continue
        target = min(targets, key=lambda m: (abs(m.pitch - n.pitch), m.start))
        target.graces = target.graces + (n.pitch,)
        graces.add(id(n))
    return [n for n in notes if id(n) not in graces], len(graces)
