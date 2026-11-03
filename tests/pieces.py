"""Small ground-truth pieces used to test the notation half of the pipeline.

Each piece is (meter, bpm, rh, lh) where a hand is a list of (pitch | chord tuple | None, beats).
"""

from __future__ import annotations

import random
from fractions import Fraction as F

from yt2sheet.notes import Note


def _alberti(a, b, c):
    return [(a, F(1, 2)), (c, F(1, 2)), (b, F(1, 2)), (c, F(1, 2))] * 2


# D major, 4/4: melody with dotted rhythms over an Alberti bass (no triplets)
MINUET = (
    "4/4",
    96,
    [
        (69, 1), (74, 1), (78, 1), (76, F(1, 2)), (74, F(1, 2)),
        (76, F(3, 2)), (78, F(1, 2)), (76, 2),
        (74, F(1, 2)), (73, F(1, 2)), (71, F(1, 2)), (73, F(1, 2)), (74, 1), (78, 1),
        (81, 3), (None, 1),
        (79, 1), (78, 1), (76, 1), (74, 1),
        (73, F(1, 2)), (74, F(1, 2)), (76, 1), (69, 2),
        (71, 1), (79, 1), (78, F(3, 4)), (76, F(1, 4)), (74, 1),
        (74, 4),
    ],
    _alberti(50, 54, 57) + _alberti(45, 49, 52)
    + [(47, F(1, 2)), (54, F(1, 2)), (50, F(1, 2)), (54, F(1, 2)), (50, F(1, 2)), (57, F(1, 2)), (54, F(1, 2)), (57, F(1, 2))]
    + [((45, 52, 57), 2), (45, 2), (43, 2), (47, 2), (45, 1), (52, 1), (45, 2), (43, 2), (45, 2), ((38, 45, 50), 4)],
)

# C# minor, 4/4: RH triplet-8th arpeggios over LH octaves (Moonlight-style texture), melody on top
T = F(1, 3)
NOCTURNE = (
    "4/4",
    66,
    [(56, T), (61, T), (64, T)] * 4
    + [(57, T), (61, T), (64, T)] * 2 + [(57, T), (62, T), (66, T)] * 2
    + [(56, T), (60, T), (66, T)] * 2 + [(56, T), (61, T), (64, T)] + [(54, T), (61, T), (63, T)]
    + [((60, 68), 2), ((60, 66, 68), 2)] + [((56, 61, 64), 4)],
    [((37, 49), 4), ((35, 47), 4), ((33, 45), 2), ((30, 42), 2), ((32, 44), 2), ((32, 44), 2), ((25, 37), 4)],
)

# G major waltz, 3/4: "oom-pah-pah" bass, melody with a pickup-free start
WALTZ = (
    "3/4",
    138,
    [
        (74, 2), (71, 1), (67, 1), (71, 1), (74, 1), (79, 3), (78, 2), (76, 1),
        (74, 1), (72, F(1, 2)), (71, F(1, 2)), (69, 1), (67, 3),
        (71, 2), (69, 1), (67, 1), (66, 1), (64, 1), (62, 3),
    ],
    [
        (43, 1), ((59, 62), 1), ((59, 62), 1), (43, 1), ((59, 62), 1), ((59, 62), 1),
        (47, 1), ((59, 62), 1), ((59, 62), 1), (50, 1), ((57, 60), 1), ((57, 60), 1),
        (50, 1), ((54, 60), 1), ((54, 60), 1), (43, 1), ((59, 62), 1), ((59, 62), 1),
        (40, 1), ((55, 59), 1), ((55, 59), 1), (38, 1), ((54, 57), 1), ((54, 57), 1),
        ((38, 50), 3),
    ],
)

# A minor, 4/4: continuous single-note arpeggios at a fast pulse. Written as 8ths at
# quarter = 72 (the tracker hears the 8th-note pulse at ~144 as the beat).
H = F(1, 2)
BALLAD = (
    "4/4",
    72,
    [(None, 4)] * 4,
    [(45, H), (52, H), (57, H), (60, H), (64, H), (60, H), (57, H), (52, H)]
    + [(41, H), (48, H), (53, H), (57, H), (60, H), (57, H), (53, H), (48, H)]
    + [(36, H), (43, H), (48, H), (52, H), (55, H), (52, H), (48, H), (43, H)]
    + [(43, H), (50, H), (55, H), (59, H), (62, H), (59, H), (55, H), (50, H)],
)

PIECES = {"minuet": MINUET, "nocturne": NOCTURNE, "waltz": WALTZ}


def truth_notes(seq) -> list[tuple[int, F, F]]:
    """(pitch, start_beat, end_beat) in beats of the piece's beat unit (quarter here)."""
    out, t = [], F(0)
    for p, d in seq:
        d = F(d)
        if p is not None:
            for q in p if isinstance(p, tuple) else (p,):
                out.append((q, t, t + d))
        t += d
    return out


def perform(piece, seed: int = 0, jitter: float = 0.012, release: float = 0.9, drift: float = 0.04) -> list[Note]:
    """Render a piece as a human-ish performance: timing jitter, early releases, tempo drift."""
    _meter, bpm, rh, lh = piece
    rng = random.Random(seed)
    spb = 60.0 / bpm
    total = float(max(e for _, _, e in truth_notes(rh) + truth_notes(lh)))
    notes = []
    for seq, vel in ((rh, 80), (lh, 60)):
        for p, s, e in truth_notes(seq):
            stretch = lambda b: b * spb * (1 + drift * float(b) / total)
            t0 = stretch(float(s)) + rng.gauss(0, jitter) + 0.5
            t1 = stretch(float(s) + float(e - s) * release) + rng.gauss(0, jitter * 1.5) + 0.5
            v = max(25, min(115, int(rng.gauss(vel, 8))))
            notes.append(Note(max(0.0, t0), max(t0 + 0.03, t1), p, v))
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes
