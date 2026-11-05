"""Make each hand's chords physically playable, and measure how playable a part is."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from fractions import Fraction

from .fingering import chord_cost, finger_events, is_black, step_cost, three_note_cost
from .notes import QNote

DEFAULT_REACH = 12  # widest chord (semitones) a hand plays at once: an octave


@dataclass
class Fixes:
    moved: int = 0  # notes handed to the other hand
    dropped: int = 0  # doubled notes removed
    rolled: set = field(default_factory=set)  # (hand, start) of chords to arpeggiate


def _span(notes: list[QNote]) -> int:
    return max(q.pitch for q in notes) - min(q.pitch for q in notes) if notes else 0


def _ok(notes: list[QNote], reach: int) -> bool:
    return len(notes) <= 5 and _span(notes) <= reach


def _drop_doubling(notes: list[QNote], keep_top: bool) -> QNote | None:
    """An inner note whose pitch class is doubled elsewhere in the chord (octave doubling).
    The melody (RH top) or bass (LH bottom) note is never chosen."""
    ordered = sorted(notes, key=lambda q: q.pitch)
    protected = ordered[-1] if keep_top else ordered[0]
    candidates = ordered if keep_top else ordered[::-1]  # drop from the inside out
    for q in candidates:
        if q is protected:
            continue
        if any(o is not q and o.pitch % 12 == q.pitch % 12 for o in notes):
            return q
    return None


def make_playable(
    rh: list[QNote], lh: list[QNote], reach: int = DEFAULT_REACH, drop_doublings: bool = True
) -> tuple[list[QNote], list[QNote], Fixes]:
    """Fix chords one hand cannot play: hand a note to the other hand if it fits there,
    else drop octave doublings, else mark the chord to be rolled (arpeggiated)."""
    fixes = Fixes()
    by_r: dict[Fraction, list[QNote]] = defaultdict(list)
    by_l: dict[Fraction, list[QNote]] = defaultdict(list)
    for q in rh:
        by_r[q.start].append(q)
    for q in lh:
        by_l[q.start].append(q)

    for start in sorted(set(by_r) | set(by_l)):
        R, L = by_r[start], by_l[start]
        # 1. give the RH's lowest / LH's highest note to the other hand when it fits there
        while not _ok(R, reach) and R:
            q = min(R, key=lambda n: n.pitch)
            if not _ok(L + [q], reach):
                break
            R.remove(q)
            L.append(q)
            fixes.moved += 1
        while not _ok(L, reach) and L:
            q = max(L, key=lambda n: n.pitch)
            if not _ok(R + [q], reach):
                break
            L.remove(q)
            R.append(q)
            fixes.moved += 1
        # 2. drop octave doublings, 3. otherwise roll the chord
        for hand, notes in (("R", R), ("L", L)):
            while drop_doublings and not _ok(notes, reach):
                q = _drop_doubling(notes, keep_top=hand == "R")
                if q is None:
                    break
                notes.remove(q)
                fixes.dropped += 1
            while len(notes) > 5:  # more notes than fingers: drop inner notes
                inner = sorted(notes, key=lambda n: n.pitch)[1:-1]
                notes.remove(inner[len(inner) // 2])
                fixes.dropped += 1
            if _span(notes) > reach:
                fixes.rolled.add((hand, start))

    new_rh = sorted((q for qs in by_r.values() for q in qs), key=lambda q: (q.start, q.pitch))
    new_lh = sorted((q for qs in by_l.values() for q in qs), key=lambda q: (q.start, q.pitch))
    return new_rh, new_lh, fixes


# --------------------------------------------------------------------------- measurement


def _events(notes: list[QNote]) -> list[tuple[float, list[int]]]:
    groups: dict[Fraction, list[int]] = defaultdict(list)
    for q in notes:
        groups[q.start].append(q.pitch)
    return [(float(s), sorted(set(ps))) for s, ps in sorted(groups.items())]


def fingering_difficulty(events, fingerings, hand: str) -> float:
    """Total fingering cost of a fingered part (same model the optimiser minimises)."""
    total, prev, prev2 = 0.0, None, None
    for (_, pitches), f in zip(events, fingerings):
        if f is None:
            prev = prev2 = None
            continue
        sign = -1 if hand == "L" else 1
        notes = sorted((sign * p, f[p], is_black(p)) for p in f)
        qs, fs, bs = (tuple(x) for x in zip(*notes))
        total += chord_cost(fs, qs, bs)
        if prev is not None:
            pairs = [step_cost(fa, qa, ba, fb, qb, bb) for qa, fa, ba in prev for qb, fb, bb in notes]
            total += sum(pairs) / len(pairs)
            if prev2 is not None and len(prev2) == len(prev) == len(notes) == 1:
                (q1, f1, _), (q2, f2, _), (q3, f3, _) = prev2[0], prev[0], notes[0]
                total += three_note_cost(f1, q1, f2, q2, f3, q3)
        prev2, prev = prev, notes
    return total


def report(rh: list[QNote], lh: list[QNote], reach: int = DEFAULT_REACH, seconds_per_beat: float = 0.6) -> dict:
    """Playability numbers for a two-hand part (lower is easier)."""
    out = {}
    for hand, notes in (("RH", rh), ("LH", lh)):
        ev = _events(notes)
        out[f"{hand} chords wider than reach"] = sum(1 for _, ps in ev if ps[-1] - ps[0] > reach)
        out[f"{hand} chords over 5 notes"] = sum(1 for _, ps in ev if len(ps) > 5)
        fing = finger_events(ev, hand[0], seconds_per_beat)
        n = sum(len(ps) for _, ps in ev) or 1
        out[f"{hand} fingering difficulty per note"] = round(fingering_difficulty(ev, fing, hand[0]) / n, 2)
    crossings = 0
    lh_by = defaultdict(list)
    for q in lh:
        lh_by[q.start].append(q.pitch)
    for q in rh:
        if q.start in lh_by and max(lh_by[q.start]) > q.pitch:
            crossings += 1
    out["hand crossings"] = crossings
    return out
