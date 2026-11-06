"""Beat tracking and rhythmic quantization of performed notes."""

from __future__ import annotations

import functools
import math
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .notes import Note, QNote

FPS = 100  # frame rate of the synthetic onset envelope used for beat tracking


@dataclass(frozen=True)
class Meter:
    numerator: int
    denominator: int

    @classmethod
    def parse(cls, s: str) -> "Meter":
        try:
            n, d = s.split("/")
            m = cls(int(n), int(d))
        except ValueError:
            raise ValueError(f"Bad time signature {s!r}; expected e.g. 4/4, 3/4, 6/8") from None
        if m.numerator <= 0 or m.denominator not in (2, 4, 8, 16):
            raise ValueError(f"Unsupported time signature {s!r}")
        return m

    @property
    def compound(self) -> bool:
        return self.denominator == 8 and self.numerator % 3 == 0 and self.numerator >= 6

    @property
    def beats_per_bar(self) -> int:
        return self.numerator // 3 if self.compound else self.numerator

    @property
    def beat_ql(self) -> Fraction:
        """Length of one beat in quarter notes."""
        return Fraction(3, 2) if self.compound else Fraction(4, self.denominator)

    def __str__(self) -> str:
        return f"{self.numerator}/{self.denominator}"


# --------------------------------------------------------------------------- accents


def local_bass(times, pitches, window: float) -> np.ndarray:
    """True for notes that are the lowest sounding pitch within +/- `window` of their onset
    (the bass line), which is a much better downbeat cue than an absolute register."""
    t = np.asarray(times, dtype=float)
    p = np.asarray(pitches)
    order = np.argsort(t)
    ts, ps = t[order], p[order]
    lo = np.searchsorted(ts, ts - window)
    hi = np.searchsorted(ts, ts + window, side="right")
    is_bass = np.array([ps[i] <= ps[a:b].min() for i, (a, b) in enumerate(zip(lo, hi))])
    out = np.empty_like(is_bass)
    out[order] = is_bass
    return out


def accent_weights(times, pitches, velocities, window: float) -> np.ndarray:
    v = np.asarray(velocities, dtype=float) / 127.0
    return v * (1.0 + 1.5 * local_bass(times, pitches, window))


# --------------------------------------------------------------------------- beat tracking


def _onset_envelope(notes: list[Note], fps: int = FPS) -> np.ndarray:
    length = int((max(n.end for n in notes) + 2.0) * fps) + 1
    env = np.zeros(length)
    w = accent_weights([n.start for n in notes], [n.pitch for n in notes], [n.velocity for n in notes], 0.6)
    for n, weight in zip(notes, w):
        env[int(round(n.start * fps))] += weight
    # Collapse chords so a 6-note chord is not 6x as "beat-like" as a single note.
    env = np.sqrt(env)
    k = np.exp(-0.5 * (np.arange(-6, 7) / 1.5) ** 2)
    return np.convolve(env, k / k.sum(), mode="same")


def track_beats(notes: list[Note], bpm: float | None = None) -> np.ndarray:
    """Return beat times (seconds) covering every note, extrapolated at both ends."""
    import librosa

    env = _onset_envelope(notes)
    kwargs = dict(onset_envelope=env, sr=FPS, hop_length=1, units="time", trim=False)
    if bpm:
        kwargs["bpm"] = float(bpm)
    _tempo, beats = librosa.beat.beat_track(**kwargs)
    beats = np.asarray(beats, dtype=float)

    first = min(n.start for n in notes)
    last = max(n.end for n in notes)
    if len(beats) < 4:  # degenerate: fall back to a constant grid
        ibi = 60.0 / (bpm or float(np.atleast_1d(_tempo)[0]) or 100.0)
        return np.arange(first, last + ibi, ibi)

    # Drop beats far outside the notes, then extrapolate to cover them fully.
    beats = beats[(beats > first - 1.0) & (beats < last + 1.0)]
    ibi_head = float(np.median(np.diff(beats[:6])))
    ibi_tail = float(np.median(np.diff(beats[-6:])))
    head = []
    t = beats[0] - ibi_head
    while t > first - ibi_head:
        head.append(t)
        t -= ibi_head
    tail = []
    t = beats[-1] + ibi_tail
    while t < last + ibi_tail:
        tail.append(t)
        t += ibi_tail
    beats = np.concatenate([np.array(head[::-1]), beats, np.array(tail)])
    if not bpm:  # a user-supplied tempo is trusted as is
        beats, _factor = refine_beat_level(notes, beats)
    return beats


def _pitch_recurrence(notes: list[Note], pos: np.ndarray, lag: float, tol: float = 0.1) -> float:
    """Fraction of notes whose pitch is struck again `lag` beats later.

    Accompaniment figures (arpeggios, Alberti bass, oom-pah) repeat once per beat or bar,
    so this peaks at the true beat length even when every onset is equally loud.
    """
    by_pitch: dict[int, list[float]] = defaultdict(list)
    for n, x in zip(notes, pos):
        by_pitch[n.pitch].append(float(x))
    hits = 0
    for xs in by_pitch.values():
        arr = np.sort(np.array(xs))
        targets = arr + lag
        j = np.clip(np.searchsorted(arr, targets), 1, len(arr) - 1) if len(arr) > 1 else None
        if j is None:
            continue
        nearest = np.minimum(np.abs(arr[j] - targets), np.abs(arr[j - 1] - targets))
        hits += int(np.sum(nearest < tol))
    return hits / len(notes)


def refine_beat_level(notes: list[Note], beats: np.ndarray) -> tuple[np.ndarray, float]:
    """Fix 2-vs-3 grouping errors of the beat tracker (e.g. it locked onto every 2nd
    triplet 8th). Returns (beats, factor) where factor is new beat length in old beats."""
    pos = time_to_beat(np.array([n.start for n in notes]), beats)

    def score(f: float) -> float:
        return _pitch_recurrence(notes, pos, f) + _pitch_recurrence(notes, pos, 2 * f)

    base = score(1.0)
    best_f, best = 1.0, base
    for f in (1.5, 2.0 / 3.0):
        s = score(f)
        if s > 1.5 * base + 0.05 and s > best:
            best_f, best = f, s
    if best_f == 1.0:
        return beats, 1.0

    # Choose the phase of the new beat grid: strongest (accented, low, long) onsets on the beat.
    sub = 0.5 if best_f == 1.5 else 1.0 / 3.0
    weights = accent_weights(pos, [n.pitch for n in notes], [n.velocity for n in notes], best_f)
    phases = np.arange(0, best_f - 1e-9, sub)
    strengths = []
    for ph in phases:
        d = (pos - ph) / best_f
        strengths.append(float(np.sum(weights[np.abs(d - np.round(d)) * best_f < 0.1])))
    ph = float(phases[int(np.argmax(strengths))])
    idx = np.arange(ph - best_f * math.ceil(ph / best_f), len(beats) - 1 + best_f, best_f)
    old = np.arange(len(beats))
    new = np.interp(idx, old, beats, left=np.nan, right=np.nan)
    # linear extrapolation at the ends
    lo, hi = idx < 0, idx > len(beats) - 1
    new[lo] = beats[0] + idx[lo] * (beats[1] - beats[0])
    new[hi] = beats[-1] + (idx[hi] - (len(beats) - 1)) * (beats[-1] - beats[-2])
    return new, best_f


def choose_beat_octave(notes: list[Note], beats: np.ndarray, meter: "Meter") -> np.ndarray:
    """Notate continuous fast motion as 8ths at half tempo instead of quarters.

    If almost every onset lands on a beat at a fast tempo, the tracker has probably
    taken the 8th-note pulse as the beat (common for flowing arpeggio accompaniments).
    Only applied in simple meters with an even number of beats per bar, so a 3/4 waltz
    at quarter = 140 is left alone.
    """
    if meter.compound or meter.beats_per_bar % 2 or tempo_bpm(beats) <= 110:
        return beats
    starts = {q.start for q in quantize(notes, beats, meter)}
    offbeat = sum(s.denominator != 1 for s in starts) / max(1, len(starts))
    if offbeat >= 0.10:
        return beats
    # keep the half-tempo grid whose beats carry the stronger (accented, low) onsets
    pos = time_to_beat(np.array([n.start for n in notes]), beats)
    w = accent_weights(pos, [n.pitch for n in notes], [n.velocity for n in notes], 2.0)
    on_beat = np.abs(pos - np.round(pos)) < 0.15
    even = float(np.sum(w[on_beat & (np.round(pos) % 2 == 0)]))
    odd = float(np.sum(w[on_beat & (np.round(pos) % 2 == 1)]))
    return beats[0::2] if even >= odd else beats[1::2]


def time_to_beat(t: np.ndarray | float, beats: np.ndarray) -> np.ndarray:
    """Piecewise-linear map from seconds to (fractional) beat index."""
    t = np.asarray(t, dtype=float)
    idx = np.clip(np.searchsorted(beats, t), 1, len(beats) - 1)
    b0, b1 = beats[idx - 1], beats[idx]
    return (idx - 1) + (t - b0) / (b1 - b0)


def tempo_bpm(beats: np.ndarray) -> float:
    return 60.0 / float(np.median(np.diff(beats)))


# --------------------------------------------------------------------------- grids


@dataclass(frozen=True)
class Grid:
    """Grid points within one beat, with a metrical 'level' per point (0 = on the beat)."""

    divisions: tuple[int, ...]  # e.g. (2, 2) = 16ths in simple time, (3,) = triplet 8ths

    @property
    def n(self) -> int:
        return math.prod(self.divisions)

    @property
    def is_triplet(self) -> bool:
        return self.divisions[:1] == (3,)

    def points(self) -> list[tuple[Fraction, int]]:
        return _grid_points(self.divisions)


@functools.cache
def _grid_points(divisions: tuple[int, ...]) -> list[tuple[Fraction, int]]:
    n = math.prod(divisions)
    pts = []
    for k in range(n + 1):
        level, prod = 0, 1
        while (k * prod) % n:
            prod *= divisions[level]
            level += 1
        pts.append((Fraction(k, n), level))
    return pts


# Cost (in beats of timing error) of writing a note at each metrical level: 0 = on the beat,
# 1 = 8th, 2 = 16th, ... In "simple" style a note with no partner a 16th away (no run, no
# dotted-rhythm partner on the next beat) is expensive to write as a 16th, so expressive
# timing (rubato, a note pushed a little early) snaps to the beat or the 8th. Notes that do
# have such a partner are snapped as played. "detailed" writes what was played everywhere.
LEVEL_COSTS = {
    "simple": (0.0, 0.03, 0.25, 0.5, 0.75),
    "detailed": (0.0, 0.03, 0.06, 0.09, 0.12),
}
PARTNER = (0.75, 1.25)  # partner distance, in grid steps (16ths), that makes a 16th believable
TRIPLET_PENALTY = {"simple": 0.1, "detailed": 0.1}  # extra cost of switching a beat to triplets


def _snap(local: float, grid: Grid, costs: tuple = LEVEL_COSTS["detailed"]) -> tuple[Fraction, float]:
    best, best_cost = Fraction(0), float("inf")
    for p, level in grid.points():
        cost = abs(local - float(p)) + costs[min(level, len(costs) - 1)]
        if cost < best_cost:
            best, best_cost = p, cost
    return best, best_cost


def grids_for(meter: Meter, smallest: int, triplets: bool) -> tuple[Grid, Grid | None]:
    """Return (default grid, optional triplet alternative) for a meter.

    `smallest` is the smallest note value (8, 16 or 32).
    """
    beat_value = meter.denominator
    if meter.compound:
        # beat = dotted quarter = three 8ths, then binary below that
        depth = max(0, int(math.log2(smallest / 8)))
        return Grid((3,) + (2,) * depth), None
    depth = max(0, int(math.log2(smallest / beat_value)))
    base = Grid((2,) * depth if depth else (1,))
    alt = Grid((3,)) if triplets else None
    return base, alt


# --------------------------------------------------------------------------- quantization


def _cluster_onsets(notes: list[Note], window: float = 0.035) -> list[float]:
    """Make near-simultaneous onsets (chords, slight rolls) share one onset time."""
    onsets = [n.start for n in notes]
    order = sorted(range(len(notes)), key=lambda i: onsets[i])
    out = list(onsets)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and onsets[order[j + 1]] - onsets[order[i]] <= window:
            j += 1
        mean = float(np.mean([onsets[order[k]] for k in range(i, j + 1)]))
        for k in range(i, j + 1):
            out[order[k]] = mean
        i = j + 1
    return out


def quantize(
    notes: list[Note],
    beats: np.ndarray,
    meter: Meter,
    smallest: int = 16,
    triplets: bool = True,
    style: str = "simple",
) -> list[QNote]:
    """Snap notes to a metrical grid. Positions are in beats relative to the first barline.

    style "simple" favours plain rhythms over literal timing (see LEVEL_COSTS); "detailed"
    keeps every 16th that was played.
    """
    if not notes:
        return []
    base, alt = grids_for(meter, smallest, triplets)
    costs, detailed = LEVEL_COSTS[style], LEVEL_COSTS["detailed"]
    tri_penalty = TRIPLET_PENALTY[style]

    onsets = time_to_beat(np.array(_cluster_onsets(notes)), beats)
    ends = time_to_beat(np.array([n.end for n in notes]), beats)

    # Assign onsets to beats (with a little look-behind tolerance) and choose a grid per beat.
    tol = 0.125
    beat_of = np.floor(onsets + tol).astype(int)
    by_beat: dict[int, list[int]] = defaultdict(list)
    for i, b in enumerate(beat_of):
        by_beat[int(b)].append(i)

    # In simple style a note that would be written on a 16th keeps that rhythm only if it has
    # a partner a 16th away, towards the nearest beat and in the same register: the beat it
    # leads into (dotted 8th + 16th), the beat it follows (16th after the beat), or a run.
    # A lone note pushed off the beat by expressive timing has no such partner.
    cluster_pos = sorted({round(float(x), 3) for x in onsets})
    pitches_at: dict[float, list[int]] = defaultdict(list)
    for i, n in enumerate(notes):
        pitches_at[round(float(onsets[i]), 3)].append(n.pitch)
    partnered: set[float] = set()
    if style == "simple":
        arr = np.array(cluster_pos)
        step = 1.0 / base.n
        lo_gap, hi_gap = PARTNER[0] * step, PARTNER[1] * step
        level_of = dict(base.points())
        for x in cluster_pos:
            b = math.floor(x + 0.125)
            p = _snap(x - b, base, detailed)[0]
            if level_of.get(p, 0) < 2:
                continue  # beat or 8th: no partner needed
            directions = [1] if p > Fraction(1, 2) else [-1] if p < Fraction(1, 2) else [1, -1]
            for d in directions:
                lo, hi = np.searchsorted(arr, x + d * hi_gap if d < 0 else x + lo_gap), \
                    np.searchsorted(arr, x - lo_gap if d < 0 else x + hi_gap, side="right")
                if any(_same_register(pitches_at[x], pitches_at[float(y)]) for y in arr[lo:hi]):
                    partnered.add(x)
                    break

    def cost_of(x_abs: float) -> tuple:
        return detailed if round(x_abs, 3) in partnered else costs

    grid_of_beat: dict[int, Grid] = {}
    costs_of_beat: dict[int, tuple] = {}
    for b, idxs in by_beat.items():
        locals_ = sorted({round(float(onsets[i] - b), 3) for i in idxs})
        # Triplets or not is decided on the timing as played; simplification only applies
        # within the chosen grid (otherwise penalising 16ths would make triplets win instead).
        cost_base = sum(_snap(x, base, detailed)[1] for x in locals_)
        grid_of_beat[b], costs_of_beat[b] = base, costs
        if alt is not None:
            snapped_alt = {_snap(x, alt, detailed)[0] for x in locals_}
            off_beat = [p for p in snapped_alt if 0 < p < 1]
            enough = len(off_beat) >= 2 or (style == "detailed" and len(off_beat) == 1 and len(locals_) >= 2)
            if enough:
                cost_alt = sum(_snap(x, alt, detailed)[1] for x in locals_) + tri_penalty
                if cost_alt < cost_base:
                    grid_of_beat[b] = alt
        # Simplifying must not merge two onsets in the same register (one hand) into a chord:
        # if it would, write this beat as played. Melody and bass a little apart may merge.
        g = grid_of_beat[b]
        landed: dict[Fraction, list[float]] = defaultdict(list)
        for x in locals_:
            landed[_snap(x, g, cost_of(b + x))[0]].append(x)
        for xs in landed.values():
            for x, y in zip(xs, xs[1:]):
                if _snap(x, g, detailed)[0] != _snap(y, g, detailed)[0] and _same_register(
                    pitches_at[round(b + x, 3)], pitches_at[round(b + y, 3)]
                ):
                    costs_of_beat[b] = detailed

    out: list[QNote] = []
    for i, n in enumerate(notes):
        b = int(beat_of[i])
        g = grid_of_beat[b]
        x_abs = float(onsets[i])
        c = detailed if costs_of_beat[b] is detailed else cost_of(x_abs)
        start = b + _snap(x_abs - b, g, c)[0]
        end = _snap_end(start, float(ends[i]), grid_of_beat, base, 3 if meter.compound else 2, costs)
        if end <= start:
            end = start + Fraction(1, g.n)
        out.append(QNote(start, end, n.pitch, n.velocity, n.graces))
    return _dedupe(out)


def _same_register(a: list[int], b: list[int]) -> bool:
    """Could one hand play both groups of notes (nearest notes within an octave)?"""
    return bool(a) and bool(b) and min(abs(p - q) for p in a for q in b) <= 12


def _snap_end(start: Fraction, end: float, grid_of_beat: dict[int, Grid], base: Grid, sub: int,
              costs: tuple = LEVEL_COSTS["detailed"]) -> Fraction:
    """Quantize a note end. Key releases are imprecise and usually early, so longer notes
    snap to coarser positions with an upward bias (a held dotted half released at 2.7 beats
    should still be a dotted half)."""
    dur = end - float(start)
    if dur >= 1.75:
        return Fraction(math.floor(end + 0.5 + 0.2))  # nearest beat, biased up
    be = int(math.floor(end + 0.125))
    ge = grid_of_beat.get(be, base)
    if dur >= 0.75:  # nearest beat subdivision (8th or triplet 8th), biased up
        s = 3 if ge.is_triplet else sub
        return Fraction(math.floor((end + 0.1) * s + 0.5), s)
    return be + _snap(end - be, ge, costs)[0]


def _dedupe(qnotes: list[QNote]) -> list[QNote]:
    """Merge duplicate (pitch, start) notes and prevent same-pitch overlaps."""
    by_key: dict[tuple[int, Fraction], QNote] = {}
    for q in qnotes:
        k = (q.pitch, q.start)
        if k in by_key:
            prev = by_key[k]
            prev.end = max(prev.end, q.end)
            prev.velocity = max(prev.velocity, q.velocity)
            prev.graces = prev.graces + tuple(g for g in q.graces if g not in prev.graces)
        else:
            by_key[k] = QNote(q.start, q.end, q.pitch, q.velocity, q.graces)
    by_pitch: dict[int, list[QNote]] = defaultdict(list)
    for q in by_key.values():
        by_pitch[q.pitch].append(q)
    out = []
    for lst in by_pitch.values():
        lst.sort(key=lambda q: q.start)
        for a, b in zip(lst, lst[1:]):
            if a.end > b.start:
                a.end = b.start
        out.extend(lst)
    out.sort(key=lambda q: (q.start, q.pitch))
    return out


def align_to_barlines(qnotes: list[QNote], meter: Meter, shift: int | None = None) -> tuple[list[QNote], int]:
    """Choose which beats are downbeats and shift so the first bar starts at position 0.

    The downbeat phase maximises accented, long and low (bass) onsets on beat 1.
    Returns (shifted notes, origin) where origin is the beat index that became 0.
    """
    bpb = meter.beats_per_bar
    if shift is None:
        strength: dict[int, float] = defaultdict(float)
        acc = accent_weights([float(q.start) for q in qnotes], [q.pitch for q in qnotes],
                             [q.velocity for q in qnotes], bpb / 2)
        for q, a in zip(qnotes, acc):
            if q.start.denominator == 1:
                dur = min(float(q.end - q.start), 4.0)
                strength[int(q.start)] += a * (1.0 + 0.5 * dur)
        scores = [sum(v for b, v in strength.items() if b % bpb == s) for s in range(bpb)]
        phase = int(np.argmax(scores))
    else:
        phase = shift % bpb
    first = math.floor(min(q.start for q in qnotes))
    origin = first - ((first - phase) % bpb)
    return [QNote(q.start - origin, q.end - origin, q.pitch, q.velocity, q.graces) for q in qnotes], origin
