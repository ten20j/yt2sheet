"""Follow the meter through a performance.

A beat tracker gives a stream of pulses, but in expressive playing it slips: when the
pianist stretches a phrase ending it counts one pulse too many or too few, and from there
on every barline is shifted (bass notes land on the "and" of the beat, everything is tied
over the barline). Choosing the downbeat once for the whole piece cannot fix that.

Here every pulse gets a position in the bar by dynamic programming. Positions normally
count up, but the search may insert or drop a pulse, at a cost, when that puts the bass
notes and pedal changes that follow back on strong beats. That is what a transcriber does
by ear: the held note at the phrase end is "a half note, ritardando", not "a half note tied
to an 8th", and the next phrase starts on the beat again.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .notes import Note
from .rhythm import Meter, _cluster_onsets, local_bass, quantize, time_to_beat, track_beats

EDIT_COST = 3.0  # inserting or dropping a pulse must be repaid by several re-aligned bass notes
GAP_EDIT_COST = 1.0  # ...but in a stretch with no notes the tracker was only guessing anyway
DROP_ONSET_COST = 4.0  # extra cost of dropping a pulse that has notes on it
BASS_SPREAD = 7  # a bass note is within this many semitones of the lowest note of its bar
SPLIT_FIGURE_COST = 2.0  # extra cost of inserting a pulse between two consecutive played pulses
PEDAL_WEIGHT = 0.7  # a pedal change counts a little less than a bass note
ISOLATED_WEIGHT = 1.5  # a lone event (nothing within 1.5 pulses) has no context to be off the beat


@dataclass
class MetricGrid:
    """Pulse times with consecutive metrical positions."""

    pulses: np.ndarray  # seconds
    pulses_per_beat: int = 1
    first_position: int = 0  # position of pulses[0] within its bar, in pulses
    edits: list = field(default_factory=list)  # ("insert" | "drop", seconds) corrections made

    def beat_position(self, t) -> np.ndarray:
        """Seconds -> beats, where every multiple of the bar length is a barline."""
        return (time_to_beat(t, self.pulses) + self.first_position) / self.pulses_per_beat

    @property
    def bpm(self) -> float:
        return 60.0 / float(np.median(np.diff(self.pulses))) / self.pulses_per_beat


def should_halve(notes: list[Note], pulses: np.ndarray, meter: Meter) -> bool:
    """Is the tracked pulse really the 8th note? True for fast continuous motion in an even
    simple meter with hardly any onsets between the pulses (flowing arpeggios); such music is
    written as 8ths at half the tempo, not as a stream of quarters."""
    tempo = 60.0 / float(np.median(np.diff(pulses)))
    if meter.compound or meter.beats_per_bar % 2 or tempo <= 110:
        return False
    starts = {q.start for q in quantize(notes, pulses, meter)}
    offbeat = sum(s.denominator != 1 for s in starts) / max(1, len(starts))
    return offbeat < 0.10


def _template(meter: Meter, ppb: int) -> np.ndarray:
    """How well a bass note / pedal change fits each position of the bar."""
    P = ppb * meter.beats_per_bar
    t = np.full(P, -0.75)  # between the beats
    t[::ppb] = 0.15  # on a beat
    if meter.beats_per_bar % 2 == 0:
        t[P // 2] = 0.75  # the half bar
    t[0] = 1.0  # the downbeat
    return t


def _evidence(notes: list[Note], pedal, pulses: np.ndarray, ppb: int, P: int) -> tuple[np.ndarray, np.ndarray]:
    """Per pulse: (strong-beat evidence from bass notes and pedal changes, has an onset)."""
    n = len(pulses)
    pos = time_to_beat(np.array(_cluster_onsets(notes)), pulses)
    pitches = np.array([x.pitch for x in notes])
    near = np.clip(np.round(pos).astype(int), 0, n - 1)
    has_onset = np.zeros(n)
    has_onset[near] = 1.0

    evidence = np.zeros(n)
    # A bass note is the lowest note around its beat and lies in the bass register of its bar
    # (an inner accompaniment note is often the lowest thing for a beat, but not for the bar).
    order = np.argsort(pos)
    sorted_pos, sorted_pitch = pos[order], pitches[order]
    lo = np.searchsorted(sorted_pos, pos - P)
    hi = np.searchsorted(sorted_pos, pos + P, side="right")
    bar_lowest = np.array([sorted_pitch[a:b].min() for a, b in zip(lo, hi)])
    is_bass = local_bass(pos, pitches, float(ppb)) & (pitches <= bar_lowest + BASS_SPREAD)
    prev_pc = None
    for i in np.argsort(pos):
        if not is_bass[i]:
            continue
        pc = int(pitches[i]) % 12
        w = 1.0 + (0.5 if prev_pc is not None and pc != prev_pc else 0.0)  # harmony changes
        prev_pc = pc
        # a bass note between two pulses counts partly for each
        off = pos[i] - near[i]
        share = float(np.clip((abs(off) - 0.2) / 0.6, 0.0, 0.5))
        other = int(np.clip(near[i] + (1 if off > 0 else -1), 0, n - 1))
        evidence[near[i]] += w * (1.0 - share)
        evidence[other] += w * share
    occupied = np.flatnonzero(has_onset)
    for j in occupied:
        others = occupied[occupied != j]
        if len(others) == 0 or np.min(np.abs(others - j)) >= 2:
            evidence[j] += ISOLATED_WEIGHT
    for down, _up in pedal:
        # the pedal is changed just after the new harmony's first note
        i = int(math.floor(float(time_to_beat(down, pulses)) + 0.25))
        if 0 <= i < n:
            evidence[i] += PEDAL_WEIGHT
    return evidence, has_onset


def align_meter(notes: list[Note], pedal, pulses: np.ndarray, meter: Meter, ppb: int = 1,
                elastic: bool = True) -> MetricGrid:
    """Give every pulse a bar position (see module docstring). With elastic=False the grid
    is kept as tracked and only the phase (which pulse is the downbeat) is chosen."""
    P = ppb * meter.beats_per_bar
    n = len(pulses)
    evidence, has_onset = _evidence(notes, pedal, pulses, ppb, P)
    E = np.outer(evidence, _template(meter, ppb))

    empty = has_onset == 0
    in_gap = empty & np.roll(empty, 1) & np.roll(empty, -1)  # no notes on this pulse or its neighbours
    drop_cost = np.where(in_gap, GAP_EDIT_COST, EDIT_COST + DROP_ONSET_COST * has_onset)  # dropping pulse j
    insert_cost = np.where(empty[:-1] & empty[1:], GAP_EDIT_COST,
                           EDIT_COST + SPLIT_FIGURE_COST * has_onset[:-1] * has_onset[1:])  # between j and j+1

    V = np.full((n, P), -np.inf)
    back = np.zeros((n, P), dtype=np.int8)  # 0 normal, 1 pulse inserted before i, 2 pulse i-1 dropped
    V[0] = E[0]
    for i in range(1, n):
        normal = np.roll(V[i - 1], 1)  # position m comes from m-1
        best, how = normal, np.zeros(P, dtype=np.int8)
        if elastic:
            ins = np.roll(V[i - 1], 2) - insert_cost[i - 1]
            how = np.where(ins > best, 1, how)
            best = np.maximum(best, ins)
            if i >= 2:
                drop = np.roll(V[i - 2], 1) - drop_cost[i - 1]
                how = np.where(drop > best, 2, how)
                best = np.maximum(best, drop)
        V[i] = best + E[i]
        back[i] = how

    i, m = n - 1, int(np.argmax(V[n - 1]))
    times, edits = [], []
    while True:
        times.append(pulses[i])
        if i == 0:
            break
        how = back[i, m]
        if how == 1:
            mid = (pulses[i - 1] + pulses[i]) / 2
            times.append(mid)
            edits.append(("insert", float(mid)))
            i, m = i - 1, (m - 2) % P
        elif how == 2:
            edits.append(("drop", float(pulses[i - 1])))
            i, m = i - 2, (m - 1) % P
        else:
            i, m = i - 1, (m - 1) % P
    return MetricGrid(np.array(times[::-1]), ppb, m, edits[::-1])


def anchor_isolated(notes: list[Note], pulses: np.ndarray) -> np.ndarray:
    """Move the nearest pulse onto every isolated onset.

    Around a lone chord (a free-time introduction, a fermata) the tracker's pulses are
    extrapolated guesses; the chord itself is the only thing that says where a pulse is."""
    pulses = pulses.copy()
    ibi = float(np.median(np.diff(pulses)))
    onsets = np.unique(np.round(_cluster_onsets(notes), 3))
    for k, t in enumerate(onsets):
        before = t - onsets[k - 1] if k > 0 else np.inf
        after = onsets[k + 1] - t if k + 1 < len(onsets) else np.inf
        if min(before, after) <= 1.5 * ibi:
            continue
        j = int(np.argmin(np.abs(pulses - t)))
        if 0 < j < len(pulses) - 1 and pulses[j - 1] < t < pulses[j + 1]:
            pulses[j] = t
    return pulses


def metric_grid(notes: list[Note], meter: Meter, pedal=(), bpm: float | None = None,
                elastic: bool = True) -> MetricGrid:
    """Track the pulse, decide whether it is the beat or the 8th, and follow the meter."""
    pulses = track_beats(notes, bpm=bpm)
    ppb = 2 if (not bpm and should_halve(notes, pulses, meter)) else 1
    if elastic:
        pulses = anchor_isolated(notes, pulses)
    return align_meter(notes, pedal, pulses, meter, ppb, elastic)
