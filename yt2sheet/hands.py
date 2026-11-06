"""Split quantized piano notes between the right hand (treble) and left hand (bass)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .notes import QNote

COMFORT_SPAN = 12  # semitones a hand covers comfortably (an octave)
MAX_SPAN = 14  # a 9th: a stretch for most hands
MAX_NOTES = 5


RECENT_BEATS = 3  # how far back a hand's recent notes inform where it is (a 3/4 bar)


def _dist(new: list[int], center: float, recent: list[int]) -> float:
    """How far (octaves) a hand must move to play chord `new`: from its running register,
    or from a note it just played (so repeating figures like oom-pah stay in one hand even
    when they jump). Measured to the chord's farthest note, once per chord, so moving a
    whole chord is not counted once per note."""

    def d(p: int) -> float:
        x = abs(p - center)
        if recent:
            x = min(x, min(abs(p - r) for r in recent) + 2)
        return x

    return max(d(p) for p in new) / 12.0


def _hand_cost(new: list[int], held: list[int], center: float, recent: list[int], is_rh: bool) -> float:
    if not new:
        return 0.0
    cost = _dist(new, center, recent)
    # Physical constraint: everything this hand would be holding must fit under it.
    sounding = new + held
    span = max(sounding) - min(sounding)
    cost += max(0, span - COMFORT_SPAN) * 0.3 + max(0, span - MAX_SPAN) * 0.5 + max(0, span - MAX_SPAN - 3) * 2.0
    cost += max(0, len(sounding) - MAX_NOTES) * 1.0
    # Notation convention: RH reads treble, LH reads bass; crossing middle C is mildly penalised.
    if is_rh:
        cost += sum(max(0, 55 - p) for p in new) / 12.0 * 0.5
    else:
        cost += sum(max(0, p - 64) for p in new) / 12.0 * 0.5
    if held:  # mild preference for giving new notes to the hand that is free
        cost += 0.1
    return cost


BEAM = 24  # hypotheses kept by the beam search
MELODY_WEIGHT = 2.0  # cost of the left hand taking a note that sounds like the melody
MELODY_WINDOW = 2.0  # beats either side used to judge how loud "loud" is
HOCKET_WEIGHT = 1.5  # cost of the right hand taking a soft note out of a left-hand figure
FIGURE_GAP = 1.0  # beats: the left hand is "in a figure" if it played within this long ago


def note_melody_likelihood(qnotes: list[QNote]) -> dict[int, float]:
    """For each note (keyed by id), how much it sounds like melody (0..1).

    Pianists bring out the melody: it is played louder than the accompaniment around it.
    Loudness is judged against the notes within MELODY_WINDOW beats, so a quiet passage's
    melody still stands out from its own accompaniment.
    """
    if not qnotes:
        return {}
    starts = np.array([float(q.start) for q in qnotes])
    vel = np.array([q.velocity for q in qnotes], dtype=float)
    order = np.argsort(starts)
    s_sorted, v_sorted = starts[order], vel[order]
    out = {}
    for q in qnotes:
        lo = np.searchsorted(s_sorted, float(q.start) - MELODY_WINDOW)
        hi = np.searchsorted(s_sorted, float(q.start) + MELODY_WINDOW, side="right")
        around = v_sorted[lo:hi]
        accomp, melody = np.percentile(around, 35), np.percentile(around, 90)
        loud = float(np.clip((q.velocity - accomp) / max(melody - accomp, 8.0), 0.0, 1.0))
        # a lone bass note in a quiet spot is loud for its surroundings but is not melody
        register = float(np.clip((q.pitch - 48) / 7.0, 0.0, 1.0))  # 0 at C3 and below, 1 from G3
        out[id(q)] = loud * register
    return out


def melody_likelihood(qnotes: list[QNote]) -> dict[Fraction, float]:
    """For each onset, the melody likelihood of its top note."""
    per_note = note_melody_likelihood(qnotes)
    tops: dict[Fraction, QNote] = {}
    for q in qnotes:
        if q.start not in tops or q.pitch > tops[q.start].pitch:
            tops[q.start] = q
    return {start: per_note[id(q)] for start, q in tops.items()}


@dataclass(frozen=True)
class _Hand:
    center: float  # running register (MIDI pitch)
    recent: tuple  # last notes played (QNote), newest last
    active: tuple  # notes still held down (QNote)

    def advance(self, start, new: list[QNote]) -> "_Hand":
        active = tuple(q for q in self.active if q.end > start) + tuple(new)
        recent = (self.recent + tuple(new))[-12:]
        center = self.center
        if new:
            center = 0.6 * center + 0.4 * (sum(q.pitch for q in new) / len(new))
        return _Hand(center, recent, active)


def _separate(rh: _Hand, lh: _Hand) -> tuple[_Hand, _Hand]:
    """Keep the two hands' registers at least a 5th apart."""
    if rh.center - lh.center >= 7:
        return rh, lh
    mid = (rh.center + lh.center) / 2
    return _Hand(mid + 3.5, rh.recent, rh.active), _Hand(mid - 3.5, lh.recent, lh.active)


def _step_cost(ps: list[int], k: int, start, rh: _Hand, lh: _Hand, mus: list[float] | None = None) -> float:
    """Cost of giving ps[:k] to the left hand and ps[k:] to the right hand at `start`.

    `mus` are the notes' melody likelihoods (aligned with ps): the LH taking the melody costs
    extra, and so does the RH taking a soft note out of the middle of a LH figure (the hands
    would alternate within one broken chord)."""
    mus = mus or [0.0] * len(ps)
    rh_held = [q.pitch for q in rh.active if q.end > start]
    lh_held = [q.pitch for q in lh.active if q.end > start]
    rh_recent = [q.pitch for q in rh.recent if q.start >= start - RECENT_BEATS]
    lh_recent = [q.pitch for q in lh.recent if q.start >= start - RECENT_BEATS]
    left, right = ps[:k], ps[k:]
    cost = _hand_cost(left, lh_held, lh.center, lh_recent, is_rh=False)
    cost += _hand_cost(right, rh_held, rh.center, rh_recent, is_rh=True)
    if 0 < k < len(ps):  # hands rarely split a chord at a small interval
        cost += max(0, 5 - (ps[k] - ps[k - 1])) * 0.15
    # Crossed hands (LH sounding above the RH) are awkward unless the music demands it.
    lh_top = max(left + lh_held, default=None)
    rh_bottom = min(right + rh_held, default=None)
    if lh_top is not None and rh_bottom is not None and lh_top > rh_bottom:
        cost += 1.0 + (lh_top - rh_bottom) / 6.0
    if k == len(ps):  # the left hand takes the top note too
        cost += MELODY_WEIGHT * mus[-1]
    if k < len(ps) and lh.recent:
        # The LH is in a figure if it played within FIGURE_GAP beats, whatever the melody did
        # meanwhile (the melody is a separate voice; the figure goes on underneath it).
        lh_last = max(q.start for q in lh.recent)
        if start - lh_last <= FIGURE_GAP:
            # the figure is everything the LH played lately (an oom-pah's chords as well as its bass)
            figure = [q.pitch for q in lh.recent if q.start >= start - 2 * FIGURE_GAP]
            for p, mu in zip(right, mus[k:]):
                if min(abs(p - f) for f in figure) <= 9:
                    cost += HOCKET_WEIGHT * (1.0 - mu)
    return cost


def split_hands(
    qnotes: list[QNote], split_point: int | None = None, beam: int = BEAM
) -> tuple[list[QNote], list[QNote]]:
    """Return (right_hand, left_hand).

    For every onset the notes are split into a lower (LH) and upper (RH) part. A beam search
    over the whole piece picks the sequence of splits with the lowest total cost, so an early
    choice is judged by how playable it leaves what follows. With `split_point` set, use a
    fixed split instead (notes >= split_point go to the RH).
    """
    if split_point is not None:
        rh = [q for q in qnotes if q.pitch >= split_point]
        lh = [q for q in qnotes if q.pitch < split_point]
        return rh, lh

    groups: dict = defaultdict(list)
    for q in qnotes:
        groups[q.start].append(q)
    starts = sorted(groups)
    ordered = [sorted(groups[s], key=lambda q: q.pitch) for s in starts]

    mel = note_melody_likelihood(qnotes)
    hyps = [(0.0, _Hand(67.0, (), ()), _Hand(50.0, (), ()), None)]
    for start, grp in zip(starts, ordered):
        ps = [q.pitch for q in grp]
        mus = [mel[id(q)] for q in grp]
        expanded = []
        for cost, rh, lh, path in hyps:
            for k in range(len(ps) + 1):
                expanded.append((cost + _step_cost(ps, k, start, rh, lh, mus), rh, lh, path, k))
        expanded.sort(key=lambda h: h[0])
        hyps, seen = [], set()
        for cost, rh, lh, path, k in expanded:
            nrh, nlh = _separate(rh.advance(start, grp[k:]), lh.advance(start, grp[:k]))
            key = (round(nrh.center), round(nlh.center), tuple(q.pitch for q in nrh.active),
                   tuple(q.pitch for q in nlh.active), tuple(q.pitch for q in nrh.recent[-2:]))
            if key in seen:
                continue
            seen.add(key)
            hyps.append((cost, nrh, nlh, (k, path)))
            if len(hyps) >= beam:
                break

    ks = []
    node = hyps[0][3]
    while node is not None:
        k, node = node
        ks.append(k)
    ks.reverse()
    rh, lh = [], []
    for grp, k in zip(ordered, ks):
        lh.extend(grp[:k])
        rh.extend(grp[k:])
    return rh, lh
