"""Split quantized piano notes between the right hand (treble) and left hand (bass)."""

from __future__ import annotations

from collections import defaultdict

from .notes import QNote

MAX_SPAN = 14  # semitones a hand comfortably covers (a 9th)
MAX_NOTES = 5


RECENT_BEATS = 2  # how far back a hand's recent notes inform where it is


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
    cost += max(0, span - MAX_SPAN) * 0.5 + max(0, span - MAX_SPAN - 3) * 2.0
    cost += max(0, len(sounding) - MAX_NOTES) * 1.0
    # Notation convention: RH reads treble, LH reads bass; crossing middle C is mildly penalised.
    if is_rh:
        cost += sum(max(0, 55 - p) for p in new) / 12.0 * 0.5
    else:
        cost += sum(max(0, p - 64) for p in new) / 12.0 * 0.5
    if held:  # mild preference for giving new notes to the hand that is free
        cost += 0.1
    return cost


def split_hands(qnotes: list[QNote], split_point: int | None = None) -> tuple[list[QNote], list[QNote]]:
    """Return (right_hand, left_hand).

    With `split_point` set, use a fixed split (notes >= split_point go to the RH).
    """
    if split_point is not None:
        rh = [q for q in qnotes if q.pitch >= split_point]
        lh = [q for q in qnotes if q.pitch < split_point]
        return rh, lh

    groups: dict = defaultdict(list)
    for q in qnotes:
        groups[q.start].append(q)

    rh_c, lh_c = 67.0, 50.0
    rh: list[QNote] = []
    lh: list[QNote] = []
    rh_active: list[QNote] = []
    lh_active: list[QNote] = []
    for start in sorted(groups):
        grp = sorted(groups[start], key=lambda q: q.pitch)
        ps = [q.pitch for q in grp]
        rh_active = [q for q in rh_active if q.end > start]
        lh_active = [q for q in lh_active if q.end > start]
        rh_held = [q.pitch for q in rh_active]
        lh_held = [q.pitch for q in lh_active]
        rh_recent = [q.pitch for q in rh[-12:] if q.start >= start - RECENT_BEATS]
        lh_recent = [q.pitch for q in lh[-12:] if q.start >= start - RECENT_BEATS]
        best_k, best_cost = 0, float("inf")
        for k in range(len(ps) + 1):
            cost = _hand_cost(ps[:k], lh_held, lh_c, lh_recent, is_rh=False) + _hand_cost(
                ps[k:], rh_held, rh_c, rh_recent, is_rh=True
            )
            if 0 < k < len(ps):  # hands rarely split a chord at a small interval
                cost += max(0, 5 - (ps[k] - ps[k - 1])) * 0.15
            if cost < best_cost:
                best_k, best_cost = k, cost
        left, right = grp[:best_k], grp[best_k:]
        lh.extend(left)
        rh.extend(right)
        lh_active.extend(left)
        rh_active.extend(right)
        if left:
            lh_c = 0.6 * lh_c + 0.4 * (sum(q.pitch for q in left) / len(left))
        if right:
            rh_c = 0.6 * rh_c + 0.4 * (sum(q.pitch for q in right) / len(right))
        if rh_c - lh_c < 7:  # keep the hands' registers apart
            mid = (rh_c + lh_c) / 2
            rh_c, lh_c = mid + 3.5, mid - 3.5
    return rh, lh
