"""Piano fingering by dynamic programming.

Each hand's part is a sequence of events (a single note or a chord). Every event gets
a finger assignment (one finger per note, increasing towards the little finger), and
the Viterbi algorithm picks the sequence with the lowest total difficulty.

Difficulty follows Parncutt, Sloboda, Clarke, Raekallio & Desain (1997), "An ergonomic
model of keyboard fingering for melodic fragments": a table of comfortable / relaxed /
practical spans per finger pair plus rules for weak fingers, the thumb on black keys
and thumb passing. Chords use the same span table between simultaneous notes, and
chord-to-chord moves average it over all note pairs.

The left hand is handled as a mirror image of the right hand (pitches negated), so one
set of rules serves both.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

# (MinPrac, MinComf, MinRel, MaxRel, MaxComf, MaxPrac) in semitones, right hand,
# for finger i on the lower note and finger j (> i) on the upper note.
SPANS = {
    (1, 2): (-5, -3, 1, 5, 8, 10),
    (1, 3): (-4, -2, 3, 7, 10, 12),
    (1, 4): (-3, -1, 5, 9, 12, 14),
    (1, 5): (-1, 1, 7, 10, 13, 15),
    (2, 3): (1, 1, 1, 2, 3, 5),
    (2, 4): (1, 1, 3, 4, 5, 7),
    (2, 5): (2, 2, 5, 6, 8, 10),
    (3, 4): (1, 1, 1, 2, 2, 4),
    (3, 5): (1, 1, 3, 4, 5, 7),
    (4, 5): (1, 1, 1, 2, 3, 5),
}
# Spacing (semitones) of each finger pair in a relaxed five-finger position; deviation
# from it is a small tie-breaker that keeps the hand in natural shapes (5-1-3-1 Alberti).
NATURAL = {(1, 2): 2, (1, 3): 4, (1, 4): 5, (1, 5): 7, (2, 3): 2, (2, 4): 3.5, (2, 5): 5.5,
           (3, 4): 1.5, (3, 5): 3.5, (4, 5): 1.5}
NATURAL_WEIGHT = 0.05
BLACK = {1, 3, 6, 8, 10}
JUMP_COST = 8.0  # a move the hand cannot stretch to is a shift: same cost whatever the fingers
SAME_FINGER_COST = 30.0
THUMB_ON_BLACK = 4.0  # tuned so no major scale puts the RH thumb on a black key
WEAK = {4: 1.0, 5: 0.1}  # 4 is the weak finger; 5 is natural on outer notes (LH bass, RH top)


def is_black(midi: int) -> bool:
    return midi % 12 in BLACK


def span_cost(f1: int, q1: int, f2: int, q2: int, stretch_only: bool = False) -> float:
    """Parncutt rules 1-3 (stretch, small span, large span) for two different fingers.

    q are mirrored pitches (hand coordinates), so the same table serves both hands.
    """
    if f1 > f2:
        f1, q1, f2, q2 = f2, q2, f1, q1
    min_prac, min_comf, min_rel, max_rel, max_comf, max_prac = SPANS[(f1, f2)]
    d = q2 - q1
    cost = NATURAL_WEIGHT * abs(d - NATURAL[(f1, f2)])
    if f1 == 1 and d < 0 and is_black(abs(q2)) and not is_black(abs(q1)):
        # A finger crossing over the thumb onto a raised black key (4 on B-flat, thumb on C
        # in F major) is far easier than the flat semitone distance suggests.
        d = min(0, d + 2)
    if d > max_comf:
        cost += 2 * (d - max_comf)
    elif d < min_comf:
        cost += 2 * (min_comf - d)
    if not stretch_only:
        w = 1 if f1 == 1 else 2
        if d < min_rel:
            cost += w * (min_rel - d)
        elif d > max_rel:
            cost += w * (d - max_rel)
    if d > max_prac:
        cost += 10 * (d - max_prac)
    elif d < min_prac:
        cost += 10 * (min_prac - d)
    return cost


def step_cost(f1: int, q1: int, b1: bool, f2: int, q2: int, b2: bool) -> float:
    """Difficulty of playing note (f2, q2) right after (f1, q1); b = on a black key."""
    if f1 == f2:
        if q1 == q2:
            return 0.0
        # Same finger on a new key breaks legato (Parncutt excludes it outright). Only for
        # leaps the hand cannot stretch anyway is it no worse than any other shift.
        if abs(q2 - q1) <= 12:
            return SAME_FINGER_COST
        # A leap with the same finger: price it like a pair with no span at all, so that
        # landing on the natural finger (thumb on G3, then 5 on C2) is clearly preferred.
        return _soft_cap(10.0 * abs(q2 - q1))
    cost = span_cost(f1, q1, f2, q2)
    # rule 8 three-to-four, rule 9 four-on-black
    if f1 == 3 and f2 == 4:
        cost += 1
    if {f1, f2} == {3, 4} and (b2 if f2 == 4 else b1) and not (b1 if f1 == 3 else b2):
        cost += 1
    # rule 10/11: thumb or little finger on a black key next to a white key
    if f2 == 1 and b2 and not b1:
        cost += 2
    if f1 == 1 and b1 and not b2:
        cost += 2
    if f2 == 5 and b2 and not b1:
        cost += 2
    if f1 == 5 and b1 and not b2:
        cost += 2
    # rule 12: thumb passing (thumb under, or finger over the thumb)
    passing = (f2 == 1 and q2 > q1) or (f1 == 1 and q2 < q1)
    if passing:
        thumb_black = b2 if f2 == 1 else b1
        other_black = b1 if f2 == 1 else b2
        if thumb_black == other_black:
            cost += 1
        elif thumb_black and not other_black:
            cost += 3
    return _soft_cap(cost)


def _soft_cap(cost: float) -> float:
    """Beyond JUMP_COST the hand shifts anyway, so extra stretch barely matters; keep a small
    slope so that, among shifts, the less contorted landing still wins (C2 with 5, not 2)."""
    return cost if cost <= JUMP_COST else JUMP_COST + 0.1 * (cost - JUMP_COST)


def chord_cost(fingers: tuple[int, ...], qs: tuple[int, ...], blacks: tuple[bool, ...]) -> float:
    """Cost of one event on its own: span between simultaneous notes, weak fingers, thumb on black."""
    cost = sum(WEAK.get(f, 0.0) for f in fingers)
    cost += sum(THUMB_ON_BLACK for f, b in zip(fingers, blacks) if f == 1 and b)
    for a, b in itertools.combinations(range(len(fingers)), 2):
        cost += span_cost(fingers[a], qs[a], fingers[b], qs[b])
    return cost


def three_note_cost(f1: int, q1: int, f2: int, q2: int, f3: int, q3: int) -> float:
    """Parncutt rules 4-5: do the 1st and 3rd of three notes fit one hand position?

    Catches shapes that every consecutive pair allows but the whole figure does not
    (e.g. 3-1-2-1 on an Alberti bass, where pianists keep the hand still with 5-1-3-1)."""
    if abs(q2 - q1) > 12 or abs(q3 - q2) > 12:
        return 0.0  # a leap moved the whole hand in between: no shared hand position to judge
    thumb_between = f2 == 1 and min(q1, q3) < q2 < max(q1, q3)
    if f1 == f3:
        if q1 == q3:
            return 0.0
        if abs(q2 - q1) > 7 or abs(q3 - q2) > 7:
            # e.g. A3(1) A2(5) E3(1): the leap already moved the hand, and the step paid for it
            return 0.3 * min(abs(q3 - q1), 7) + 0.5
        # stepwise crawling with the same finger (1-2-1-2, or 2-1-2 around a tucked thumb):
        # a full position change of that interval
        return min(abs(q3 - q1), 7) + (2.0 if thumb_between else 1.0)
    lo_f, lo_q, hi_f, hi_q = (f1, q1, f3, q3) if f1 < f3 else (f3, q3, f1, q1)
    min_prac, min_comf, _, _, max_comf, max_prac = SPANS[(lo_f, hi_f)]
    d = hi_q - lo_q
    cost = NATURAL_WEIGHT * abs(d - NATURAL[(lo_f, hi_f)])
    if min_comf <= d <= max_comf:
        return cost
    size = (min_comf - d) if d < min_comf else (d - max_comf)
    full = thumb_between and not (min_prac <= d <= max_prac)
    return cost + size + (2.0 if full else 1.0)


def move_cost(prev: "_State", cur: "_State") -> float:
    if len(prev.fingers) == 1 and len(cur.fingers) == 1:
        return step_cost(prev.fingers[0], prev.qs[0], prev.blacks[0], cur.fingers[0], cur.qs[0], cur.blacks[0])
    costs = [
        step_cost(fa, qa, ba, fb, qb, bb)
        for fa, qa, ba in zip(prev.fingers, prev.qs, prev.blacks)
        for fb, qb, bb in zip(cur.fingers, cur.qs, cur.blacks)
    ]
    return sum(costs) / len(costs)


@dataclass(frozen=True)
class _State:
    fingers: tuple[int, ...]  # aligned with qs (ascending hand coordinates)
    qs: tuple[int, ...]
    blacks: tuple[bool, ...]


def _states(pitches: list[int], hand: str) -> list[_State]:
    qs = sorted(-p for p in pitches) if hand == "L" else sorted(pitches)
    blacks = tuple(is_black(abs(q)) for q in qs)
    return [_State(f, tuple(qs), blacks) for f in itertools.combinations(range(1, 6), len(qs))]


def finger_events(
    events: list[tuple[float, list[int]]], hand: str, seconds_per_unit: float = 0.5
) -> list[dict[int, int] | None]:
    """Assign fingers to a hand's events.

    events: (onset time in any unit, pitches) in time order; hand: "R" or "L".
    Returns, per event, {midi pitch: finger}, or None for an event with more than five notes.
    A long gap before an event lets the hand reposition, so moves across it cost less.

    Exact second-order Viterbi: a state is the fingering of the previous and current
    event, so the three-note rules are scored exactly.
    """
    out: list[dict[int, int] | None] = [None] * len(events)
    segment: list[int] = []
    for i, (_, pitches) in enumerate(events):
        if 0 < len(set(pitches)) <= 5:
            segment.append(i)
            continue
        _solve(events, segment, hand, seconds_per_unit, out)
        segment = []
    _solve(events, segment, hand, seconds_per_unit, out)
    return out


def _gap_weight(t0: float, t1: float, seconds_per_unit: float) -> float:
    gap = max(0.0, (t1 - t0) * seconds_per_unit)
    return 1.0 if gap <= 0.35 else max(0.15, 0.35 / gap)


def _solve(events, idx: list[int], hand: str, spu: float, out) -> None:
    if not idx:
        return
    layers = [_states(sorted(set(events[i][1])), hand) for i in idx]
    here = [[chord_cost(s.fingers, s.qs, s.blacks) for s in layer] for layer in layers]
    if len(idx) == 1:
        b = min(range(len(layers[0])), key=lambda k: here[0][k])
        out[idx[0]] = _as_dict(layers[0][b], hand)
        return
    w = [_gap_weight(events[idx[k - 1]][0], events[idx[k]][0], spu) for k in range(1, len(idx))]
    # V[(a, b)] = best cost of the prefix ending with states a (event k-1) and b (event k)
    V = {}
    for a, sa in enumerate(layers[0]):
        for b, sb in enumerate(layers[1]):
            V[(a, b)] = here[0][a] + w[0] * move_cost(sa, sb) + here[1][b]
    backs = []
    for k in range(2, len(idx)):
        prev_l, cur_l, next_l = layers[k - 2], layers[k - 1], layers[k]
        single = len(prev_l[0].fingers) == len(cur_l[0].fingers) == len(next_l[0].fingers) == 1
        wk, w3 = w[k - 1], w[k - 1] * w[k - 2]
        moves = {(b, c): move_cost(sb, sc) for b, sb in enumerate(cur_l) for c, sc in enumerate(next_l)}
        NV, back = {}, {}
        for (b, c), mv in moves.items():
            best_a, best = None, float("inf")
            for a in range(len(prev_l)):
                v = V[(a, b)]
                if single:
                    pa, pb, pc = prev_l[a], cur_l[b], next_l[c]
                    v += w3 * three_note_cost(pa.fingers[0], pa.qs[0], pb.fingers[0], pb.qs[0],
                                              pc.fingers[0], pc.qs[0])
                if v < best:
                    best_a, best = a, v
            NV[(b, c)] = best + wk * mv + here[k][c]
            back[(b, c)] = best_a
        V = NV
        backs.append(back)
    b, c = min(V, key=V.get)
    path = [c, b]
    for back in reversed(backs):
        a = back[(b, c)]
        path.append(a)
        b, c = a, b
    path.reverse()
    for pos, (k, s) in enumerate(zip(idx, path)):
        out[k] = _as_dict(layers[pos][s], hand)


def _as_dict(state: "_State", hand: str) -> dict[int, int]:
    return {(-q if hand == "L" else q): f for q, f in zip(state.qs, state.fingers)}


def hand_position(fingering: dict[int, int], hand: str) -> float:
    """Where the thumb sits (semitones) for a fingered event, assuming a relaxed five-finger shape."""
    offset = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7}
    sign = -1 if hand == "L" else 1
    est = sorted(p - sign * offset[f] for p, f in fingering.items())
    return est[len(est) // 2]
