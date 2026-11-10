"""Build an engraved-ready piano score (music21) from quantized notes."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction

from .notes import QNote
from .rhythm import Meter

LEGATO_FILL = Fraction(1, 2)  # close gaps up to this many beats before the next onset
MIN_SHOWN = Fraction(1, 2)  # show detached short notes as at least an 8th when there is room

SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES = ["C", "D-", "D", "E-", "E", "F", "G-", "G", "A-", "A", "B-", "B"]
NEUTRAL_NAMES = ["C", "C#", "D", "E-", "E", "F", "F#", "G", "G#", "A", "B-", "B"]


@dataclass
class Event:
    start: Fraction  # quarter lengths from the first barline
    end: Fraction
    pitches: list[int]
    velocity: int
    tie: str | None = None  # 'start' | 'continue' | 'stop'
    graces: tuple = ()  # grace-note pitches played just before this event


# --------------------------------------------------------------------------- staff events


def _pedal_release(pos: Fraction, pedal: list[tuple[float, float]]) -> Fraction | None:
    """If the sustain pedal is down at beat `pos`, the beat (8th-rounded) where it lifts."""
    x = float(pos)
    for a, b in pedal:
        if a - 0.1 <= x <= b:
            return Fraction(round(b * 2), 2)
    return None


def _is_plain(beats: Fraction, meter: Meter) -> bool:
    """Can this many beats be written as one note (plain or single-dotted)?"""
    ql = beats * meter.beat_ql
    for base in (Fraction(1, 4), Fraction(1, 2), Fraction(1), Fraction(2), Fraction(4)):
        if ql in (base, base * 3 / 2):
            return True
    return False


def _plain_end(start: Fraction, end: Fraction, at_least: Fraction, meter: Meter) -> Fraction:
    """The latest end not after `end` (and not before `at_least`) that makes one plain note
    ending on a beat, the half bar or the barline; `end` itself if there is none."""
    if _is_plain(end - start, meter):
        return end
    bar = Fraction(meter.beats_per_bar)
    bar_start = (start // bar) * bar
    marks = {bar_start + b for b in range(1, meter.beats_per_bar + 1)}
    if meter.beats_per_bar % 2 == 0:
        marks.add(bar_start + bar / 2)
    for m in sorted(marks, reverse=True):
        if at_least <= m <= end and m > start and _is_plain(m - start, meter):
            return m
    return end


def staff_events(qnotes: list[QNote], meter: Meter, pedal: list[tuple[float, float]] | None = None,
                 melody: bool = True) -> list[Event]:
    """Collapse one hand's notes into a single chord stream (onset -> chord).

    `pedal` are sustain-pedal-down spans in beats; a note released under the pedal keeps
    sounding, so it is notated up to the next onset (or pedal lift) rather than with a rest.
    In a melody staff that may tie it over the barline (the line is still singing). In an
    accompaniment staff (`melody=False`) a note only the pedal sustains stays inside its
    bar and gets a plain length, with rests after it: a tie would claim a finger holds it.
    """
    pedal = pedal or []
    groups: dict[Fraction, list[QNote]] = defaultdict(list)
    for q in qnotes:
        groups[q.start].append(q)
    starts = sorted(groups)
    bar = Fraction(meter.beats_per_bar)
    events: list[Event] = []
    for i, s in enumerate(starts):
        grp = groups[s]
        held = max(q.end for q in grp)  # where the keys were released
        barline = (s // bar + 1) * bar
        # Accompaniment that only the pedal sustains is written inside its bar. Notes whose key
        # is held across the barline keep their tie.
        nxt = starts[i + 1] if i + 1 < len(starts) else None
        in_bar = not melody and nxt is not None and held <= barline  # the last note may ring on
        limit = barline if in_bar else None
        end = held
        lift = _pedal_release(end, pedal)
        if lift is not None and lift > end:
            end = lift if limit is None else min(lift, limit)
        if nxt is not None:
            end = min(max(end, s + MIN_SHOWN), nxt)
            gap = nxt - end
            if 0 < gap <= LEGATO_FILL and gap <= end - s and (limit is None or nxt <= limit):
                end = nxt
        else:
            end = max(end, s + MIN_SHOWN)
        if limit is not None and end > held and (nxt is None or end < nxt or end == limit):
            # the pedal chose this length, so choose a plain one: no double dots or ties
            end = _plain_end(s, end, max(held, s + MIN_SHOWN), meter)
        events.append(
            Event(
                start=s * meter.beat_ql,
                end=end * meter.beat_ql,
                pitches=sorted({q.pitch for q in grp}),
                velocity=max(q.velocity for q in grp),
                graces=tuple(dict.fromkeys(g for q in grp for g in q.graces)),
            )
        )
    return _split_tuplet_crossings(events, meter.beat_ql)


def _split_tuplet_crossings(events: list[Event], beat: Fraction) -> list[Event]:
    """Split events that mix tuplet and plain rhythms across a beat boundary into tied pieces.

    e.g. a note from the last triplet 8th of beat 1 to the middle of beat 2 becomes
    triplet-8th + 8th, tied, so every piece has a notatable duration.
    """
    out: list[Event] = []
    for ev in events:
        tupletish = (ev.start.denominator % 3 == 0) or (ev.end.denominator % 3 == 0)
        first_boundary = math.ceil(ev.start / beat) * beat
        if not tupletish or first_boundary >= ev.end:
            out.append(ev)
            continue
        cuts = [ev.start]
        if first_boundary > ev.start:
            cuts.append(first_boundary)
        last_boundary = math.floor(ev.end / beat) * beat
        if last_boundary > cuts[-1]:
            cuts.append(last_boundary)
        cuts.append(ev.end)
        cuts = sorted(set(cuts))
        pieces = list(zip(cuts, cuts[1:]))
        for j, (a, b) in enumerate(pieces):
            tie = None
            if len(pieces) > 1:
                tie = "start" if j == 0 else ("stop" if j == len(pieces) - 1 else "continue")
            out.append(Event(a, b, ev.pitches, ev.velocity, tie, ev.graces if j == 0 else ()))
    return out


# --------------------------------------------------------------------------- pitch spelling


def detect_key(qnotes: list[QNote]):
    from music21 import note, stream

    s = stream.Stream()
    for q in qnotes:
        n = note.Note(q.pitch)
        n.quarterLength = float(min(q.end - q.start, 4))
        s.append(n)
    return s.analyze("key")


def spelling_for_key(k) -> list[str]:
    """Map pitch class -> note name (music21 syntax) appropriate for key `k`."""
    base = SHARP_NAMES if k.sharps > 0 else FLAT_NAMES if k.sharps < 0 else NEUTRAL_NAMES
    names = list(base)
    scale = list(k.getPitches())
    if k.mode == "minor":  # raised 6th and 7th degrees
        scale += [k.tonic.transpose("M6"), k.tonic.transpose("M7")]
    for p in scale:
        names[p.pitchClass] = p.name
    return names


def _pitch(midi: int, names: list[str]):
    from music21 import pitch

    p = pitch.Pitch(names[midi % 12] + "4")
    p.octave += (midi - p.midi) // 12
    return p


# --------------------------------------------------------------------------- clefs


def _next_clef(current: str, pitches: list[int]) -> str:
    """Clef for a measure given the previous one; hysteresis avoids flip-flopping."""
    if not pitches:
        return current
    ps = sorted(pitches)
    lo, hi, med = ps[0], ps[-1], ps[len(ps) // 2]
    if current == "bass" and lo >= 57 and med >= 64:
        return "treble"
    if current == "treble" and hi <= 62 and med < 55:
        return "bass"
    return current


def _assign_clefs(part) -> None:
    """Change clef per measure when a hand moves far out of its staff's register."""
    from music21 import clef

    measures = list(part.getElementsByClass("Measure"))
    current = None
    for m in measures:
        pitches = [p.midi for n in m.recurse().notes for p in n.pitches]
        if current is None:
            first = m.clef or clef.TrebleClef()
            current = "bass" if isinstance(first, clef.BassClef) else "treble"
            continue
        new = _next_clef(current, pitches)
        if new != current:
            m.clef = clef.TrebleClef() if new == "treble" else clef.BassClef()
            current = new


# --------------------------------------------------------------------------- score


REMIND_AFTER = 8  # bars before a hand move that was already marked gets its fingering printed again


def _same_position(prev: dict[int, int], cur: dict[int, int], hand: str) -> bool:
    """Can the hand play `cur` right after `prev` without shifting or crossing: every pair of
    fingers within its comfortable span, no finger passing the thumb, no finger changing key?"""
    from .fingering import SPANS

    sign = -1 if hand == "L" else 1
    for pa, fa in prev.items():
        for pb, fb in cur.items():
            if fa == fb:
                if pa != pb:
                    return False
                continue
            lo_f, lo_q, hi_f, hi_q = (fa, sign * pa, fb, sign * pb) if fa < fb else (fb, sign * pb, fa, sign * pa)
            _, min_comf, _, _, max_comf, _ = SPANS[(lo_f, hi_f)]
            d = hi_q - lo_q
            if d < max(min_comf, 0) or d > max_comf:
                return False
    return True


def fingering_marks(events: list[Event], hand: str, mode: str, seconds_per_ql: float,
                    beat_ql: Fraction, bar_ql: Fraction = Fraction(4)) -> list[dict[int, int] | None]:
    """Fingering to print for each event ({pitch: finger}), or None.

    mode "all" prints every fingered event; "none" prints nothing. "smart" prints a fingering
    where the hand has to move (a shift, a stretch beyond the comfortable span, a thumb
    crossing) or comes back in after a rest, and does not repeat a move it has already marked
    in the last REMIND_AFTER bars, which is how fingering is marked in printed music.
    """
    from .fingering import finger_events

    out: list[dict[int, int] | None] = [None] * len(events)
    if mode == "none":
        return out
    idx = [i for i, ev in enumerate(events) if ev.tie in (None, "start")]  # tied pieces are held, not struck
    fingered = finger_events([(float(events[i].start), events[i].pitches) for i in idx], hand, seconds_per_ql)
    by_event = dict(zip(idx, fingered))
    prev, prev_end = None, None
    marked: dict[tuple, Fraction] = {}  # move -> where it was last printed
    for i, ev in enumerate(events):
        if i not in by_event:  # continuation of a tie
            prev_end = ev.end
            continue
        f = by_event[i]
        if f is None:
            prev = None
            continue
        if mode == "all":
            out[i] = f
        else:
            rested = prev_end is not None and ev.start - prev_end >= beat_ql
            if prev is None or rested:
                move = ("enter", tuple(sorted(f.values())))
                show = True
            else:
                show = not _same_position(prev, f, hand)
                # the same fingers making the same move (transposed or not) need no second mark
                move = (tuple(sorted(prev.values())), tuple(sorted(f.values())), min(f) - min(prev))
            if show and (move not in marked or ev.start - marked[move] >= REMIND_AFTER * bar_ql):
                out[i] = f
                marked[move] = ev.start
        prev, prev_end = f, ev.end
    return out
    idx = [i for i, ev in enumerate(events) if ev.tie in (None, "start")]  # tied pieces are held, not struck
    fingered = finger_events([(float(events[i].start), events[i].pitches) for i in idx], hand, seconds_per_ql)
    by_event = dict(zip(idx, fingered))
    prev_pos, prev_end = None, None
    for i, ev in enumerate(events):
        if i not in by_event:  # continuation of a tie
            prev_end = ev.end
            continue
        f = by_event[i]
        if f is None:
            prev_pos = None
            continue
        pos = hand_position(f, hand)
        rested = prev_end is not None and ev.start - prev_end >= beat_ql
        if mode == "all" or prev_pos is None or rested or abs(pos - prev_pos) >= 2:
            out[i] = f
        prev_pos, prev_end = pos, ev.end
    return out


def build_score(
    rh_notes: list[QNote],
    lh_notes: list[QNote],
    meter: Meter,
    bpm: float,
    title: str,
    key_override: str | None = None,
    pedal: list[tuple[float, float]] | None = None,
    fingering: str = "smart",
    rolled: set | None = None,
):
    """`pedal`: sustain-pedal-down spans in beats (same origin as the notes).
    `fingering`: "smart", "all" or "none". `rolled`: (hand "R"/"L", start beat) of chords to arpeggiate."""
    from music21 import (
        articulations,
        chord,
        expressions,
        clef,
        duration,
        key,
        layout,
        metadata,
        meter as m21meter,
        note,
        stream,
        tempo,
        tie,
    )

    all_notes = rh_notes + lh_notes
    k = key.Key(key_override) if key_override else detect_key(all_notes)
    names = spelling_for_key(k)

    bar_ql = Fraction(4 * meter.numerator, meter.denominator)
    total_ql = max(q.end for q in all_notes) * meter.beat_ql
    total_ql = Fraction(math.ceil(total_ql / bar_ql)) * bar_ql

    score = stream.Score()
    score.metadata = metadata.Metadata()
    score.metadata.title = title
    score.metadata.composer = "Automatic transcription (yt2sheet)"

    parts = []
    for name, qnotes, default_clef in (("RH", rh_notes, "treble"), ("LH", lh_notes, "bass")):
        part = stream.PartStaff(id=name)
        median = sorted(q.pitch for q in qnotes)[len(qnotes) // 2] if qnotes else None
        use = default_clef
        if median is not None and default_clef == "bass" and median >= 62:
            use = "treble"
        elif median is not None and default_clef == "treble" and median < 54:
            use = "bass"
        part.insert(0, clef.TrebleClef() if use == "treble" else clef.BassClef())
        part.insert(0, key.KeySignature(k.sharps))
        part.insert(0, m21meter.TimeSignature(str(meter)))
        if name == "RH":
            mm = tempo.MetronomeMark(number=round(bpm), referent=duration.Duration(float(meter.beat_ql)))
            part.insert(0, mm)
        if name == "LH" and pedal:
            span = float(max(q.end for q in all_notes)) or 1.0
            if sum(b - a for a, b in pedal) / span > 0.5:  # pedalled throughout
                ped = expressions.TextExpression("con Ped.")
                ped.style.fontStyle = "italic"
                part.insert(0, ped)

        hand = name[0]
        events = staff_events(qnotes, meter, pedal, melody=hand == "R")
        seconds_per_ql = 60.0 / bpm / float(meter.beat_ql)
        marks = fingering_marks(events, hand, fingering, seconds_per_ql, meter.beat_ql, bar_ql)
        cursor = Fraction(0)
        for ev, mark in zip(events, marks):
            if ev.start > cursor:
                r = note.Rest()
                r.duration = duration.Duration(ev.start - cursor)
                part.insert(cursor, r)
            for g in ev.graces:  # ornaments: small slashed grace notes before the main note
                gn = note.Note(_pitch(g, names), type="eighth").getGrace()
                gn.duration.slash = True  # acciaccatura: small slashed 8th
                part.insert(ev.start, gn)
            if len(ev.pitches) == 1:
                el = note.Note(_pitch(ev.pitches[0], names))
            else:
                el = chord.Chord([_pitch(p, names) for p in ev.pitches])
            el.duration = duration.Duration(ev.end - ev.start)
            el.volume.velocity = ev.velocity
            if ev.tie:
                el.tie = tie.Tie(ev.tie)
            if mark:
                for p in ev.pitches:  # ascending, matching the chord's note order
                    fa = articulations.Fingering(mark[p])
                    fa.placement = "above" if hand == "R" else "below"
                    el.articulations.append(fa)
            if rolled and len(ev.pitches) > 1 and ev.tie in (None, "start") and (hand, ev.start / meter.beat_ql) in rolled:
                el.expressions.append(expressions.ArpeggioMark())
            part.insert(ev.start, el)
            cursor = ev.end
        if total_ql > cursor:
            r = note.Rest()
            r.duration = duration.Duration(total_ql - cursor)
            part.insert(cursor, r)
        parts.append(part)

    for part in parts:
        measured = part.makeNotation()
        _assign_clefs(measured)
        score.insert(0, measured)
    measured = list(score.parts)
    score.insert(0, layout.StaffGroup(measured, name="Piano", abbreviation="Pno.", symbol="brace"))
    return score, k
