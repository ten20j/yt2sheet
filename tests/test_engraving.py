"""How notes are written: pedal-sustained lengths and where finger numbers are printed."""

from __future__ import annotations

from fractions import Fraction as F

from yt2sheet.notes import QNote
from yt2sheet.rhythm import Meter
from yt2sheet.score import fingering_marks, staff_events

METER = Meter.parse("4/4")
PEDAL = [(0.0, 12.0)]  # pedal down throughout


def ends(notes, melody):
    return [(e.start, e.end) for e in staff_events(notes, METER, PEDAL, melody=melody)]


def test_pedalled_accompaniment_is_not_tied_over_the_barline():
    # released on beat 4.5, next note in this hand on beat 2 of the next bar
    notes = [QNote(F(3), F(7, 2), 43, 40), QNote(F(5), F(6), 48, 40), QNote(F(6), F(7), 50, 40)]
    assert ends(notes, melody=False)[0] == (F(3), F(4))  # stops at the barline (a rest follows)
    assert ends(notes, melody=True)[0] == (F(3), F(5))  # a melody note sings on, tied


def test_pedalled_accompaniment_gets_a_plain_length():
    # an 8th after the beat, then nothing until the next bar: not a double-dotted half
    notes = [QNote(F(0), F(1, 2), 36, 40), QNote(F(1, 2), F(1), 43, 40), QNote(F(4), F(5), 36, 40)]
    assert ends(notes, melody=False)[1] == (F(1, 2), F(2))  # dotted quarter up to the half bar


def test_a_key_that_is_really_held_keeps_its_tie():
    notes = [QNote(F(3), F(6), 43, 40), QNote(F(6), F(7), 48, 40)]
    assert ends(notes, melody=False)[0] == (F(3), F(6))


def test_last_note_rings_on():
    notes = [QNote(F(0), F(1), 43, 40), QNote(F(3), F(7, 2), 36, 40)]
    assert ends(notes, melody=False)[-1][1] > F(4)


def test_smart_fingering_marks_a_repeated_figure_once():
    events = staff_events(
        [QNote(F(k, 2), F(k + 1, 2), p, 40) for k, p in enumerate([48, 55, 52, 55] * 8)], METER, melody=False
    )
    smart = fingering_marks(events, "L", "smart", 0.8, METER.beat_ql)
    every = fingering_marks(events, "L", "all", 0.8, METER.beat_ql)
    assert sum(m is not None for m in every) == len(events)
    assert 1 <= sum(m is not None for m in smart) <= 3
    assert all(m is None for m in fingering_marks(events, "L", "none", 0.8, METER.beat_ql))
