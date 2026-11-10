"""Rhythm simplification and grace notes."""

from __future__ import annotations

import numpy as np

from yt2sheet.fingering import finger_events
from yt2sheet.notes import Note
from yt2sheet.ornaments import extract_graces
from yt2sheet.meter import metric_grid
from yt2sheet.rhythm import Meter, quantize, trim_leading_bars

from pieces import PIECES, perform

METER = Meter.parse("4/4")


def test_rubato_snaps_to_simple_rhythms_in_simple_style():
    # straight 8ths at 72 BPM over a steady quarter-note bass; a few melody notes are pushed
    # early or late by 0.15-0.2 beats, as in expressive playing
    spb = 60 / 72
    beats = np.arange(-2, 40) * spb
    notes = [Note(b * spb, b * spb + 0.3, 43, 70) for b in range(32)]
    push = {5: -0.17, 11: 0.18, 18: -0.2, 26: 0.16, 33: -0.15}
    for k in range(64):
        x = k / 2 + push.get(k, 0.0)
        notes.append(Note(x * spb, x * spb + 0.25, 72 + (k % 5), 80))
    simple = quantize(notes, beats, METER, style="simple")
    detailed = quantize(notes, beats, METER, style="detailed")
    assert all(q.start.denominator in (1, 2) for q in simple)
    assert any(q.start.denominator == 4 for q in detailed)


def test_cleanly_played_dotted_rhythm_survives_simple_style():
    notes = perform(PIECES["minuet"], seed=0)
    q, _ = trim_leading_bars(quantize(notes, metric_grid(notes, METER), METER, style="simple"), METER)
    # bar 7: F#5 dotted 8th, E5 16th (beat 2 + 3/4 of a beat, counting from 0)
    assert any(n.pitch == 76 and n.start == 24 + 2 + 0.75 for n in q)


def test_ornament_becomes_a_grace_note():
    notes = [Note(1.00, 1.11, 62, 45), Note(1.12, 1.9, 64, 60), Note(1.12, 1.9, 43, 50)]
    out, n = extract_graces(notes)
    assert n == 1
    main = next(x for x in out if x.pitch == 64)
    assert main.graces == (62,) and len(out) == 2


def test_fast_scale_run_is_not_turned_into_grace_notes():
    run = [Note(1 + 0.1 * k, 1 + 0.1 * k + 0.09, p, 70) for k, p in enumerate([60, 62, 64, 65, 67, 69, 71])]
    run.append(Note(1.7, 2.5, 72, 70))  # held top note
    out, n = extract_graces(run)
    assert n == 0 and len(out) == len(run)


def test_left_hand_leap_lands_on_the_little_finger():
    # thumb on G3, then a leap down to a C2-G2-E3 broken chord: 1, then 5-2-1
    events = [(0.0, [50]), (0.5, [55]), (1.0, [36]), (1.5, [43]), (2.0, [52])]
    f = finger_events(events, "L", seconds_per_unit=0.42)
    assert [f[i][p] for i, (_, (p,)) in enumerate(events)][2:] == [5, 2, 1]
    assert f[1][55] != f[2][36]
