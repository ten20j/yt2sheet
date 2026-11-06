"""Hand split and playability fixes."""

from __future__ import annotations

from fractions import Fraction as F

import pytest

from yt2sheet.hands import split_hands
from yt2sheet.notes import QNote
from yt2sheet.playability import make_playable, report
from yt2sheet.rhythm import Meter, align_to_barlines, choose_beat_octave, quantize, track_beats

from pieces import PIECES, perform


def q(pitch, start=0, end=1):
    return QNote(F(start), F(end), pitch, 80)


def test_too_wide_rh_chord_hands_a_note_to_a_free_lh():
    rh, lh, fixes = make_playable([q(55), q(64), q(72)], [q(48)])  # RH spans G3-C5 (17)
    assert sorted(n.pitch for n in rh) == [64, 72]
    assert sorted(n.pitch for n in lh) == [48, 55]
    assert fixes.moved == 1 and not fixes.rolled


def test_octave_doubling_is_dropped_but_never_the_melody():
    # RH: C4 E4 G4 C5 E5 (16 semitones); the LH is busy far below, so it cannot help
    rh, lh, fixes = make_playable([q(60), q(64), q(67), q(72), q(76)], [q(28), q(40)])
    pitches = sorted(n.pitch for n in rh)
    assert 76 in pitches and pitches[-1] - pitches[0] <= 12
    assert fixes.dropped >= 1


def test_unfixable_spread_is_rolled():
    rh, lh, fixes = make_playable([q(60), q(67), q(76)], [q(28), q(40)])  # C-G-E over a 10th, no doublings
    assert ("R", F(0)) in fixes.rolled
    assert len(rh) == 3


def test_never_more_than_five_notes_per_hand():
    rh, lh, _ = make_playable([q(p) for p in (60, 62, 64, 65, 67, 69, 71)], [q(30)])
    assert len(rh) <= 5


@pytest.mark.parametrize("name", list(PIECES))
def test_split_is_playable(name):
    piece = PIECES[name]
    meter = Meter.parse(piece[0])
    notes = perform(piece, seed=0)
    beats = choose_beat_octave(notes, track_beats(notes), meter)
    qn, _ = align_to_barlines(quantize(notes, beats, meter), meter)
    rh, lh, _ = make_playable(*split_hands(qn))
    r = report(rh, lh)
    assert r["RH chords wider than reach"] == r["LH chords wider than reach"] == 0
    assert r["hand crossings"] == 0


def melody_dip_piece():
    """LH broken chords (soft) under a louder melody that dips to G3, alone on its beat,
    right after the LH played F#3 a semitone below it (bar 9 of the user's piece)."""
    notes = []
    for bar in range(4):
        o = bar * 4
        for t, p in ((0, 43), (0.5, 50), (1, 54), (2, 43), (2.5, 50), (3, 54)):
            notes.append(QNote(F(o) + F(t), F(o) + F(t) + F(1, 2), p, 34))
        for t, p, d in ((0, 62, 1), (1.5, 55, 0.5), (2, 59, 1), (3.5, 60, 0.5)):
            notes.append(QNote(F(o) + F(t), F(o) + F(t) + F(d), p, 62))
    return notes


def test_melody_stays_in_the_right_hand_when_it_dips_low():
    rh, lh = split_hands(melody_dip_piece())
    assert all(n.velocity == 62 for n in rh) and all(n.velocity == 34 for n in lh)


def test_melody_dip_test_detects_the_old_behaviour(monkeypatch):
    import yt2sheet.hands as hands

    monkeypatch.setattr(hands, "MELODY_WEIGHT", 0.0)
    rh, lh = split_hands(melody_dip_piece())
    assert any(n.velocity == 62 for n in lh)  # without the melody cue, the G3s go to the LH
