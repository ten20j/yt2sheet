"""Hand split and playability fixes."""

from __future__ import annotations

from fractions import Fraction as F

import pytest

from yt2sheet.hands import split_hands
from yt2sheet.notes import QNote
from yt2sheet.playability import make_playable, report
from yt2sheet.meter import metric_grid
from yt2sheet.rhythm import Meter, quantize, trim_leading_bars

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
    qn, _ = trim_leading_bars(quantize(notes, metric_grid(notes, meter), meter), meter)
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


def broken_chord_under_melody():
    """LH broken chords (soft 8ths, climbing to D4) that keep going while a louder RH
    melody plays now and then in the same register (bars 19-20 of the user's piece)."""
    notes = []
    for bar in range(4):
        o = bar * 4
        for t, p in enumerate((36, 43, 52, 55, 55, 62, 55, 48)):  # C2 G2 E3 G3 G3 D4 G3 C3
            notes.append(QNote(F(o) + F(t, 2), F(o) + F(t + 1, 2), p, 33))
        for t, p in ((0, 71), (1.5, 64), (3, 67), (3.5, 69)):  # melody B4 E4 G4 A4
            notes.append(QNote(F(o) + F(t), F(o) + F(t) + F(1, 2), p, 58))
    return notes


def test_left_hand_keeps_its_broken_chord_instead_of_alternating_hands():
    rh, lh = split_hands(broken_chord_under_melody())
    assert all(n.velocity == 58 for n in rh)  # only the melody in the RH
    assert all(n.velocity == 33 for n in lh)


def test_broken_chord_test_detects_the_old_behaviour(monkeypatch):
    import yt2sheet.hands as hands

    monkeypatch.setattr(hands, "HOCKET_WEIGHT", 0.0)
    rh, lh = split_hands(broken_chord_under_melody())
    assert any(n.velocity == 33 for n in rh)  # without the cost, the RH steals figure notes


def rocking_piece():
    """A slow, louder melody over a bass and a soft rocking figure around middle C: the
    figure is the left hand's, even though it is an octave above the bass (bars 34-35)."""
    notes = []
    for bar in range(4):
        o = F(bar * 4)
        notes += [QNote(o, o + 2, 64, 62), QNote(o + 2, o + 4, 67, 62)]  # melody E4, G4
        notes += [QNote(o, o + F(1, 2), 36, 34), QNote(o + F(1, 2), o + 1, 43, 34)]  # bass C2, G2
        for t, ps in ((1, (55,)), (1.5, (60, 62)), (2, (55,)), (2.5, (60,)), (3, (55,)), (3.5, (60, 62))):
            notes += [QNote(o + F(t), o + F(t) + F(1, 2), p, 33) for p in ps]
    return notes


def test_soft_figure_below_the_melody_goes_to_the_left_hand():
    rh, lh = split_hands(rocking_piece())
    assert {n.pitch for n in rh} == {64, 67}
    assert all(n.velocity < 60 for n in lh)


def no_melody_piece():
    """The right hand ends a phrase and falls silent; then a bass note and a soft broken chord
    around middle C, with nothing to sing above it (bars 7-8)."""
    notes = [QNote(F(0), F(1), 60, 64), QNote(F(1), F(2), 64, 64)]
    for bar in (1, 2):
        o = F(bar * 4)
        notes.append(QNote(o, o + F(1, 2), 43, 30))
        notes += [QNote(o + F(k, 2), o + F(k + 1, 2), p, 30)
                  for k, p in enumerate([55, 59, 62, 55, 59, 62, 59], start=1)]
    return notes


def test_without_a_melody_the_left_hand_plays_the_accompaniment(monkeypatch):
    import yt2sheet.hands as hands

    rh, lh = split_hands(no_melody_piece())
    assert all(n.velocity == 64 for n in rh)  # only the phrase it sang
    monkeypatch.setattr(hands, "NO_MELODY", 0.0)
    rh, lh = split_hands(no_melody_piece())
    assert any(n.velocity == 30 for n in rh)  # without the rule the nearer hand grabs the figure


def restruck_piece():
    """The right hand plays B3 + E4; the left hand's broken chord rises towards them and the
    same two keys sound again, softer, two beats later (bar 15)."""
    notes = []
    for bar in range(3):
        o = F(bar * 4)
        notes += [QNote(o, o + 2, 59, 40), QNote(o, o + 2, 64, 60)]
        notes += [QNote(o + F(k, 2), o + F(k + 1, 2), p, v)
                  for k, (p, v) in enumerate([(36, 33), (43, 25), (52, 35), (55, 33)])]
        notes += [QNote(o + 2, o + 4, 59, 30), QNote(o + 2, o + 4, 64, 42)]
    return notes


def test_a_repeated_note_stays_in_the_hand_that_played_it(monkeypatch):
    import yt2sheet.hands as hands

    def restruck_in_lh():
        rh, lh = split_hands(restruck_piece())
        return len([n for n in lh if n.start % 4 == 2])

    with_rule = restruck_in_lh()
    monkeypatch.setattr(hands, "SAME_KEY", 0.0)
    without_rule = restruck_in_lh()  # the left hand takes over keys the right hand just played
    assert with_rule <= 1 and without_rule >= 3


def test_a_soft_note_that_continues_the_melody_by_step_stays_in_the_right_hand():
    # waltz: melody G4, F#4, E4 over oom-pah-pah; the F#4 is played as softly as the chords
    notes = []
    for bar in range(4):
        o = F(bar * 3)
        notes += [QNote(o, o + 1, 38, 50), QNote(o + 1, o + 2, 54, 52), QNote(o + 1, o + 2, 57, 52),
                  QNote(o + 2, o + 3, 54, 52), QNote(o + 2, o + 3, 57, 52)]
        notes += [QNote(o, o + 1, 67, 80), QNote(o + 1, o + 2, 66, 52), QNote(o + 2, o + 3, 64, 80)]
    rh, lh = split_hands(notes)
    assert sorted({n.pitch for n in rh}) == [64, 66, 67]
    assert not any(n.pitch == 66 for n in lh)
