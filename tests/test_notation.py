"""End-to-end tests of the notation stage: humanized performance -> quantized, hand-split score."""

from __future__ import annotations

import pytest

from yt2sheet.hands import split_hands
from yt2sheet.rhythm import Meter, align_to_barlines, choose_beat_octave, quantize, tempo_bpm, track_beats
from yt2sheet.score import build_score

from pieces import BALLAD, PIECES, perform, truth_notes


def notate(piece, seed):
    meter = Meter.parse(piece[0])
    notes = perform(piece, seed=seed)
    beats = choose_beat_octave(notes, track_beats(notes), meter)
    q, _ = align_to_barlines(quantize(notes, beats, meter), meter)
    return meter, beats, q


def onset_accuracy(piece, q) -> float:
    truth = {(p, s) for p, s, _ in truth_notes(piece[2]) + truth_notes(piece[3])}
    got = {(n.pitch, n.start) for n in q}
    return len(truth & got) / len(truth)


@pytest.mark.parametrize("name", list(PIECES))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_onsets_recovered(name, seed):
    piece = PIECES[name]
    _, _, q = notate(piece, seed)
    assert onset_accuracy(piece, q) >= 0.95, name


@pytest.mark.parametrize("name", list(PIECES))
def test_hands(name):
    piece = PIECES[name]
    _, _, q = notate(piece, 0)
    rh, lh = split_hands(q)
    truth_rh = {(p, s) for p, s, _ in truth_notes(piece[2])}
    got_rh = {(n.pitch, n.start) for n in rh}
    got_lh = {(n.pitch, n.start) for n in lh}
    correct = len(truth_rh & got_rh) + len(got_lh - truth_rh)
    assert correct / (len(rh) + len(lh)) >= 0.95


def test_no_spurious_triplets():
    _, _, q = notate(PIECES["minuet"], 0)
    assert all(n.start.denominator in (1, 2, 4) for n in q)


def test_triplets_detected():
    _, _, q = notate(PIECES["nocturne"], 0)
    # 12 arpeggio notes per bar, 8 of them off the beat, over 4 bars
    assert sum(n.start.denominator == 3 for n in q) >= 30 * 0.75


@pytest.mark.parametrize("name", list(PIECES))
def test_score_builds_and_exports(name, tmp_path):
    piece = PIECES[name]
    meter, beats, q = notate(piece, 0)
    rh, lh = split_hands(q)
    score, key = build_score(rh, lh, meter, 100, name)
    out = tmp_path / f"{name}.musicxml"
    score.write("musicxml", fp=str(out))
    assert out.stat().st_size > 1000
    # the nocturne's Neapolitan (D major) chord makes C# minor vs F# minor genuinely ambiguous
    expected_sharps = {"minuet": {2}, "nocturne": {3, 4}, "waltz": {1}}[name]
    assert key.sharps in expected_sharps


def test_fast_continuous_motion_is_notated_as_eighths():
    meter, beats, q = notate(BALLAD, 0)
    assert 60 <= tempo_bpm(beats) <= 85
    assert onset_accuracy(BALLAD, q) >= 0.95


def test_clef_changes_follow_register():
    from yt2sheet.score import _next_clef

    assert _next_clef("bass", [67, 71, 74, 79]) == "treble"  # LH climbs high
    assert _next_clef("treble", [67, 60, 62]) == "treble"  # hysteresis: stays
    assert _next_clef("treble", [43, 47, 50, 55]) == "bass"
    assert _next_clef("bass", [48, 60, 64]) == "bass"
    assert _next_clef("bass", []) == "bass"
