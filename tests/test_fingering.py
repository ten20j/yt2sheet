"""Fingering engine against fingerings every piano teacher would write."""

from __future__ import annotations

from yt2sheet.fingering import finger_events


def fingers(pitches, hand, chords=False):
    events = [(i * 0.5, p if chords else [p]) for i, p in enumerate(pitches)]
    out = finger_events(events, hand, seconds_per_unit=0.5)
    if chords:
        return [tuple(f[p] for p in sorted(ch)) for f, ch in zip(out, pitches)]
    return [f[p] for f, p in zip(out, pitches)]


C_MAJOR = [60, 62, 64, 65, 67, 69, 71, 72]


def test_rh_c_major_scale():
    assert fingers(C_MAJOR, "R") == [1, 2, 3, 1, 2, 3, 4, 5]
    assert fingers(C_MAJOR[::-1], "R") == [5, 4, 3, 2, 1, 3, 2, 1]


def test_lh_c_major_scale():
    low = [p - 12 for p in C_MAJOR]
    assert fingers(low, "L") == [5, 4, 3, 2, 1, 3, 2, 1]
    assert fingers(low[::-1], "L") == [1, 2, 3, 1, 2, 3, 4, 5]


def test_rh_g_and_f_major_scales_keep_thumb_off_black_keys():
    g = [67, 69, 71, 72, 74, 76, 78, 79]
    assert fingers(g, "R") == [1, 2, 3, 1, 2, 3, 4, 5]
    f = [65, 67, 69, 70, 72, 74, 76, 77]
    assert fingers(f, "R")[:7] == [1, 2, 3, 4, 1, 2, 3]


def test_five_finger_position():
    assert fingers([60, 62, 64, 65, 67], "R") == [1, 2, 3, 4, 5]
    assert fingers([48, 50, 52, 53, 55], "L") == [5, 4, 3, 2, 1]


def test_triads():
    assert fingers([[60, 64, 67]], "R", chords=True) == [(1, 3, 5)]
    assert fingers([[48, 52, 55]], "L", chords=True) == [(5, 3, 1)]


def test_alberti_bass():
    assert fingers([48, 55, 52, 55, 48, 55, 52, 55], "L") == [5, 1, 3, 1, 5, 1, 3, 1]


def test_no_same_finger_on_stepwise_notes():
    melody = [64, 62, 60, 62, 64, 64, 64, 62, 62, 62, 64, 67, 67]
    fs = fingers(melody, "R")
    for (p1, f1), (p2, f2) in zip(zip(melody, fs), zip(melody[1:], fs[1:])):
        if p1 != p2:
            assert f1 != f2


def test_too_many_notes_is_left_unfingered():
    out = finger_events([(0, [48, 52, 55, 60, 64, 67])], "R")
    assert out == [None]


MAJOR = [0, 2, 4, 5, 7, 9, 11, 12]
# standard one-octave fingerings, (root, RH ascending, LH ascending)
STANDARD = {
    "C": (60, [1, 2, 3, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 3, 2, 1]),
    "G": (67, [1, 2, 3, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 3, 2, 1]),
    "D": (62, [1, 2, 3, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 3, 2, 1]),
    "A": (69, [1, 2, 3, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 3, 2, 1]),
    "E": (64, [1, 2, 3, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 3, 2, 1]),
    "F": (65, [1, 2, 3, 4, 1, 2, 3, 4], [5, 4, 3, 2, 1, 3, 2, 1]),
    "Bb": (70, [2, 1, 2, 3, 1, 2, 3, 4], None),
    "Eb": (63, None, None),
    "Ab": (68, None, None),
}


def test_major_scales():
    for name, (root, rh, lh) in STANDARD.items():
        scale = [root + i for i in MAJOR]
        got_rh = fingers(scale, "R")
        got_lh = fingers([p - 24 for p in scale], "L")
        if rh:
            assert got_rh == rh, name
        if lh:
            assert got_lh == lh, name
        # whatever the exact choice, the right-hand thumb never lands on a black key
        assert not any(f == 1 and p % 12 in (1, 3, 6, 8, 10) for p, f in zip(scale, got_rh)), name


def test_common_chord_shapes():
    def chord(ps, hand):
        f = finger_events([(0, ps)], hand)[0]
        return [f[p] for p in ps]

    assert chord([43, 50], "L") == [5, 1]  # open fifth: pinky on the bass
    assert chord([36, 43, 52], "L") == [5, 2, 1]  # spread 1-5-10 accompaniment chord
    assert chord([60, 72], "R") == [1, 5]  # octave
    assert chord([64, 67, 72], "R") == [1, 2, 5]  # first-inversion triad
