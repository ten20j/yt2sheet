"""Following the meter through rubato: barlines stay with the bass notes."""

from __future__ import annotations

from yt2sheet.meter import metric_grid
from yt2sheet.notes import Note
from yt2sheet.rhythm import Meter, _cluster_onsets, quantize, trim_leading_bars

METER = Meter.parse("4/4")
EIGHTH = 60 / 72 / 2  # flowing 8ths at quarter = 72
CHORDS = [(36, 43, 48, 52, 55), (33, 40, 45, 48, 52)]  # C major, A minor broken chords


def flowing_piece(bars=8, pause_after_bar=None, intro=()):
    """8th-note broken chords (bass on beat 1, up and back down) under a melody note per half
    bar. `pause_after_bar`: the player lingers one extra 8th before that bar (rubato).
    `intro`: times of free-time chords played before the piece starts."""
    notes = [Note(t, t + 1.5, p, 45) for t in intro for p in (43, 50, 59)]
    t0 = (intro[-1] + 3.1) if intro else 0.5
    for bar in range(bars):
        chord = CHORDS[bar % 2]
        late = EIGHTH if pause_after_bar is not None and bar >= pause_after_bar else 0.0
        for k, p in enumerate(chord + chord[-2:0:-1]):  # 8 notes: up 5, down 3
            t = t0 + (bar * 8 + k) * EIGHTH + late
            notes.append(Note(t, t + 0.3, p, 36))
        for k in (0, 4):
            t = t0 + (bar * 8 + k) * EIGHTH + late
            notes.append(Note(t, t + 1.4, 72 + (bar + k) % 5, 70))
    return sorted(notes, key=lambda n: n.start)


def bass_positions(notes, elastic):
    grid = metric_grid(notes, METER, elastic=elastic)
    q, _ = trim_leading_bars(quantize(notes, grid, METER), METER)
    return grid, [x.start % 4 for x in q if x.pitch <= 36]


def test_steady_playing_needs_no_correction():
    grid, bass = bass_positions(flowing_piece(), elastic=True)
    assert grid.pulses_per_beat == 2 and not grid.edits
    assert all(b == 0 for b in bass)


def test_barlines_recover_after_a_rubato_pause():
    notes = flowing_piece(pause_after_bar=4)
    grid, bass = bass_positions(notes, elastic=True)
    assert grid.edits  # the extra pulse was taken out
    assert all(b == 0 for b in bass)  # every bass note is still a downbeat
    _, strict = bass_positions(notes, elastic=False)
    assert any(b != 0 for b in strict)  # without it, the bars after the pause are shifted


def test_free_time_chords_land_on_downbeats():
    notes = flowing_piece(intro=(0.9, 3.6, 6.5))
    grid = metric_grid(notes, METER)
    q, _ = trim_leading_bars(quantize(notes, grid, METER), METER)
    chord_starts = sorted({x.start for x in q if x.pitch == 59})
    assert len(chord_starts) == 3 and all(s % 4 == 0 for s in chord_starts)


def test_hands_slightly_apart_are_one_event():
    notes = [Note(1.00, 1.5, 43, 40), Note(1.09, 1.6, 71, 60),  # bass leads the melody by 90 ms
             Note(2.00, 2.2, 60, 50), Note(2.09, 2.3, 62, 50)]  # two quick notes in one register
    onsets = _cluster_onsets(notes)
    assert onsets[0] == onsets[1]
    assert onsets[2] != onsets[3]
