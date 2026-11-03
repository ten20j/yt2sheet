# yt2sheet

Piano sheet music from a YouTube link or an audio file.

```
YouTube link / mp3 ──► audio ──► Transkun (piano transcription) ──► notes + pedal
        ──► beat tracking ──► rhythm quantization ──► hand split ──► key & spelling
        ──► MusicXML ──► LilyPond ──► PDF
```

Outputs, per piece:

| file | what it is |
|---|---|
| `<title>.pdf` | engraved grand-staff score |
| `<title>.musicxml` | the same score, editable in MuseScore / Finale / Sibelius / Dorico |
| `<title>-score.mid` | the quantized score as MIDI |
| `<title>-performance.mid` | the raw transcription (exact timing, velocities, sustain pedal) |

## Install

You need Python 3.10+ and `ffmpeg` on your PATH.

```bash
python -m venv .venv
```

```bash
.venv/bin/pip install -e ".[web]"
```

That pulls in PyTorch, the Transkun model (its weights ship inside the pip package), LilyPond
(the `lilypond` pip package bundles the binaries) and Verovio as a fallback engraver.

For YouTube links, yt-dlp also needs a JavaScript runtime. [Deno](https://deno.com) is the
default; Node.js or Bun also work and are picked up automatically if installed.

## Use

```bash
.venv/bin/yt2sheet "https://www.youtube.com/watch?v=..." -o output
```

```bash
.venv/bin/yt2sheet recording.mp3 --title "My Piece" --time-signature 3/4
```

Or run the local web page (paste a link or upload a file, watch progress, download the results):

```bash
.venv/bin/yt2sheet-web
```

then open http://127.0.0.1:5000.

A 4-minute piece takes about 2 minutes on a 4-core CPU; with an NVIDIA GPU add `--device cuda`.

### Options that matter most

| option | when to use it |
|---|---|
| `--time-signature 3/4` | the meter is not 4/4 (also 2/4, 6/8, 12/8, 2/2, ...) |
| `--bpm 72` | the tempo came out doubled/halved; give roughly the beat you tap along to |
| `--start 30 --end 95` | transcribe only part of the recording (seconds) |
| `--key E-` | override key detection (`E-` = E-flat major, `f#` = F-sharp minor) |
| `--no-triplets` | rubato made straight rhythms come out as triplets |
| `--grid 8` | only use 8th notes and longer (simpler, less exact) |
| `--split-point 60` | fixed hand split at middle C instead of the automatic one |
| `--downbeat 1` | the barlines are shifted; this moves the first downbeat by N beats |
| `--cookies cookies.txt` | YouTube asks you to sign in |
| `--engine verovio` | use Verovio instead of LilyPond for the PDF |

A `.mid` file can be given instead of audio to skip transcription, which is handy for re-engraving
with different options: `yt2sheet output/my-piece-performance.mid --time-signature 3/4`.

## How it works

- **Transcription**: [Transkun](https://github.com/Yujia-Yan/Transkun) (Yan & Duan, 2024), a
  state-of-the-art piano model trained on MAESTRO. It returns notes with onsets, releases and velocities,
  plus the sustain pedal.
- **Beats**: librosa's dynamic-programming beat tracker on an onset envelope built from the
  transcribed notes. The tracker's 2-vs-3 grouping mistakes (e.g. triplet arpeggios) are corrected
  by checking at which beat length the same pitches recur. Fast continuous motion in even meters is
  written as 8ths at half tempo.
- **Rhythm**: onsets are mapped onto the beat grid (following tempo changes) and snapped to a
  metrical grid that prefers simpler rhythms; each beat may switch to triplets when they clearly fit
  better. Long notes' ends snap more coarsely (key releases are imprecise), and notes released under
  the sustain pedal are written to the next note instead of with rests.
- **Barlines**: the downbeat is placed where accented, long bass notes fall.
- **Hands**: a cost model per chord: distance from each hand's recent notes, a hand-span limit
  that includes notes still held, and a preference against splitting close chords.
- **Notation**: key detection (Krumhansl-Schmuckler via music21), key-aware note spelling, then
  music21 for measures, ties and beams, and LilyPond for engraving.

## Limits

Automatic transcription gives a solid first draft rather than a publication-ready score. It works
best on solo piano recordings. Expect to fix things in MuseScore using the `.musicxml`:

- Rubato-heavy or free-time passages (intros, cadenzas) get awkward rhythms.
- Each staff is one chord stream; a held melody note under moving inner voices is shortened.
- The meter is not detected; pass `--time-signature` for anything other than 4/4.
- Non-piano instruments, vocals and heavy reverb reduce transcription accuracy.

Only transcribe music you have the right to use. Downloading from YouTube is subject to its terms of service.

## Tests

```bash
.venv/bin/pip install pytest
```

```bash
.venv/bin/python -m pytest
```

The tests render small ground-truth pieces (an Alberti-bass minuet, a triplet nocturne, a waltz, a
fast arpeggio ballad) as humanized performances with timing jitter, early releases and tempo drift.
They check that onsets, triplets, hands and key come back correctly. `tests/make_test_piece.py`
also synthesizes audio with FluidSynth for a full audio-to-PDF check.
