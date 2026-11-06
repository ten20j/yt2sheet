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
| `--rhythm detailed` | write every 16th as played (default `simple`: plain rhythms, expressive timing ignored) |
| `--no-grace-notes` | write ornaments as ordinary notes instead of small grace notes |
| `--no-triplets` | rubato made straight rhythms come out as triplets |
| `--grid 8` | only use 8th notes and longer (simpler, less exact) |
| `--fingering all` | finger numbers on every note (default `smart`: where the hand moves; `none` to hide) |
| `--reach 14` | widest chord one hand plays at once, in semitones (default 12 = octave; 14 for large hands) |
| `--keep-doublings` | never drop octave-doubled notes from too-wide chords (they are rolled instead) |
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
- **Simple rhythm** (default): a note that would be written on a 16th keeps that rhythm only
  if it has a partner a 16th away in the same register (the beat a dotted rhythm leads into,
  the beat it follows, or a run). A lone note pushed off the beat by expressive timing is
  written on the beat or 8th instead. Whether a beat is in triplets is still decided on the
  timing as played.
- **Grace notes**: a short note a step away from a longer note struck right after it is an
  ornament, and is engraved as a small slashed grace note rather than as a 16th.
- **Barlines**: the downbeat is placed where accented, long bass notes fall.
- **Hands**: every onset's notes are split into a lower (LH) and upper (RH) part. A beam search
  over the whole piece picks the splits with the lowest total cost: distance from each hand's
  recent notes, a span limit that includes notes still held, crossed hands, splitting a
  close chord between the hands, the left hand taking the melody, and the right hand taking a
  soft note out of a left-hand broken chord (so a figure stays in one hand instead of the hands
  alternating). Melody is recognised by loudness relative to the surrounding two beats
  (pianists bring the melody out), so a melody note that dips below middle C stays in the
  right hand while the accompaniment around it stays in the left.
- **Playability**: a chord one hand cannot reach (`--reach`, default an octave) gives a note to the
  other hand if it fits there, else drops an octave doubling (never the melody or bass note), else
  is marked as rolled. A hand never gets more than five notes at once.
- **Fingering**: dynamic programming over each hand's notes, minimising the ergonomic cost model of
  Parncutt et al. (1997): comfortable and practical spans per finger pair, weak fingers, thumb on
  black keys, thumb passing, and three-note hand-position rules. It is solved exactly with a
  second-order Viterbi search, so a fingering is chosen with the following notes in mind. With
  `smart` fingering, numbers are printed where the hand changes position or starts after a rest,
  as in printed music.
- **Notation**: key detection (Krumhansl-Schmuckler via music21), key-aware note spelling, then
  music21 for measures, ties and beams, and LilyPond for engraving.

## Limits

Automatic transcription gives a solid first draft rather than a publication-ready score. It works
best on solo piano recordings. Expect to fix things in MuseScore using the `.musicxml`:

- Rubato-heavy or free-time passages (intros, cadenzas) get awkward rhythms.
- Each staff is one chord stream; a held melody note under moving inner voices is shortened.
- Fingering is a good starting point, not a teacher: it follows a published ergonomic model and
  gets standard scales, triads and Alberti basses right, but it does not know your hand size.
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

Fingering is checked against fingerings every pianist knows: major scales in both hands
(the right-hand thumb never lands on a black key), five-finger positions, triads, open fifths,
spread left-hand chords and the Alberti bass.

The tests also render small ground-truth pieces (an Alberti-bass minuet, a triplet nocturne, a waltz, a
fast arpeggio ballad) as humanized performances with timing jitter, early releases and tempo drift.
They check that onsets, triplets, hands and key come back correctly, and that the hand split
leaves no chord wider than an octave and no crossed hands. `tests/make_test_piece.py`
also synthesizes audio with FluidSynth for a full audio-to-PDF check.
