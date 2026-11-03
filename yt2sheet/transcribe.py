"""Piano audio -> notes, using the Transkun model (Yan & Duan), weights bundled in its wheel."""

from __future__ import annotations

from importlib import resources

import numpy as np

from .notes import Note

SAMPLE_RATE = 44100  # Transkun 2.0 operates at 44.1 kHz


def _load_model(device: str):
    import moduleconf
    import torch

    pretrained = resources.files("transkun") / "pretrained"
    conf_manager = moduleconf.parseFromFile(str(pretrained / "2.0.conf"))
    TransKun = conf_manager["Model"].module.TransKun
    conf = conf_manager["Model"].config
    # The checkpoint ships inside the installed transkun package; it stores
    # plain python objects alongside tensors, so it needs weights_only=False.
    checkpoint = torch.load(str(pretrained / "2.0.pt"), map_location=device, weights_only=False)
    model = TransKun(conf=conf).to(device)
    state = checkpoint.get("best_state_dict", checkpoint.get("state_dict"))
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def transcribe(audio: np.ndarray, device: str = "cpu") -> tuple[list[Note], list[tuple[float, float]]]:
    """Transcribe audio (samples, channels) at 44.1 kHz.

    Returns (notes, sustain_pedal_intervals).
    """
    import torch

    model = _load_model(device)
    if model.fs != SAMPLE_RATE:
        raise RuntimeError(f"Unexpected model sample rate {model.fs}")
    torch.set_grad_enabled(False)
    x = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).to(device)
    events = model.transcribe(x, discardSecondHalf=False)

    notes: list[Note] = []
    pedal: list[tuple[float, float]] = []
    for e in events:
        if e.pitch > 0:
            notes.append(Note(float(e.start), float(e.end), int(e.pitch), int(e.velocity)))
        elif -e.pitch == 64:  # sustain pedal is encoded as negative CC number
            pedal.append((float(e.start), float(e.end)))
    notes.sort(key=lambda n: (n.start, n.pitch))
    pedal.sort()
    return notes, pedal


def notes_to_midi(notes: list[Note], path: str, pedal: list[tuple[float, float]] | None = None) -> None:
    import pretty_midi

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, name="Piano")
    for n in notes:
        inst.notes.append(pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=n.start, end=n.end))
    for s, e in pedal or []:
        inst.control_changes.append(pretty_midi.ControlChange(64, 127, s))
        inst.control_changes.append(pretty_midi.ControlChange(64, 0, e))
    pm.instruments.append(inst)
    pm.write(path)


def midi_to_notes(path: str) -> list[Note]:
    """Load notes from an existing MIDI file (lets you skip transcription)."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(path)
    notes = [
        Note(n.start, n.end, n.pitch, n.velocity)
        for inst in pm.instruments
        if not inst.is_drum
        for n in inst.notes
    ]
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes


def midi_to_pedal(path: str) -> list[tuple[float, float]]:
    """Sustain-pedal (CC64) down intervals from a MIDI file."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(path)
    spans, down = [], None
    events = sorted((c.time, c.value) for inst in pm.instruments for c in inst.control_changes if c.number == 64)
    for t, v in events:
        if v >= 64 and down is None:
            down = t
        elif v < 64 and down is not None:
            spans.append((down, t))
            down = None
    if down is not None:
        spans.append((down, pm.get_end_time()))
    return spans
