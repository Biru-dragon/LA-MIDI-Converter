#!/usr/bin/env python3
"""
midi_to_arrows.py: turns MIDI files into save strings (base64) for the arrows game.

Usage:
    python midi_to_arrows.py Rainbow_Tylenol.mid        # writes Rainbow_Tylenol.json (game save file)
    python midi_to_arrows.py song.mid -o save.txt --speed 4 --width 256
    python midi_to_arrows.py song.mid --map Lead=Piano --map ch3=SawWave
    python midi_to_arrows.py song.mid --preset chiptune   # faithful, orchestra, chiptune, piano, synth
    python midi_to_arrows.py -i song.mid        # interactive: preset / free choice per track + speed
    python midi_to_arrows.py                    # same, and also offers the MIDI files next to the script
    python midi_to_arrows.py *.mid --speed 3    # several files at once

How the generated circuit works
-------------------------------
* A single "timeline": a chain of arrows that advances 1 tick per cell (or 2 ticks per cell
  with Delay_arrow arrows (ID4) when there is nothing to play, to keep it short).
  It is folded into a serpentine (bands of `--width` cells) so that it fits in a rectangle.
* Whenever a note must be played, the timeline cell is replaced by a T_Splitter (ID8):
  it continues the timeline AND sends the signal left/right, i.e. to the Music_Arrow cells placed on
  both sides of the line.
  => at most 2 notes per tick; if more than 2 notes fall on the same instant, they are spread
  over neighbouring ticks (a shift of only 1-2 ticks).
* The path starts with a directional button (ID 24) facing the first arrow: press it ONCE
  (its position is printed at the end of the run).
* The first cell of the path also powers a command block (ID 33) just below it, whose script
  sets the game speed to the chosen level (1 = 3 ticks/s ... 5 = 1200 ticks/s) by itself.

Requirement: arrows.py (the provided library) in the same folder. Python >= 3.11.
"""
from __future__ import annotations

import argparse
import copy
import bisect
import heapq
import json
import struct
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arrows import (Arrow, ArrowType, Direction, Instrument, Map,  # noqa: E402
                    INSTRUMENT_RANGES, Pitch)

# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------
SPEED_TICKS_PER_SECOND = {1: 3, 2: 12, 3: 60, 4: 300, 5: 1200}
DEFAULT_SPEED = 4          # 300 ticks/s: ~3 ms precision, reasonably sized circuit

# Octave numbering: middle C (MIDI 60) = C4, so octave = note // 12 - 1.
# If the game's octaves are shifted, change this value (e.g. -1 or +1).
OCTAVE_OFFSET = 0

# Drums: GM percussion note -> pitch of the game's Drums instrument.
# By default the pitch class of the GM note is kept (kick 36 -> C, snare 38 -> D,
# closed hi-hat 42 -> F#, crash 49 -> C#...). Adjust to the game's actual drum sounds.
DRUM_PITCH_OVERRIDES: dict[int, Pitch] = {
    # 36: Pitch.C, 38: Pitch.D, 42: Pitch.FSharp, 49: Pitch.CSharp,
}

# General MIDI programs (0-127) -> game instrument.
_GM_RANGES: list[tuple[int, int, Instrument]] = [
    (0, 7, Instrument.Piano),
    (8, 15, Instrument.Xylophone),
    (16, 23, Instrument.Organ),
    (24, 31, Instrument.Guitar),
    (32, 39, Instrument.Bass),
    (40, 41, Instrument.Violin),
    (42, 42, Instrument.Cello),
    (43, 43, Instrument.Contrabass),
    (44, 45, Instrument.Violin),
    (46, 46, Instrument.Harp),
    (47, 47, Instrument.Timpani),
    (48, 51, Instrument.Violin),
    (52, 54, Instrument.Organ),
    (55, 56, Instrument.Trumpet),
    (57, 57, Instrument.Trombone),
    (58, 58, Instrument.Tuba),
    (59, 59, Instrument.Trumpet),
    (60, 60, Instrument.Horn),
    (61, 63, Instrument.Trumpet),
    (64, 67, Instrument.Clarinet),      # saxophones -> reed instrument
    (68, 69, Instrument.Oboe),
    (70, 70, Instrument.Bassoon),
    (71, 71, Instrument.Clarinet),
    (72, 79, Instrument.Flute),
    (80, 80, Instrument.SquareWave),
    (81, 81, Instrument.SawWave),
    (82, 83, Instrument.TriangleWave),
    (84, 84, Instrument.SawWave),
    (85, 85, Instrument.TriangleWave),
    (86, 87, Instrument.SawWave),
    (88, 95, Instrument.TriangleWave),
    (96, 103, Instrument.SawWave),
    (104, 106, Instrument.Guitar),
    (107, 107, Instrument.Harp),
    (108, 108, Instrument.Xylophone),
    (109, 109, Instrument.Oboe),
    (110, 110, Instrument.Violin),
    (111, 111, Instrument.Oboe),
    (112, 115, Instrument.Xylophone),
    (116, 118, Instrument.Timpani),
    (119, 119, Instrument.Drums),
    (120, 127, Instrument.SawWave),
]


# Presets: substitutions applied to the result of the GM mapping above
# ("faithful" game instrument -> replacement). Percussion (channel 10) always stays Drums.
I = Instrument
PRESETS: dict[str, tuple[str, dict[Instrument, Instrument]]] = {
    "faithful": ("Closest match to the GM instruments of the file", {}),
    "orchestra": ("Orchestral sound: guitars -> Harp, basses -> Contrabass, synths -> Flute/Trumpet", {
        I.Guitar: I.Harp, I.Bass: I.Contrabass, I.SquareWave: I.Flute,
        I.TriangleWave: I.Flute, I.SawWave: I.Trumpet,
    }),
    "chiptune": ("8-bit sound: everything as SquareWave / TriangleWave / SawWave", {
        I.Piano: I.SquareWave, I.Xylophone: I.TriangleWave, I.Organ: I.TriangleWave,
        I.Guitar: I.SawWave, I.Bass: I.TriangleWave, I.Violin: I.SawWave, I.Cello: I.SawWave,
        I.Contrabass: I.TriangleWave, I.Harp: I.SawWave, I.Timpani: I.TriangleWave,
        I.Trumpet: I.SquareWave, I.Trombone: I.SquareWave, I.Tuba: I.TriangleWave,
        I.Horn: I.SquareWave, I.Clarinet: I.SquareWave, I.Oboe: I.SquareWave,
        I.Bassoon: I.TriangleWave, I.Flute: I.TriangleWave,
    }),
    "piano": ("Everything on piano (basses stay Bass)", {
        i: I.Piano for i in Instrument if i not in (I.Drums, I.Bass, I.Piano)
    }),
    "synth": ("All waveforms (Square for melodies, Triangle for low parts, Saw for the rest)", {
        I.Piano: I.SquareWave, I.Xylophone: I.SquareWave, I.Organ: I.SawWave, I.Guitar: I.SawWave,
        I.Bass: I.TriangleWave, I.Violin: I.SawWave, I.Cello: I.TriangleWave,
        I.Contrabass: I.TriangleWave, I.Harp: I.SquareWave, I.Timpani: I.TriangleWave,
        I.Trumpet: I.SawWave, I.Trombone: I.SawWave, I.Tuba: I.TriangleWave, I.Horn: I.SawWave,
        I.Clarinet: I.SquareWave, I.Oboe: I.SquareWave, I.Bassoon: I.TriangleWave, I.Flute: I.SquareWave,
    }),
}
del I


# Fixed fields of the output JSON (taken from the game's own save format).
PREVIEW_DATA = ("BQABeJwVkMlOAkEURW81g6+rTRjUKIGWxLgixqA7ExZ0QzeTUcBZGfwUPsFP8hMAZ2UILlzpxpULE71ULSpVdarOuy8M1QMCtixaFaSsMrasEnKWj33LQ0sX0dYFdLSLrnaQxLYEkMEBokgr1eO0QyPJ49Oq4deqYiwOJuJiKgXCXxLGfISiRlStB9IBO1LXeTR0DU1dxZGu4FiXcaJLONU+zrSHcxovaLyk8YrGFvk2+Q75LvmEGZM+4lJXS5JVy3KDFclilWdrEkVCoJJyrVJyq5IhRwwYLGFTQtiTICzuD7lmlI8cIrBZmuoZdqQvRQzEw1B83EkJ91LGg1TwKFU8SQ3PTPjCZK9M9sZkI/Jj8hPyU/IZNcMPksYHmzPP/CcKip8b9kKd+RrMZ5gO4qaLDbOAXbMI1/TQNH02KcbqMtjhg3k3g326B3QP6X6n+9vM83qGfz4cR0E=")


SPEED_SCRIPT_TEMPLATE = """const SpeedValue = {speed};

try {{
    __nativeSetGameSpeed(SpeedValue, 0);
    log("Game Speed : " + SpeedValue);
}} catch (error) {{
    log("Error : " + error.message);
}}"""


def make_speed_script(level: int) -> str:
    """Script stored in the command block: sets the game speed to `level` (1 to 5)."""
    return SPEED_SCRIPT_TEMPLATE.format(speed=level)


def make_save_json(name: str, save: str) -> dict:
    """Wraps the base64 save string in the game's JSON save format."""
    return {
        "name": name,
        "version": 1,
        "theme": "dark",
        "background_style": "grid",
        "preview_data": PREVIEW_DATA,
        "tags": ["sound"],
        "_preview_header_end": {"version": 1, "fields": 435},
        "data": save,
        "arrows": {"30": False},
    }


def gm_to_instrument(program: int) -> Instrument:
    for lo, hi, inst in _GM_RANGES:
        if lo <= program <= hi:
            return inst
    return Instrument.Piano


# --------------------------------------------------------------------------------------
# MIDI reading (no dependency)
# --------------------------------------------------------------------------------------
class MidiNote:
    __slots__ = ("tick", "channel", "note", "track")

    def __init__(self, tick: int, channel: int, note: int, track: int):
        self.tick, self.channel, self.note, self.track = tick, channel, note, track


class MidiData:
    def __init__(self):
        self.division = 480
        self.notes: list[MidiNote] = []
        self.tempos: list[tuple[int, int]] = []             # (tick, microseconds per quarter note)
        self.programs: dict[int, list[tuple[int, int]]] = defaultdict(list)  # channel -> [(tick, program)]
        self.track_names: dict[int, str] = {}                # track index -> name
        self.channel_track: dict[int, int] = {}              # channel -> first track using it


def _vlq(data: bytes, i: int) -> tuple[int, int]:
    v = 0
    while True:
        b = data[i]
        i += 1
        v = (v << 7) | (b & 0x7F)
        if not b & 0x80:
            return v, i


def read_midi(path: str) -> MidiData:
    data = Path(path).read_bytes()
    if data[:4] != b"MThd":
        raise ValueError("This is not a MIDI file (MThd header missing)")
    _fmt, ntracks, division = struct.unpack(">HHH", data[8:14])
    if division & 0x8000:
        raise ValueError("SMPTE time division is not supported")
    md = MidiData()
    md.division = division
    i = 8 + struct.unpack(">I", data[4:8])[0]
    for tr in range(ntracks):
        if data[i:i + 4] != b"MTrk":
            raise ValueError("Invalid MIDI track")
        length = struct.unpack(">I", data[i + 4:i + 8])[0]
        i += 8
        end = i + length
        t = 0
        running = None
        while i < end:
            dt, i = _vlq(data, i)
            t += dt
            b = data[i]
            if b == 0xFF:                                   # meta event
                typ = data[i + 1]
                ln, j = _vlq(data, i + 2)
                payload = data[j:j + ln]
                i = j + ln
                if typ == 0x51 and ln == 3:
                    md.tempos.append((t, int.from_bytes(payload, "big")))
                elif typ == 0x03 and tr not in md.track_names:
                    md.track_names[tr] = payload.decode("latin-1").strip("\x00 ")
                continue
            if b in (0xF0, 0xF7):                           # sysex
                ln, j = _vlq(data, i + 1)
                i = j + ln
                continue
            if b & 0x80:
                running = b
                i += 1
            kind, ch = running >> 4, running & 15
            nbytes = 1 if kind in (0xC, 0xD) else 2
            args = data[i:i + nbytes]
            i += nbytes
            if kind == 0x9 and args[1] > 0:
                md.notes.append(MidiNote(t, ch, args[0], tr))
                md.channel_track.setdefault(ch, tr)
            elif kind == 0xC:
                md.programs[ch].append((t, args[0]))
        i = end
    md.tempos.sort(key=lambda x: x[0])
    for ch in md.programs:
        md.programs[ch].sort(key=lambda x: x[0])
    return md


class TempoMap:
    def __init__(self, md: MidiData):
        segs = [(0, 500000)]
        for tick, us in md.tempos:
            if tick == 0:
                segs[0] = (0, us)
            else:
                segs.append((tick, us))
        self.starts, self.secs, self.us = [], [], []
        sec = 0.0
        for k, (tick, us) in enumerate(segs):
            if k > 0:
                sec = self.secs[-1] + (tick - self.starts[-1]) * self.us[-1] / 1e6 / md.division
            self.starts.append(tick)
            self.secs.append(sec)
            self.us.append(us)
        self.division = md.division

    def seconds(self, tick: int) -> float:
        k = bisect.bisect_right(self.starts, tick) - 1
        return self.secs[k] + (tick - self.starts[k]) * self.us[k] / 1e6 / self.division


# --------------------------------------------------------------------------------------
# MIDI notes -> game notes
# --------------------------------------------------------------------------------------
NoteKey = tuple  # (instrument:int, pitch:int, octave:int)


def parse_overrides(items: list[str]) -> dict[str, Instrument]:
    out = {}
    for it in items or []:
        key, _, val = it.partition("=")
        try:
            out[key.strip().lower()] = Instrument[val.strip()]
        except KeyError:
            names = ", ".join(i.name for i in Instrument)
            raise SystemExit(f"Unknown instrument: {val!r}. Possible values: {names}")
    return out


def fold_octave(inst: Instrument, octave: int) -> tuple[int, bool]:
    lo, hi = INSTRUMENT_RANGES[inst]
    folded = False
    while octave < lo:
        octave += 1
        folded = True
    while octave > hi:
        octave -= 1
        folded = True
    return octave, folded


def convert_notes(md: MidiData, overrides: dict[str, Instrument], preset: str = "faithful"):
    """Returns (list of (second, NoteKey), per-track summary)."""
    tempo = TempoMap(md)
    out = []
    summary: dict[str, dict] = {}
    for n in md.notes:
        track_name = md.track_names.get(n.track, f"track{n.track}")
        if n.channel == 9:
            inst = Instrument.Drums
            pitch = DRUM_PITCH_OVERRIDES.get(n.note, Pitch(n.note % 12))
            octave = 1
            folded = False
        else:
            progs = md.programs.get(n.channel, [])
            k = bisect.bisect_right([p[0] for p in progs], n.tick) - 1
            program = progs[k][1] if k >= 0 else 0
            inst = overrides.get(track_name.lower())
            if inst is None:
                inst = overrides.get(f"ch{n.channel}")
            if inst is None:
                inst = PRESETS[preset][1].get(gm_to_instrument(program), gm_to_instrument(program))
            if inst == Instrument.Drums:
                pitch, octave, folded = Pitch(n.note % 12), 1, False
            else:
                pitch = Pitch(n.note % 12)
                octave, folded = fold_octave(inst, n.note // 12 - 1 + OCTAVE_OFFSET)
        s = summary.setdefault(track_name, {"inst": inst, "count": 0, "folded": 0})
        s["count"] += 1
        s["folded"] += int(folded)
        out.append((tempo.seconds(n.tick), (int(inst), int(pitch), octave)))
    return out, summary


# --------------------------------------------------------------------------------------
# Placing the notes on the timeline
# --------------------------------------------------------------------------------------
MAX_SPREAD = 24
INTRO_TICKS = 6      # margin ticks at the start (the first line cell never carries a tap)


def _spread_offsets():
    yield 0
    for d in range(1, MAX_SPREAD + 1):
        yield -d
        yield d


def assign_slots(groups: dict[int, list[NoteKey]], blocked: set[int]) -> dict[int, list[tuple[NoteKey, int]]]:
    """Assigns each note to a tap tick (2 notes max per tick), as close as possible to its target."""
    slots: dict[int, list] = {}
    # Biggest chords first: they get ticks well centred around their target,
    # the smaller groups make do with what is left.
    for t in sorted(groups, key=lambda t: (-len(groups[t]), t)):
        notes = sorted(groups[t], key=lambda k: (k[0] != int(Instrument.Drums), k))
        for note in notes:
            for off in _spread_offsets():
                s = t + off
                if s < 0 or s in blocked:
                    continue
                lst = slots.get(s)
                if lst is None:
                    slots[s] = [(note, t)]
                    break
                if len(lst) < 2:
                    lst.append((note, t))
                    break
            else:
                raise RuntimeError("Cannot place a note (density too high)")
    return slots


# --------------------------------------------------------------------------------------
# Geometry: serpentine
# --------------------------------------------------------------------------------------
BAND_PITCH = 4      # rows per band: N notes / line / S notes / empty safety row


def side_cells(x: int, y: int, direction: Direction):
    """The two neighbouring cells (left, right) of a tap facing `direction`,
    i.e. the ones the T_Splitter feeds in addition to the next cell."""
    out = []
    for lx in (-1, 1):
        dx, dy = _rotate(lx, 0, direction, False)
        out.append((x + dx, y + dy))
    return out


ELASTIC = 10   # bands may be longer / shorter by +-ELASTIC cells


def build_cells(slots: dict[int, list], width: int):
    """Builds the path cell by cell. Returns (cells, conflicting_ticks).

    A cell = (x, y, direction, type, activation tick).
    Geometry: horizontal bands (E, then W, then E...) every BAND_PITCH rows; between two bands,
    a connector column of 5 cells (A0 A1 A2 A3 A4). The first and last cell of a band, as well as
    A0 and A4, never carry a tap (their neighbours must stay empty because A1..A3 are vertical
    taps). A band only turns when these 4 cells fall on ticks without a note, hence the
    elastic width."""
    tap_set = set(slots)
    taps_sorted = sorted(tap_set)
    last = taps_sorted[-1]
    cells: list = []
    conflicts: set[int] = set()

    def free(c0: int) -> bool:
        return not any(t in tap_set for t in (c0, c0 + 1, c0 + 5, c0 + 6))

    def plain(x, y, d, c):
        if c in tap_set:
            conflicts.add(c)
        cells.append((x, y, d, ArrowType.Arrow, c))

    c, r, x, j, east = 0, 1, 2, 0, True
    xe_min, xe_max = width + 1, width + 1 + ELASTIC
    xw_max, xw_min = 2 + ELASTIC, 2
    while c <= last:
        d = Direction.East if east else Direction.West
        if j == 0:
            plain(x, r, d, c)
            c += 1
        else:
            if east:
                eligible, forced = x >= xe_min, x >= xe_max
            else:
                eligible, forced = x <= xw_max, x <= xw_min
            if forced or (eligible and free(c)):
                # last cell of the band, then connector column
                plain(x, r, d, c)
                cx = x + 1 if east else x - 1
                plain(cx, r, Direction.South, c + 1)                       # A0
                for m in (1, 2, 3):                                        # A1..A3: vertical taps
                    t = c + 1 + m
                    typ = ArrowType.T_Splitter if t in tap_set else ArrowType.Arrow
                    cells.append((cx, r + m, Direction.South, typ, t))
                plain(cx, r + 4, Direction.West if east else Direction.East, c + 5)   # A4
                c += 6
                r += BAND_PITCH
                east = not east
                j = 0
                continue
            if c in tap_set:
                cells.append((x, r, d, ArrowType.T_Splitter, c))
                c += 1
            else:
                nt = taps_sorted[bisect.bisect_left(taps_sorted, c)]
                if nt - c >= 2:
                    cells.append((x, r, d, ArrowType.Delay_arrow, c))
                    c += 2
                else:
                    cells.append((x, r, d, ArrowType.Arrow, c))
                    c += 1
        j += 1
        x += 1 if east else -1
    return cells, conflicts


def plan_circuit(groups, width):
    blocked: set[int] = set()
    for _ in range(10000):
        slots = assign_slots(groups, blocked)
        cells, conflicts = build_cells(slots, width)
        if not conflicts:
            return slots, cells
        blocked |= conflicts
    raise RuntimeError("Placement does not converge")


# --------------------------------------------------------------------------------------
# Simulation (verification) following the given propagation rules
# --------------------------------------------------------------------------------------
# type -> (list of outputs in the arrow's local frame [up = (0,-1), right = (1,0)], delay)
_RULES = {
    ArrowType.Arrow: ([(0, -1)], 1),
    ArrowType.Directional_button: ([(0, -1)], 1),
    ArrowType.Delay_arrow: ([(0, -1)], 2),
    ArrowType.Detector: ([(0, -1)], 1),
    ArrowType.I_Splitter: ([(0, -1), (0, 1)], 1),
    ArrowType.L_Splitter: ([(0, -1), (1, 0)], 1),
    ArrowType.T_Splitter: ([(0, -1), (-1, 0), (1, 0)], 1),
    ArrowType.Pulse_generator: ([(0, -2)], 1),
    ArrowType.Blue_arrow: ([(1, -1)], 1),
    ArrowType.Diagonal_arrow: ([(0, -1), (0, -2)], 1),
    ArrowType.Blue_I_Splitter: ([(0, -2), (1, 0)], 1),
    ArrowType.Blue_L_Splitter: ([(0, -1), (1, -1)], 1),
}


def _rotate(dx: int, dy: int, direction: int, flipped: bool):
    if flipped:
        dx = -dx
    for _ in range(int(direction)):      # clockwise rotation (screen, y pointing down)
        dx, dy = -dy, dx
    return dx, dy


def simulate(m: Map, start: tuple[int, int]):
    cells = {}
    for (cx, cy), chunk in m.chunks.items():
        for i, a in enumerate(chunk.arrows):
            if a.type != ArrowType.Empty_cell:
                cells[(cx * 16 + i % 16, cy * 16 + i // 16)] = a
    times: dict[tuple[int, int], list[int]] = defaultdict(list)
    heap = [(-1, start)]      # the button fires one tick before the first arrow (activated at tick 0)
    while heap:
        t, pos = heapq.heappop(heap)
        times[pos].append(t)
        a = cells.get(pos)
        rule = _RULES.get(a.type) if a else None
        if not rule:
            continue
        outs, delay = rule
        for ox, oy in outs:
            dx, dy = _rotate(ox, oy, a.direction, a.flipped)
            heapq.heappush(heap, (t + delay, (pos[0] + dx, pos[1] + dy)))
    return cells, times


# --------------------------------------------------------------------------------------
# Main program
# --------------------------------------------------------------------------------------
def choose_midi_files() -> list[str]:
    """Offers the .mid / .midi files located next to the script (and in the current folder)."""
    found: list[Path] = []
    for folder in (Path(__file__).resolve().parent, Path.cwd()):
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in (".mid", ".midi") and f.is_file() and f not in found:
                found.append(f)
    if found:
        print("MIDI files found next to the script:")
        for n, f in enumerate(found, 1):
            print(f"  {n}. {f.name}")
        prompt = "Number(s) separated by spaces, 'all', or another path: "
    else:
        print("No MIDI file next to the script.")
        prompt = "Path of the MIDI file: "
    while True:
        ans = input(prompt).strip()
        if not ans:
            continue
        if ans.lower() in ("all", "*") and found:
            return [str(f) for f in found]
        picked = []
        for tok in ans.replace(",", " ").split():
            if tok.isdigit() and 1 <= int(tok) <= len(found):
                picked.append(str(found[int(tok) - 1]))
            elif Path(tok.strip('"')).is_file():
                picked.append(tok.strip('"'))
            else:
                print(f"  {tok!r}: not found.")
                picked = []
                break
        if picked:
            return picked


def choose_preset(default: str = "faithful") -> str | None:
    """Returns the chosen preset name, or None for free mode."""
    names = list(PRESETS)
    print("\nInstrument presets:")
    for n, name in enumerate(names, 1):
        print(f"  {n}. {name:<10} {PRESETS[name][0]}")
    print(f"  {len(names) + 1}. free       You pick the instrument of every track yourself")
    while True:
        ans = input(f"Preset [{names.index(default) + 1}]: ").strip().lower()
        if not ans:
            return default
        if ans in ("free", str(len(names) + 1)):
            return None
        if ans.isdigit() and 1 <= int(ans) <= len(names):
            return names[int(ans) - 1]
        if ans in PRESETS:
            return ans
        print("  Invalid choice.")


def ask_instruments(midi_path: str, preset: str | None) -> list[str]:
    """Interactive mode: applies the preset (or free mode if None) then lets you adjust track by track.
    Returns the matching list of --map entries."""
    md = read_midi(midi_path)
    _notes, summary = convert_notes(md, {}, preset or "faithful")
    names = [i.name for i in Instrument]
    print("\nTracks and instruments:")
    for track, info in summary.items():
        print(f"  {track!r:16} {info['count']:5d} notes -> {info['inst'].name}")
    if preset is not None:
        if input("Adjust some tracks? [y/N]: ").strip().lower() not in ("y", "yes"):
            return []
    print("Game instruments: " + ", ".join(f"{i.value}={i.name}" for i in Instrument))
    result = []
    for track, info in summary.items():
        if info["inst"] == Instrument.Drums:
            print(f"Track {track!r}: percussion -> Drums (cannot be changed)")
            continue
        while True:
            ans = input(f"Track {track!r}, instrument [{info['inst'].name}] "
                        f"(name or number, Enter = keep): ").strip()
            if not ans:
                break
            if ans.isdigit() and int(ans) < len(names):
                ans = names[int(ans)]
            match = [n for n in names if n.lower() == ans.lower()]
            if match:
                result.append(f"{track}={match[0]}")
                break
            print("  Unknown instrument, try again.")
    return result


def ask_speed(default: int) -> int:
    print("\nSpeed levels: " + ", ".join(f"{k}={v} ticks/s" for k, v in SPEED_TICKS_PER_SECOND.items()))
    print("(the faster, the more accurate the music but the bigger the circuit)")
    while True:
        ans = input(f"Game speed [{default}]: ").strip()
        if not ans:
            return default
        if ans.isdigit() and int(ans) in SPEED_TICKS_PER_SECOND:
            return int(ans)
        print("  Enter a number from 1 to 5.")


def main():
    ap = argparse.ArgumentParser(description="Converts MIDI files into save strings for the arrows game")
    ap.add_argument("midi", nargs="*", help="one or more .mid files")
    ap.add_argument("-i", "--interactive", action="store_true",
                    help="asks for the instruments and the speed (enabled when no file is given)")
    ap.add_argument("-o", "--output", help="output JSON file (default: <MIDI name>.json next to the MIDI; single MIDI only)")
    ap.add_argument("--speed", type=int, choices=SPEED_TICKS_PER_SECOND, default=DEFAULT_SPEED,
                    help="target game speed level (1 to 5, default 4 = 300 ticks/s)")
    ap.add_argument("--width", type=int, default=256, help="width of a serpentine band, in cells")
    ap.add_argument("--preset", choices=list(PRESETS), default="faithful",
                    help="instrument mapping: " + " ; ".join(f"{k} = {v[0]}" for k, v in PRESETS.items()))
    ap.add_argument("--map", action="append", metavar="TRACK=Instrument",
                    help="forces the instrument of a track (track name) or of a channel (ch0..ch15)")
    ap.add_argument("--keep-intro", action="store_true", help="keeps the silence before the first note")
    ap.add_argument("--no-verify", action="store_true", help="skips the verification simulation")
    args = ap.parse_args()

    files = args.midi
    interactive = args.interactive or not files
    if not files:
        files = choose_midi_files()
    for f in files:
        a = copy.copy(args)
        a.map = list(args.map or [])
        if interactive:
            print(f"\n=== {f} ===")
            choice = choose_preset(args.preset)
            a.preset = choice or "faithful"
            a.map += ask_instruments(f, choice)
            a.speed = ask_speed(args.speed)
        if len(files) > 1:
            a.output = None
        run(a, f)


def run(args, midi_path):
    rate = SPEED_TICKS_PER_SECOND[args.speed]
    md = read_midi(midi_path)
    notes, summary = convert_notes(md, parse_overrides(args.map), args.preset)
    if not notes:
        raise SystemExit("No notes in this MIDI file")

    t0 = 0.0 if args.keep_intro else min(s for s, _ in notes)
    groups: dict[int, set] = defaultdict(set)
    for sec, key in notes:
        groups[INTRO_TICKS + round((sec - t0) * rate)].add(key)
    groups = {t: sorted(v) for t, v in groups.items()}
    n_notes = sum(len(v) for v in groups.values())

    try:
        slots, cells = plan_circuit(groups, args.width)
    except RuntimeError as e:
        raise SystemExit(f"{e}\nSpeed {args.speed} ({rate} ticks/s) is too slow for this piece: "
                         f"try a higher level with --speed.")

    m = Map()
    m.version = 5
    for x, y, d, typ, _t in cells:
        m.set(x, y, Arrow(typ, d))
    # Command block that sets the game speed. The first line cell becomes a splitter whose south
    # output powers the block (the block is a dead end, facing away from the line).
    x0, y0, d0, _typ0, c0 = cells[0]
    cells[0] = (x0, y0, d0, ArrowType.T_Splitter, c0)
    m.set(x0, y0, Arrow(ArrowType.T_Splitter, d0))
    cmd_pos = side_cells(x0, y0, d0)[1]          # south of the first cell (east-facing path)
    assert cmd_pos == (x0, y0 + 1), cmd_pos
    m.set(cmd_pos[0], cmd_pos[1],
          Arrow.text(ArrowType.Script_block, make_speed_script(args.speed), Direction.South))
    # Start button: a directional button (ID 24) right before the first arrow, facing it.
    button = (cells[0][0] - 1, cells[0][1])
    m.set(button[0], button[1], Arrow(ArrowType.Directional_button, cells[0][2]))
    placed = {}                                  # (x, y) -> (note key, target tick, tap tick)
    by_tick = {c[4]: c for c in cells}
    for s, lst in slots.items():
        x, y, d, _typ, _t = by_tick[s]
        for (note, target), (nx, ny) in zip(lst, side_cells(x, y, d)):
            inst, pitch, octave = note
            side = {(0, -1): Direction.North, (1, 0): Direction.East,
                    (0, 1): Direction.South, (-1, 0): Direction.West}[(nx - x, ny - y)]
            m.set(nx, ny, Arrow.note(Instrument(inst), Pitch(pitch), octave, side))
            placed[(nx, ny)] = (note, target, s)

    save = m.export()
    out_path = Path(args.output) if args.output else Path(midi_path).with_suffix(".json")
    out_path.write_text(json.dumps(make_save_json(Path(midi_path).stem, save), indent=2, ensure_ascii=False),
                        encoding="utf-8")

    # ---------------- report ----------------
    start = button
    last_tick = max(slots) + 1
    shifts = [abs(s - target) for s, lst in slots.items() for _n, target in lst]
    print(f"File           : {midi_path}  ({md.division} ticks/quarter note)")
    for name, s in summary.items():
        extra = f", {s['folded']} notes moved into the octave range" if s["folded"] else ""
        print(f"  track {name!r:14} -> {s['inst'].name:<12} {s['count']:5d} notes{extra}")
    print(f"Notes played   : {n_notes} (identical simultaneous notes merged)")
    print(f"Game speed     : level {args.speed} ({rate} ticks/s), set automatically by the command block")
    print(f"Duration       : {last_tick / rate:.1f} s ({last_tick} ticks)")
    print(f"Circuit        : {len(cells)} line cells, {len(placed)} notes, "
          f"{max(c[1] for c in cells) + 2} rows high, {max(c[0] for c in cells) + 2} columns wide")
    print(f"Max shift      : {max(shifts)} tick(s) ({max(shifts) / rate * 1000:.1f} ms) from the exact instant")
    print(f"INPUT          : press the directional button at x={start[0]}, y={start[1]} "
          f"(start of the path, top-left of the save)")
    print(f"Save file      : {out_path}  (JSON, {len(save)} characters of save data)")

    # ---------------- verification ----------------
    if not args.no_verify:
        m2 = Map(save)                            # re-read the exported string
        cells2, times = simulate(m2, start)
        errors = 0
        for pos, a in cells2.items():
            ts = times.get(pos, [])
            if len(ts) != 1:
                errors += 1
                if errors <= 5:
                    print(f"  ! cell {pos} ({a.type.name}) activated {len(ts)} times")
                continue
            if a.type == ArrowType.Music_Arrow:
                note, target, s = placed[pos]
                got = (int(a.instrument), int(a.pitch), a.octave)
                if got != note or ts[0] != s + 1:
                    errors += 1
                    if errors <= 5:
                        print(f"  ! note at {pos}: expected {note} at {s + 1}, got {got} at {ts[0]}")
        cb = cells2.get(cmd_pos)
        if (cb is None or cb.type != ArrowType.Script_block or cb.string != make_speed_script(args.speed)
                or times.get(cmd_pos) != [1]):
            errors += 1
            print(f"  ! command block at {cmd_pos} missing, wrong script, or not powered once at tick 1")
        n_note_cells = sum(1 for a in cells2.values() if a.type == ArrowType.Music_Arrow)
        if n_note_cells != len(placed):
            errors += 1
            print(f"  ! {n_note_cells} notes read back, {len(placed)} expected")
        if errors:
            raise SystemExit(f"VERIFICATION FAILED ({errors} errors)")
        print("Verification   : OK (re-read + simulation: every note played exactly once, at the right tick)")


if __name__ == "__main__":
    main()
