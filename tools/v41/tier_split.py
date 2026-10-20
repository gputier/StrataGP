#!/usr/bin/env python3
"""tier_split.py - split the routed-expert tensors of the first shards of a GGUF into one tensor set per tier.

Input: shards 1..k of a GGUF split in N files (routed experts in MXFP4, shape [ne0, ne1, n_expert]), the tier table
written by tools/v41/tier_table.py, and an output directory.  Output: k shards with the same names.

The table (JSON) is the contract shared with tier_imatrix.py:
    {"mode": "p3" | "identity", "n_expert": 384, "n_layer": 40, "tier_types": [t0, t1, t2],
     "layers": [[[e, ...], [e, ...], [e, ...]], ...]}
`layers[L][K]` is the increasing list of the experts of tier K at layer L; the position of an expert in its tier is
its index in that list.  The three lists of a layer partition 0..n_expert-1.

What each output shard holds:
  * the GGUF header with the new tensor count, and the key-value pairs copied byte for byte, except the integer
    `split.tensors.count` (first shard) which gets the new total of the whole split file;
  * the tensor infos: every `blk.L.ffn_{gate,up,down}_exps.weight` is replaced by the three tensors
    `blk.L.tK.ffn_{gate,up,down}_exps.weight` of shape [ne0, ne1, n_k] and the same type, followed (after the last
    expert tensor of the layer) by the three I32 maps `blk.L.tK.ffn_exps_map.weight` of shape [1, n_expert]: the
    position of the expert in tier K, or -1 when it is not there.  Other tensors keep their order;
  * the data, offsets recomputed with `general.alignment` (32 by default).  Expert bytes are copied one expert at a
    time (`nb[2]` bytes, 6 266 880 for DeepSeek V4.1), never loaded whole.

The shards after k are not read: `split.count` and `split.no` are checked against the file names and kept.  In
identity mode every expert tensor must be MXFP4.  The script refuses an output directory that is an input directory
or already holds one of the output shards.

    python3 -I tools/v41/tier_split.py --table tiers.json --out-dir DIR [--dry-run] SHARD [SHARD ...]
"""
import argparse
import dataclasses
import hashlib
import json
import os
import re
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gguf_reader import GGUF_META, GGUFFile, TensorInfo  # noqa: E402

HEADER_BYTES = 24  # magic, version, tensor count (u64), key-value count (u64)
TYPE_I32 = 26
TYPE_MXFP4 = 39
N_TIERS = 3
COPY_CHUNK = 16 << 20

EXPERT_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")
SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")
INT_FMT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 10: "Q", 11: "q"}
SPLIT_KEYS = ("split.no", "split.count", "split.tensors.count")


class SplitError(Exception):
    pass


@dataclasses.dataclass
class Table:
    mode: str
    n_expert: int
    layers: dict[int, list[list[int]]]
    sha256: str


@dataclasses.dataclass
class Entry:
    """One tensor of an output shard: its info, and where its bytes come from (a copied range, a gather of
    experts, or an in-memory payload)."""
    name: str
    shape: list[int]
    type_id: int
    nbytes: int
    src_offset: int = 0
    experts: list[int] | None = None
    expert_bytes: int = 0
    payload: bytes | None = None
    offset: int = 0


@dataclasses.dataclass
class Shard:
    path: Path
    number: int
    count: int
    info: GGUFFile
    kv_end: int
    kv_values: dict[str, tuple[int, int]]  # split key -> (value offset in the file, GGUF value type)


# ---------------------------------------------------------------------------------------------------------------
# table


def load_table(path: Path) -> Table:
    raw = path.read_bytes()
    try:
        doc = json.loads(raw)
        mode, n_expert, n_layer = doc["mode"], doc["n_expert"], doc["n_layer"]
        tier_types, layers = doc["tier_types"], doc["layers"]
    except (ValueError, KeyError) as exc:
        raise SplitError(f"{path}: not a tier table ({exc!r})") from exc
    if mode not in ("p3", "identity"):
        raise SplitError(f"{path}: mode {mode!r}, expected 'p3' or 'identity'")
    if not isinstance(n_expert, int) or n_expert < N_TIERS:
        raise SplitError(f"{path}: n_expert {n_expert!r}")
    if len(tier_types) != N_TIERS:
        raise SplitError(f"{path}: {len(tier_types)} tier types, expected {N_TIERS}")
    if len(layers) != n_layer:
        raise SplitError(f"{path}: n_layer {n_layer} but {len(layers)} layers")
    for layer, tiers in enumerate(layers):
        if len(tiers) != N_TIERS:
            raise SplitError(f"{path}: layer {layer} has {len(tiers)} tiers, expected {N_TIERS}")
        for tier, experts in enumerate(tiers):
            if not experts:
                raise SplitError(f"{path}: layer {layer} tier {tier} is empty")
            if any(a >= b for a, b in zip(experts, experts[1:])):
                raise SplitError(f"{path}: layer {layer} tier {tier} is not strictly increasing")
        if sorted(e for experts in tiers for e in experts) != list(range(n_expert)):
            raise SplitError(f"{path}: layer {layer} is not a partition of 0..{n_expert - 1}")
    return Table(mode, n_expert, dict(enumerate(layers)), hashlib.sha256(raw).hexdigest())


# ---------------------------------------------------------------------------------------------------------------
# input shards


def locate_kv(fh, n_kv: int) -> tuple[int, dict[str, tuple[int, int]]]:
    """Walk the key-value block without decoding it.  Returns the offset where it ends and, for the split keys,
    the offset and type of their value."""
    found = {}
    fh.seek(HEADER_BYTES)
    for _ in range(n_kv):
        (n,) = struct.unpack("<Q", fh.read(8))
        key = fh.read(n).decode("utf-8", "replace")
        (vtype,) = struct.unpack("<I", fh.read(4))
        if key in SPLIT_KEYS:
            found[key] = (fh.tell(), vtype)
        kind, size = GGUF_META[vtype]
        if kind == "string":
            skip_strings(fh, 1)
        elif kind == "array":
            etype, count = struct.unpack("<IQ", fh.read(12))
            ekind, esize = GGUF_META[etype]
            if ekind == "string":
                skip_strings(fh, count)
            elif ekind == "array":
                raise SplitError(f"{key}: nested arrays are not supported")
            else:
                fh.seek(esize * count, os.SEEK_CUR)
        else:
            fh.seek(size, os.SEEK_CUR)
    return fh.tell(), found


def skip_strings(fh, count: int) -> None:
    for _ in range(count):
        (n,) = struct.unpack("<Q", fh.read(8))
        fh.seek(n, os.SEEK_CUR)


def info_bytes(name: str, n_dims: int) -> int:
    """Size of one tensor info: name, dimension count, dimensions, type, offset."""
    return 8 + len(name.encode()) + 4 + 8 * n_dims + 4 + 8


def read_shard(path: Path) -> Shard:
    match = SHARD_RE.search(path.name)
    if not match:
        raise SplitError(f"{path.name}: name does not end with -NNNNN-of-NNNNN.gguf")
    info = GGUFFile(path)
    with path.open("rb") as fh:
        fh.seek(16)
        (n_kv,) = struct.unpack("<Q", fh.read(8))
        kv_end, kv_values = locate_kv(fh, n_kv)
    end_of_infos = kv_end + sum(info_bytes(t.name, len(t.shape)) for t in info.tensors)
    if -(-end_of_infos // info.alignment) * info.alignment != info.data_start:
        raise SplitError(f"{path.name}: the tensor infos do not end where the reader says (non UTF-8 name?)")
    return Shard(path, int(match.group(1)), int(match.group(2)), info, kv_end, kv_values)


def check_split_keys(shards: list[Shard]) -> int:
    """split.no and split.count against the file names; returns the split.tensors.count of the first shard."""
    for expected, shard in enumerate(shards, 1):
        meta = shard.info.metadata
        if shard.number != expected:
            raise SplitError(f"{shard.path.name}: shard {shard.number} given in position {expected}, "
                             "shards 1..k are expected in order")
        for key, want in (("split.no", shard.number - 1), ("split.count", shard.count)):
            if meta.get(key) != want:
                raise SplitError(f"{shard.path.name}: {key} is {meta.get(key)!r}, the name says {want}")
        if shard.count != shards[0].count:
            raise SplitError(f"{shard.path.name}: split of {shard.count} files, the first shard says {shards[0].count}")
    if shards[-1].number > shards[0].count:
        raise SplitError("more shards than the split holds")
    total = shards[0].info.metadata.get("split.tensors.count")
    if not isinstance(total, int):
        raise SplitError(f"{shards[0].path.name}: no integer split.tensors.count")
    return total


# ---------------------------------------------------------------------------------------------------------------
# plan


def plan_shard(shard: Shard, table: Table, seen_layers: set[int]) -> list[Entry]:
    tensors = shard.info.tensors
    layer_tensors: dict[int, list[int]] = {}
    for index, t in enumerate(tensors):
        match = EXPERT_RE.match(t.name)
        if match:
            layer_tensors.setdefault(int(match.group(1)), []).append(index)
    for layer, indexes in layer_tensors.items():
        if layer not in table.layers:
            raise SplitError(f"{shard.path.name}: layer {layer} has expert tensors but no row in the table")
        if layer in seen_layers:
            raise SplitError(f"layer {layer}: expert tensors in more than one shard")
        if sorted(EXPERT_RE.match(tensors[i].name).group(2) for i in indexes) != ["down", "gate", "up"]:
            raise SplitError(f"{shard.path.name}: layer {layer} does not hold exactly gate, up and down")
        seen_layers.add(layer)
    last_of_layer = {indexes[-1]: layer for layer, indexes in layer_tensors.items()}

    entries = []
    for index, t in enumerate(tensors):
        match = EXPERT_RE.match(t.name)
        if not match:
            entries.append(copy_entry(shard, t))
            continue
        layer = int(match.group(1))
        entries.extend(tier_entries(shard, t, layer, match.group(2), table))
        if index in last_of_layer:
            entries.extend(map_entries(layer, table))
    return entries


def copy_entry(shard: Shard, t: TensorInfo) -> Entry:
    nbytes = t.expected_bytes()
    if nbytes is None:
        raise SplitError(f"{shard.path.name}: {t.name}: size of type {t.type_name} {t.shape} is unknown")
    return Entry(t.name, t.shape, t.type_id, nbytes, src_offset=t.offset)


def tier_entries(shard: Shard, t: TensorInfo, layer: int, kind: str, table: Table) -> list[Entry]:
    if len(t.shape) != 3 or t.shape[2] != table.n_expert:
        raise SplitError(f"{shard.path.name}: {t.name}: shape {t.shape}, expected [ne0, ne1, {table.n_expert}]")
    if table.mode == "identity" and t.type_id != TYPE_MXFP4:
        raise SplitError(f"{shard.path.name}: {t.name}: type {t.type_name}, identity mode needs MXFP4")
    total = t.expected_bytes()
    if total is None or total % table.n_expert:
        raise SplitError(f"{shard.path.name}: {t.name}: {t.type_name} {t.shape} does not split into experts")
    expert_bytes = total // table.n_expert
    out = []
    for tier, experts in enumerate(table.layers[layer]):
        out.append(Entry(f"blk.{layer}.t{tier}.ffn_{kind}_exps.weight", [t.shape[0], t.shape[1], len(experts)],
                         t.type_id, expert_bytes * len(experts), src_offset=t.offset, experts=experts,
                         expert_bytes=expert_bytes))
    return out


def map_entries(layer: int, table: Table) -> list[Entry]:
    out = []
    for tier, experts in enumerate(table.layers[layer]):
        positions = [-1] * table.n_expert
        for position, expert in enumerate(experts):
            positions[expert] = position
        out.append(Entry(f"blk.{layer}.t{tier}.ffn_exps_map.weight", [1, table.n_expert], TYPE_I32,
                         4 * table.n_expert, payload=struct.pack(f"<{table.n_expert}i", *positions)))
    return out


def layout(entries: list[Entry], shard: Shard) -> tuple[int, int]:
    """Assign the data offsets; returns the offset where the data starts and the size of the file."""
    align = shard.info.alignment
    infos = sum(info_bytes(e.name, len(e.shape)) for e in entries)
    data_start = -(-(shard.kv_end + infos) // align) * align
    offset = 0
    for e in entries:
        e.offset = offset
        offset += -(-e.nbytes // align) * align
    return data_start, data_start + offset


# ---------------------------------------------------------------------------------------------------------------
# write


def check_source_ranges(shard: Shard) -> None:
    """Data ranges of the input tensors (start, end), to check that no tensor overlaps another or leaves the file."""
    ranges = sorted((t.offset, t.offset + (t.expected_bytes() or 0)) for t in shard.info.tensors)
    size = shard.path.stat().st_size - shard.info.data_start
    for (_, end), (start, _) in zip(ranges, ranges[1:]):
        if end > start:
            raise SplitError(f"{shard.path.name}: overlapping tensors")
    if ranges and ranges[-1][1] > size:
        raise SplitError(f"{shard.path.name}: a tensor ends past the end of the file")


def encode_infos(entries: list[Entry]) -> bytes:
    out = bytearray()
    for e in entries:
        name = e.name.encode()
        out += struct.pack("<Q", len(name)) + name
        out += struct.pack(f"<I{len(e.shape)}QIQ", len(e.shape), *e.shape, e.type_id, e.offset)
    return bytes(out)


def header_bytes(shard: Shard, n_tensors: int, new_total: int) -> bytes:
    with shard.path.open("rb") as fh:
        head = bytearray(fh.read(shard.kv_end))
    head[8:16] = struct.pack("<Q", n_tensors)
    if "split.tensors.count" in shard.kv_values:
        offset, vtype = shard.kv_values["split.tensors.count"]
        if vtype not in INT_FMT:
            raise SplitError(f"{shard.path.name}: split.tensors.count has type {vtype}, expected an integer")
        size = struct.calcsize(INT_FMT[vtype])
        try:
            head[offset:offset + size] = struct.pack("<" + INT_FMT[vtype], new_total)
        except struct.error as exc:
            raise SplitError(f"{shard.path.name}: split.tensors.count {new_total} does not fit its type") from exc
    return bytes(head)


def write_zeros(out, count: int) -> None:
    out.write(bytes(count))


def write_shard(shard: Shard, entries: list[Entry], new_total: int, placed: tuple[int, int], target: Path) -> None:
    data_start, size = placed
    head = header_bytes(shard, len(entries), new_total)
    with shard.path.open("rb") as src, target.open("xb") as out:
        out.write(head)
        out.write(encode_infos(entries))
        write_zeros(out, data_start - out.tell())
        for e in entries:
            write_zeros(out, data_start + e.offset - out.tell())
            if e.payload is not None:
                out.write(e.payload)
            elif e.experts is not None:
                for expert in e.experts:
                    src.seek(shard.info.data_start + e.src_offset + expert * e.expert_bytes)
                    out.write(read_exact(src, e.expert_bytes))
            else:
                src.seek(shard.info.data_start + e.src_offset)
                left = e.nbytes
                while left:
                    out.write(read_exact(src, min(left, COPY_CHUNK)))
                    left -= min(left, COPY_CHUNK)
        write_zeros(out, size - out.tell())


def read_exact(fh, n: int) -> bytes:
    data = fh.read(n)
    if len(data) != n:
        raise SplitError(f"{fh.name}: short read ({len(data)} of {n} bytes)")
    return data


def verify_output(target: Path, shard: Shard, entries: list[Entry], size: int, new_total: int) -> None:
    """Reread the written shard with the reader: size, alignment, infos and key-value pairs."""
    if target.stat().st_size != size:
        raise SplitError(f"{target.name}: {target.stat().st_size} bytes written, {size} planned")
    got = GGUFFile(target)
    want = [(e.name, e.shape, e.type_id, e.offset) for e in entries]
    if [(t.name, t.shape, t.type_id, t.offset) for t in got.tensors] != want:
        raise SplitError(f"{target.name}: tensor infos differ from the plan")
    expected = dict(shard.info.metadata)
    if "split.tensors.count" in expected:
        expected["split.tensors.count"] = new_total
    if got.metadata != expected:
        raise SplitError(f"{target.name}: key-value pairs differ from the input")
    if any(e.offset % got.alignment for e in entries):
        raise SplitError(f"{target.name}: unaligned tensor offset")


# ---------------------------------------------------------------------------------------------------------------


def run(shard_paths: list[Path], table_path: Path, out_dir: Path, dry_run: bool) -> int:
    table = load_table(table_path)
    out_dir = out_dir.resolve()
    shards = [read_shard(p) for p in shard_paths]
    old_total = check_split_keys(shards)
    for shard in shards:
        target = out_dir / shard.path.name
        if shard.path.resolve().parent == out_dir:
            raise SplitError(f"{out_dir}: output directory is an input directory")
        if target.exists() or target.is_symlink():
            raise SplitError(f"{target}: already exists, nothing is overwritten")

    seen_layers: set[int] = set()
    plans = []
    for shard in shards:
        check_source_ranges(shard)
        plans.append(plan_shard(shard, table, seen_layers))
    if seen_layers != set(table.layers):
        raise SplitError(f"layers without expert tensors in the input: {sorted(set(table.layers) - seen_layers)}")
    new_total = old_total + sum(len(plan) - len(s.info.tensors) for plan, s in zip(plans, shards))

    print(f"table {table_path.name} sha256 {table.sha256} mode {table.mode}")
    sizes = []
    for shard, entries in zip(shards, plans):
        data_start, size = layout(entries, shard)
        sizes.append((data_start, size))
        print(f"{shard.path.name}: {len(shard.info.tensors)} -> {len(entries)} tensors, {size} bytes")
    print(f"split.tensors.count {old_total} -> {new_total}")
    if dry_run:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    try:
        for shard, entries, placed in zip(shards, plans, sizes):
            target = out_dir / shard.path.name
            written.append(target)
            write_shard(shard, entries, new_total, placed, target)
            verify_output(target, shard, entries, placed[1], new_total)
            print(f"wrote {target}")
    except BaseException:
        for target in written:
            target.unlink(missing_ok=True)
        raise
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("shards", nargs="+", type=Path, help="input shards 1..k, in order")
    ap.add_argument("--table", type=Path, required=True, help="tier table JSON (tier_table.py)")
    ap.add_argument("--out-dir", type=Path, required=True, help="output directory (must not hold the shards)")
    ap.add_argument("--dry-run", action="store_true", help="print the tensor counts and shard sizes, write nothing")
    args = ap.parse_args(argv)
    try:
        return run(args.shards, args.table, args.out_dir, args.dry_run)
    except (SplitError, ValueError) as exc:
        print(f"tier_split: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
