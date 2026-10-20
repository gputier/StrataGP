"""Tests of tier_split.py on a synthetic MXFP4 GGUF, expected values worked out by hand.

The model: 2 layers of 6 experts, expert tensors [64, 8, 6] in MXFP4 (272 bytes per expert: 64 x 8 / 32 blocks of
17 bytes / 6 experts), split in 2 shards of a 3-file split (the third file holds 2 tensors and is never read).

    python -m pytest tools/v41/test_tier_split.py
"""
import hashlib
import json
import random
import struct
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tier_split as ts  # noqa: E402
from gguf_reader import GGUFFile  # noqa: E402

MXFP4 = 39
F32 = 0
F16 = 1
I32 = 26
EXPERT_BYTES = 272  # 64 x 8 weights = 16 blocks of 17 bytes
N_EXPERT = 6
THIRD_SHARD_TENSORS = 2
NAMES = ("m-00001-of-00003.gguf", "m-00002-of-00003.gguf")

# layer -> three increasing expert lists that partition 0..5
LAYERS = [[[0, 3], [1, 4, 5], [2]], [[5], [0, 1, 2], [3, 4]]]


def s(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def kv_scalar(key, type_id, fmt, value):
    return s(key) + struct.pack("<I" + fmt, type_id, value)


def kv_string(key, value):
    return s(key) + struct.pack("<I", 8) + s(value)


def kv_array(key, etype, items):
    body = b"".join(s(i) for i in items) if etype == 8 else b"".join(items)
    return s(key) + struct.pack("<IIQ", 9, etype, len(items)) + body


def align_up(n, a):
    return -(-n // a) * a


def write_gguf(path, kvs, tensors, alignment=32, reverse=False):
    """tensors: list of (name, shape, type_id, data).  `reverse` stores the data in the reverse of the table order."""
    offsets = {}
    order = list(reversed(tensors)) if reverse else tensors
    cursor = 0
    for name, _, _, data in order:
        offsets[name] = cursor
        cursor = align_up(cursor + len(data), alignment)
    head = struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(kvs)) + b"".join(kvs)
    for name, shape, type_id, _ in tensors:
        head += s(name) + struct.pack(f"<I{len(shape)}QIQ", len(shape), *shape, type_id, offsets[name])
    head += bytes(align_up(len(head), alignment) - len(head))
    body = bytearray(cursor)
    for name, _, _, data in tensors:
        body[offsets[name]:offsets[name] + len(data)] = data
    path.write_bytes(head + bytes(body))


def blob(rng, n):
    return bytes(rng.randrange(256) for _ in range(n))


def build(directory: Path, alignment=32):
    """Write the two input shards; returns {tensor name: bytes} of everything they hold."""
    rng = random.Random(41)
    data = {}

    def tensor(name, shape, type_id, nbytes):
        data[name] = blob(rng, nbytes)
        return (name, shape, type_id, data[name])

    def experts(layer, kind):
        return tensor(f"blk.{layer}.ffn_{kind}_exps.weight", [64, 8, N_EXPERT], MXFP4, N_EXPERT * EXPERT_BYTES)

    def kvs(no, extra):
        out = [kv_string("general.architecture", "toy"), kv_array("tokenizer.ggml.tokens", 8, ["a", "bc", "", "dé"]),
               kv_array("toy.ids", 5, [struct.pack("<i", v) for v in (-1, 7)]),
               kv_scalar("toy.eps", 6, "f", 0.5),
               kv_scalar("split.no", 2, "H", no), kv_scalar("split.count", 2, "H", 3)]
        out += extra
        if alignment != 32:
            out.append(kv_scalar("general.alignment", 4, "I", alignment))
        return out

    first = [tensor("token_embd.weight", [4, 3], F32, 48), experts(0, "gate"), experts(0, "up"), experts(0, "down"),
             tensor("blk.0.attn_norm.weight", [8], F32, 32)]
    # 5 + 4 + 2 tensors in the whole split
    write_gguf(directory / NAMES[0], kvs(0, [kv_scalar("split.tensors.count", 5, "i", 11)]), first, alignment)
    second = [experts(1, "up"), experts(1, "gate"), tensor("blk.1.attn_q.weight", [4, 4], F16, 32), experts(1, "down")]
    write_gguf(directory / NAMES[1], kvs(1, []), second, alignment, reverse=True)
    return data


def table(mode="p3", layers=LAYERS):
    return {"mode": mode, "n_expert": N_EXPERT, "n_layer": len(layers), "tier_types": ["mxfp4", "iq2_xs", "iq1_m"],
            "layers": layers}


@pytest.fixture
def env(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    data = build(src)
    path = tmp_path / "tiers.json"
    path.write_text(json.dumps(table()))
    return tmp_path, src, data, path


def run(src, tpath, out, *extra):
    return ts.main([str(src / n) for n in NAMES] + ["--table", str(tpath), "--out-dir", str(out), *extra])


def tensor_bytes(gf: GGUFFile, name: str, n: int | None = None) -> bytes:
    t = next(t for t in gf.tensors if t.name == name)
    n = n or t.expected_bytes()
    with gf.path.open("rb") as fh:
        fh.seek(gf.data_start + t.offset)
        return fh.read(n)


def test_layout_and_bytes(env):
    tmp, src, data, tpath = env
    out = tmp / "out"
    assert run(src, tpath, out) == 0

    first, second = (GGUFFile(out / n) for n in NAMES)
    names = [[t.name for t in f.tensors] for f in (first, second)]
    assert names[0] == (["token_embd.weight"]
                        + [f"blk.0.t{k}.ffn_{kind}_exps.weight" for kind in ("gate", "up", "down") for k in range(3)]
                        + [f"blk.0.t{k}.ffn_exps_map.weight" for k in range(3)]
                        + ["blk.0.attn_norm.weight"])
    # 14 tensors in the first shard, 13 in the second, 11 + 9 + 9 = 29 in the whole split
    assert (len(names[0]), len(names[1])) == (14, 13)
    assert names[1] == ([f"blk.1.t{k}.ffn_up_exps.weight" for k in range(3)]
                        + [f"blk.1.t{k}.ffn_gate_exps.weight" for k in range(3)]
                        + ["blk.1.attn_q.weight"]
                        + [f"blk.1.t{k}.ffn_down_exps.weight" for k in range(3)]
                        + [f"blk.1.t{k}.ffn_exps_map.weight" for k in range(3)])

    for gf, layers in ((first, [0]), (second, [1])):
        for layer in layers:
            for kind in ("gate", "up", "down"):
                original = data[f"blk.{layer}.ffn_{kind}_exps.weight"]
                for k, experts in enumerate(LAYERS[layer]):
                    t = next(t for t in gf.tensors if t.name == f"blk.{layer}.t{k}.ffn_{kind}_exps.weight")
                    assert (t.shape, t.type_id) == ([64, 8, len(experts)], MXFP4)
                    got = tensor_bytes(gf, t.name)
                    assert len(got) == len(experts) * EXPERT_BYTES
                    for position, e in enumerate(experts):
                        assert got[position * EXPERT_BYTES:(position + 1) * EXPERT_BYTES] == original[e * EXPERT_BYTES:(e + 1) * EXPERT_BYTES]
            for k, experts in enumerate(LAYERS[layer]):
                name = f"blk.{layer}.t{k}.ffn_exps_map.weight"
                t = next(t for t in gf.tensors if t.name == name)
                assert (t.shape, t.type_id) == ([1, N_EXPERT], I32)
                want = [experts.index(e) if e in experts else -1 for e in range(N_EXPERT)]
                assert list(struct.unpack("<6i", tensor_bytes(gf, name, 4 * N_EXPERT))) == want
    for gf, name, type_id in ((first, "token_embd.weight", F32), (first, "blk.0.attn_norm.weight", F32),
                              (second, "blk.1.attn_q.weight", F16)):
        t = next(t for t in gf.tensors if t.name == name)
        assert (t.type_id, tensor_bytes(gf, name)) == (type_id, data[name])
    # 3 x 272 bytes is not a multiple of 32: the offsets must be padded
    for gf in (first, second):
        assert all(t.offset % gf.alignment == 0 for t in gf.tensors)
        assert gf.data_start % gf.alignment == 0


def test_split_keys_and_verbatim_kv(env):
    tmp, src, _, tpath = env
    out = tmp / "out"
    assert run(src, tpath, out) == 0
    for n, no in zip(NAMES, (0, 1)):
        before, after = GGUFFile(src / n), GGUFFile(out / n)
        assert (after.metadata["split.no"], after.metadata["split.count"]) == (no, 3)
        want = dict(before.metadata)
        if no == 0:
            want["split.tensors.count"] = 29
        assert after.metadata == want
        # key-value block byte for byte: only the 4 bytes of split.tensors.count may differ
        with (src / n).open("rb") as a, (out / n).open("rb") as b:
            kv_before, kv_after = a.read(before.data_start), b.read(after.data_start)
        end = ts.read_shard(src / n).kv_end
        patched = bytearray(kv_before[24:end])
        if no == 0:
            at = patched.index(b"split.tensors.count") + len("split.tensors.count") + 4
            patched[at:at + 4] = struct.pack("<i", 29)
        assert kv_after[24:end] == bytes(patched)
        assert kv_after[:8] == kv_before[:8] and kv_after[16:24] == kv_before[16:24]
        assert struct.unpack("<Q", kv_after[8:16])[0] == len(after.tensors)


@pytest.mark.parametrize("alignment", [32, 64])
def test_alignment(tmp_path, alignment):
    src = tmp_path / "src"
    src.mkdir()
    data = build(src, alignment)
    tpath = tmp_path / "t.json"
    tpath.write_text(json.dumps(table()))
    assert run(src, tpath, tmp_path / "out") == 0
    for n in NAMES:
        gf = GGUFFile(tmp_path / "out" / n)
        assert gf.alignment == alignment
        assert gf.data_start % alignment == 0
        assert all(t.offset % alignment == 0 for t in gf.tensors)
        assert (tmp_path / "out" / n).stat().st_size % alignment == 0
    gf = GGUFFile(tmp_path / "out" / NAMES[1])
    assert tensor_bytes(gf, "blk.1.attn_q.weight") == data["blk.1.attn_q.weight"]


def test_dry_run_writes_nothing_and_reports_sizes(env, capsys):
    tmp, src, _, tpath = env
    out = tmp / "out"
    assert run(src, tpath, out, "--dry-run") == 0
    report = capsys.readouterr().out
    assert not out.exists()
    assert "5 -> 14 tensors" in report and "4 -> 13 tensors" in report
    assert "split.tensors.count 11 -> 29" in report
    assert f"sha256 {hashlib.sha256(tpath.read_bytes()).hexdigest()}" in report
    assert run(src, tpath, out) == 0
    for n in NAMES:
        assert f"{n}: " in report and f"{(out / n).stat().st_size} bytes" in report


def test_refuses_input_directory_and_existing_output(env):
    tmp, src, _, tpath = env
    assert run(src, tpath, src) == 1
    out = tmp / "out"
    out.mkdir()
    (out / NAMES[1]).write_bytes(b"x")
    assert run(src, tpath, out) == 1
    assert [p.name for p in out.iterdir()] == [NAMES[1]]
    assert (out / NAMES[1]).read_bytes() == b"x"
    link = tmp / "link"
    link.symlink_to(src)
    assert run(src, tpath, link) == 1


def test_identity_mode_needs_mxfp4(env):
    tmp, src, data, tpath = env
    tpath.write_text(json.dumps(table("identity", [[[0, 3], [1, 4], [2, 5]], [[0, 3], [1, 4], [2, 5]]])))
    assert run(src, tpath, tmp / "ok") == 0
    other = tmp / "other"
    other.mkdir()
    rng = random.Random(1)
    rows = [("blk.0.ffn_gate_exps.weight", [64, 8, N_EXPERT], F16, blob(rng, 64 * 8 * 2 * N_EXPERT)),
            ("blk.0.ffn_up_exps.weight", [64, 8, N_EXPERT], F16, blob(rng, 64 * 8 * 2 * N_EXPERT)),
            ("blk.0.ffn_down_exps.weight", [64, 8, N_EXPERT], F16, blob(rng, 64 * 8 * 2 * N_EXPERT))]
    kvs = [kv_scalar("split.no", 2, "H", 0), kv_scalar("split.count", 2, "H", 3),
           kv_scalar("split.tensors.count", 5, "i", 3)]
    write_gguf(other / NAMES[0], kvs, rows)
    for mode in ("identity", "p3"):
        tpath.write_text(json.dumps(table(mode, [LAYERS[0]])))
        code = ts.main([str(other / NAMES[0]), "--table", str(tpath), "--out-dir", str(tmp / f"out-{mode}")])
        assert code == (1 if mode == "identity" else 0)
    assert not (tmp / "out-identity").exists()


def test_planning_error_leaves_no_output(env):
    tmp, src, _, tpath = env
    tpath.write_text(json.dumps(table(layers=[LAYERS[0]])))
    assert run(src, tpath, tmp / "out") == 1
    assert not (tmp / "out").exists()


@pytest.mark.parametrize("bad", [
    [[0, 3], [1, 4, 5], [2, 3]],       # expert 3 twice
    [[0, 3], [1, 4], [2]],             # expert 5 missing
    [[3, 0], [1, 4, 5], [2]],          # not increasing
    [[0, 3, 1], [4, 5], [2]],          # not increasing
    [[0, 1, 2, 3, 4, 5], [], []],      # empty tiers
    [[0, 3], [1, 4, 5, 6], [2]],       # expert out of range
])
def test_bad_tables(env, bad):
    tmp, src, _, tpath = env
    tpath.write_text(json.dumps(table(layers=[bad, LAYERS[1]])))
    assert run(src, tpath, tmp / "out") == 1
    assert not (tmp / "out").exists()


def test_bad_table_fields(env):
    tmp, src, _, tpath = env
    for change in ({"mode": "other"}, {"n_layer": 3}, {"tier_types": ["mxfp4"]}, {"n_expert": 5}):
        tpath.write_text(json.dumps({**table(), **change}))
        assert run(src, tpath, tmp / "out") == 1
    tpath.write_text("{}")
    assert run(src, tpath, tmp / "out") == 1


def test_split_key_mismatch(env):
    tmp, src, _, tpath = env
    assert ts.main([str(src / NAMES[1]), str(src / NAMES[0]), "--table", str(tpath), "--out-dir", str(tmp / "o1")]) == 1
    renamed = src / "m-00001-of-00004.gguf"
    renamed.write_bytes((src / NAMES[0]).read_bytes())
    assert ts.main([str(renamed), "--table", str(tpath), "--out-dir", str(tmp / "o2")]) == 1
    plain = src / "plain.gguf"
    plain.write_bytes((src / NAMES[0]).read_bytes())
    assert ts.main([str(plain), "--table", str(tpath), "--out-dir", str(tmp / "o3")]) == 1


def test_wrong_expert_count_in_tensor(env):
    tmp, src, _, tpath = env
    tpath.write_text(json.dumps({**table(), "n_expert": 7, "layers": [[[0, 3], [1, 4, 5], [2, 6]]] * 2}))
    assert run(src, tpath, tmp / "out") == 1
