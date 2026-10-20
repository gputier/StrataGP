"""Tests of check_split_mount.py on synthetic headers, expected values worked out by hand.

    python -m pytest tools/v41/test_check_split_mount.py
"""
import copy
import hashlib
import json
import struct
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_split_mount as cm  # noqa: E402

# a toy model: 10 shards, expert shards 1-7 (2 tensors each), shard 8 and 9 one table each, shard 10 two tensors
EXPERT_SHARDS = {n: [f"blk.{n}.ffn_gate_exps.weight", f"blk.{n}.ffn_up_exps.weight"] for n in range(1, 8)}


def tensors(names, ttype, shape=(4, 4, 2)):
    return [[n, list(shape), ttype, 32 * i] for i, n in enumerate(names)]


def make(expert_type, rest_type):
    """Headers {shard: {'kv', 'tensors'}} of a toy quantization."""
    h = {n: {"kv": {}, "tensors": tensors(names, expert_type)} for n, names in EXPERT_SHARDS.items()}
    h[8] = {"kv": {}, "tensors": tensors(["blk.1.engram_embd.weight"], rest_type, (2, 9))}
    h[9] = {"kv": {}, "tensors": tensors(["blk.14.engram_embd.weight"], rest_type, (2, 9))}
    h[10] = {"kv": {}, "tensors": tensors(["token_embd.weight", "output.weight"], rest_type, (3, 5))}
    h[1]["kv"] = {"general.architecture": "deepseek41", "general.file_type": 10, "deepseek41.embedding_length": 5120,
                  "split.no": 0, "split.count": 10, "split.tensors.count": 18,
                  "tokenizer.ggml.tokens": {"array_len": 3, "element_type": 8, "sha256": "aa"}}
    return h


@pytest.fixture
def pair():
    q2k = make("Q2_K", "Q2_K")
    q80 = make("MXFP4", "Q8_0")
    q80[1]["kv"] = {**q2k[1]["kv"], "general.file_type": 38}
    return q2k, q80


def test_a_compatible_mount_is_loadable(pair):
    r = cm.check(*pair)
    assert r["verdict"] == "montage chargeable"
    assert r["reasons"] == []
    assert r["checks"]["mounted_tensor_total"] == 18       # 7 x 2 + 1 + 1 + 2
    assert r["checks"]["shards_1_to_7_foreign_tensors"] == []
    # the only metadata difference is the file type, which is a display name
    diffs = r["checks"]["metadata_differences"]
    assert [(d["key"], d["matters"]) for d in diffs] == [("general.file_type", False)]


def test_a_shape_that_differs_in_one_shard_is_a_gap(pair):
    q2k, q80 = pair
    q80[3]["tensors"][0][1] = [4, 4, 3]
    r = cm.check(q2k, q80)
    assert r["verdict"] == "écart"
    assert "shard 3" in r["reasons"][0]


def test_a_renamed_tensor_in_a_replaced_shard_is_a_gap(pair):
    q2k, q80 = pair
    q80[9]["tensors"][0][0] = "blk.15.engram_embd.weight"
    r = cm.check(q2k, q80)
    assert r["verdict"] == "écart"
    assert "shard 9" in r["reasons"][0]


def test_an_expert_that_is_not_mxfp4_in_the_expert_shards_is_a_gap(pair):
    q2k, q80 = pair
    q80[4]["tensors"][1][2] = "Q8_0"
    r = cm.check(q2k, q80)
    assert r["verdict"] == "écart"
    assert r["checks"]["shards_1_to_7_expert_types_not_mxfp4"] == ["blk.4.ffn_up_exps.weight (Q8_0)"]
    assert "1 expert tensors are not in MXFP4" in r["reasons"][0]


def test_the_same_tensors_in_another_order_are_still_loadable(pair):
    q2k, q80 = pair
    q80[3]["tensors"].reverse()
    r = cm.check(q2k, q80)
    assert r["verdict"] == "montage chargeable"
    assert r["checks"]["layout_by_shard"][3]["names_and_shapes_equal"]


def test_a_dense_tensor_in_the_expert_shards_is_a_gap(pair):
    q2k, q80 = pair
    for h in (q2k, q80):
        h[2]["tensors"][1][0] = "blk.2.attn_q_a.weight"
    r = cm.check(q2k, q80)
    assert r["verdict"] == "écart"
    assert r["checks"]["shards_1_to_7_foreign_tensors"] == ["blk.2.attn_q_a.weight"]


def test_the_announced_tensor_total_must_match_the_mounted_shards(pair):
    q2k, q80 = pair
    q80[1]["kv"]["split.tensors.count"] = 19
    r = cm.check(q2k, q80)
    assert r["verdict"] == "écart"
    assert "split.tensors.count is 19 but the mounted shards hold 18" in r["reasons"][0]


def test_a_total_that_counts_the_replaced_shards_of_the_other_file_is_caught(pair):
    # a Q8_0 shard 10 with one more tensor than the Q2_K one: the mounted total (Q2_K shard 10) is still 18
    q2k, q80 = pair
    q80[10]["tensors"].append(["extra.weight", [1], "Q8_0", 99])
    q80[1]["kv"]["split.tensors.count"] = 19
    r = cm.check(q2k, q80)
    assert r["verdict"] == "écart"
    assert any("shard 10" in x for x in r["reasons"])


def test_split_count_and_number_are_checked(pair):
    q2k, q80 = pair
    q80[1]["kv"]["split.count"] = 9
    q80[1]["kv"]["split.no"] = 1
    r = cm.check(q2k, q80)
    assert r["verdict"] == "écart"
    assert any("split.count" in x for x in r["reasons"]) and any("split.no" in x for x in r["reasons"])


def test_an_unknown_metadata_difference_matters(pair):
    q2k, q80 = pair
    q80[1]["kv"]["deepseek41.embedding_length"] = 4096
    r = cm.check(q2k, q80)
    assert r["verdict"] == "écart"
    assert any("deepseek41.embedding_length" in x for x in r["reasons"])
    d = {x["key"]: x for x in r["checks"]["metadata_differences"]}
    assert d["deepseek41.embedding_length"]["matters"] and not d["general.file_type"]["matters"]


def test_a_key_that_only_one_side_has_is_a_difference(pair):
    q2k, q80 = pair
    del q80[1]["kv"]["tokenizer.ggml.tokens"]
    r = cm.check(q2k, q80)
    assert r["verdict"] == "écart"


def test_a_different_vocabulary_digest_is_a_gap(pair):
    q2k, q80 = pair
    q80[1]["kv"]["tokenizer.ggml.tokens"] = {"array_len": 3, "element_type": 8, "sha256": "bb"}
    assert cm.check(q2k, q80)["verdict"] == "écart"


def test_missing_shards_make_the_verdict_unknown_not_loadable(pair):
    q2k, q80 = pair
    for n in range(2, 8):
        del q80[n]
    r = cm.check(q2k, q80)
    assert r["verdict"] == "indéterminé"
    assert "Q8_0 shard 2" in r["reasons"][0]


def test_placeholders_for_long_arrays_make_the_verdict_unknown(pair):
    q2k, q80 = pair
    q2k[1]["kv"]["tokenizer.ggml.merges"] = "<arr 100000>"
    q80[1]["kv"]["tokenizer.ggml.merges"] = "<arr 100000>"
    r = cm.check(q2k, q80)
    assert r["verdict"] == "indéterminé"
    assert "placeholders" in r["reasons"][0]


def test_a_gap_wins_over_an_unknown(pair):
    q2k, q80 = pair
    del q80[4]
    q80[3]["tensors"][0][1] = [9]
    assert cm.check(q2k, q80)["verdict"] == "écart"


def test_the_input_headers_are_not_modified(pair):
    before = copy.deepcopy(pair)
    cm.check(*pair)
    assert pair == before


# ---- the GGUF reader

def gguf_bytes(kv_items, tensor_items):
    """A GGUF v3 header: kv_items [(key, type id, payload bytes)], tensor_items [(name, dims, type id, offset)]."""
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensor_items), len(kv_items))
    for key, vtype, payload in kv_items:
        out += string(key) + struct.pack("<I", vtype) + payload
    for name, dims, ttype, offset in tensor_items:
        out += string(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
        out += struct.pack("<IQ", ttype, offset)
    return out


def string(text):
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


SAMPLE_KV = [
    ("general.architecture", 8, string("deepseek41")),
    ("split.count", 2, struct.pack("<H", 10)),
    ("deepseek41.block_count", 4, struct.pack("<I", 40)),
    ("short", 9, struct.pack("<IQ", 4, 2) + struct.pack("<II", 7, 9)),
    ("vocab", 9, struct.pack("<IQ", 8, 3) + string("a") + string("bc") + string("")),
    ("ids", 9, struct.pack("<IQ", 4, 100) + struct.pack("<100I", *range(100))),
]
SAMPLE_TENSORS = [("blk.0.ffn_up_exps.weight", [5, 4, 3], 39, 0), ("token_embd.weight", [8], 8, 64)]


def test_the_reader_parses_a_header_and_digests_the_arrays():
    h = cm.read_header(cm.BytesReader(gguf_bytes(SAMPLE_KV, SAMPLE_TENSORS)))
    assert h["version"] == 3
    kv = h["kv"]
    assert kv["general.architecture"] == "deepseek41" and kv["split.count"] == 10
    assert kv["deepseek41.block_count"] == 40
    assert kv["short"] == [7, 9]                        # short numeric arrays are kept as they are
    # strings are hashed with an 8-byte length in front of each
    expect = hashlib.sha256(struct.pack("<Q", 1) + b"a" + struct.pack("<Q", 2) + b"bc" + struct.pack("<Q", 0))
    assert kv["vocab"] == {"array_len": 3, "element_type": 8, "sha256": expect.hexdigest()}
    # a long numeric array is hashed over its raw bytes
    assert kv["ids"] == {"array_len": 100, "element_type": 4,
                         "sha256": hashlib.sha256(struct.pack("<100I", *range(100))).hexdigest()}
    assert h["tensors"] == [["blk.0.ffn_up_exps.weight", [5, 4, 3], "MXFP4", 0], ["token_embd.weight", [8], "Q8_0", 64]]


def test_the_reader_refuses_a_bad_magic_and_a_cut_header():
    data = gguf_bytes(SAMPLE_KV, SAMPLE_TENSORS)
    with pytest.raises(cm.HeaderError, match="not a GGUF"):
        cm.read_header(cm.BytesReader(b"GGML" + data[4:]))
    with pytest.raises(cm.HeaderError, match="ends too early"):
        cm.read_header(cm.BytesReader(data[:-5]))


def serve(data, honour_range=True):
    """A local HTTP server answering `Range: bytes=a-b` of `data` with 206 and its Content-Range (416 past the end);
    with `honour_range` false it ignores the header and answers 200 with the whole file.  Returns (url, server)."""
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            a, b = self.headers["Range"].split("=")[1].split("-")
            a, b = int(a), int(b)
            body = data[a:b + 1] if honour_range else data
            self.send_response((206 if body else 416) if honour_range else 200)
            if honour_range and body:
                self.send_header("Content-Range", f"bytes {a}-{a + len(body) - 1}/{len(data)}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass
    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_port}/f.gguf", server


def test_the_range_reader_reads_a_header_across_chunk_boundaries():
    data = gguf_bytes(SAMPLE_KV, SAMPLE_TENSORS) + b"\0" * 1000          # weights follow the header
    url, server = serve(data)
    try:
        for chunk in (7, 64, 1 << 20):                                   # a chunk smaller than a value, and larger
            h = cm.read_header(cm.RangeReader(url, chunk))
            assert h["tensors"][1] == ["token_embd.weight", [8], "Q8_0", 64]
            assert h["kv"]["vocab"]["array_len"] == 3
    finally:
        server.shutdown()


def test_the_range_reader_refuses_a_file_that_ends_early():
    url, server = serve(b"GGUF\0\0")
    try:
        with pytest.raises(cm.HeaderError, match="ends before"):
            cm.read_header(cm.RangeReader(url, 16))
    finally:
        server.shutdown()


def test_a_server_that_ignores_range_is_refused():
    # a 200 with the whole file would put the header bytes at the wrong place without any error
    url, server = serve(gguf_bytes(SAMPLE_KV, SAMPLE_TENSORS), honour_range=False)
    try:
        with pytest.raises(cm.HeaderError, match="answered 200"):
            cm.read_header(cm.RangeReader(url, 64))
    finally:
        server.shutdown()


# ---- the command line

def test_main_writes_the_verdict_and_returns_the_exit_code(pair, tmp_path, capsys):
    q2k, q80 = pair
    files = {}
    for name, h in (("q2k", q2k), ("q80", q80)):
        files[name] = tmp_path / f"{name}.json"
        files[name].write_text(json.dumps({str(k): v for k, v in h.items()}))
    out = tmp_path / "verdict.json"
    assert cm.main(["--q2k", str(files["q2k"]), "--q80", str(files["q80"]), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["verdict"] == "montage chargeable"
    assert "verdict: montage chargeable" in capsys.readouterr().out
    q80[3]["tensors"][0][1] = [1]
    files["q80"].write_text(json.dumps({str(k): v for k, v in q80.items()}))
    assert cm.main(["--q2k", str(files["q2k"]), "--q80", str(files["q80"])]) == 1
    del q80[3]
    files["q80"].write_text(json.dumps({str(k): v for k, v in q80.items()}))
    assert cm.main(["--q2k", str(files["q2k"]), "--q80", str(files["q80"])]) == 2
