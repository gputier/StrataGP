#!/usr/bin/env python3
"""check_split_mount.py - can the loader open the split GGUF made of Q8_0 shards 1-7 and Q2_K shards 8-10?

The quality measure builds its reference model R from the experts of the Q8_0 GGUF (shards 1 to 7, MXFP4) and the
rest of the Q2_K oracle (shards 8 to 10: engram tables and dense part), hard-linked under the Q8_0 names.  Nothing
in the loader forbids it, but nothing has run it, so this script checks the headers before any download.

What ik_llama's loader reads (src/llama-model-loader.cpp, 362-416 at c446f1e0): the metadata of the first file only
(architecture, hyperparameters, vocabulary, `split.count`, `split.no`, `split.tensors.count`); of every other shard it
keeps the tensor infos and nothing else, and the sum of the tensors found must equal `split.tensors.count`.  The
model builder then looks tensors up by name and shape, whatever the shard.  `general.file_type` is only a display
name (llama-model-loader.cpp, llama-hparams.cpp:2605, llama.cpp:3018).

Checks, on the mounted file (Q8_0 shards 1-7 plus Q2_K shards 8-10):
  * same `split.count`, `split.no` = 0 on the first shard, and the shard count it announces;
  * the tensors of every shard have the names and shapes of the same shard of the Q2_K oracle (which loads);
  * `split.tensors.count` equals the number of tensors of the mounted shards;
  * shards 1 to 7 hold only `ffn_*_exps` tensors;
  * every metadata key of the Q8_0 first shard against the Q2_K first shard: each difference is listed and flagged
    as mattering or not (an unknown difference matters).

The recorded headers (--q2k, --q80: JSON written in phase 2a) hold the shards 1, 8, 9 and 10 of the Q8_0 only, and
replace the long arrays of the first shard by a placeholder.  --fetch reads what is missing with HTTP `Range`
requests on Hugging Face (headers only, a few MB, no weights): the Q8_0 shards 2 to 7, and the first shard of both
quantizations with a digest of every array (vocabulary included).  Without it, missing shards give the verdict
"indéterminé".

Verdicts, in French as the plan words them: "montage chargeable", "écart", "indéterminé".  The exit code is 0 for
the first, 1 for "écart", 2 for "indéterminé".

    python3 -I tools/v41/check_split_mount.py --q2k q2k_headers.json --q80 q80_headers.json --fetch
        [--out verdict.json] [--save-headers headers.json]
"""
import argparse
import hashlib
import json
import re
import struct
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = "https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-GGUF/resolve/main"
SHARD_URL = REPO + "/{quant}/DeepSeek-V4.1-Flash-1017-{quant}-{no:05d}-of-00010.gguf"
N_SHARDS = 10
FROM_Q80 = range(1, 8)       # shards taken from the Q8_0
FROM_Q2K = range(8, 11)      # shards taken from the Q2_K oracle
EXPERT_TENSOR = re.compile(r"^blk\.\d+\.ffn_(gate|up|down)_exps\.weight$")

VERDICT_OK, VERDICT_GAP, VERDICT_UNKNOWN = "montage chargeable", "écart", "indéterminé"

# metadata keys that differ between the two quantizations without any effect on loading, with the reason
HARMLESS_KEYS = {
    "split.tensors.count": "the Q8_0 value is checked against the tensors of the mounted shards on its own",
    "general.file_type": "display name of the file type only (llama-model-loader.cpp 'general.file_type', "
                         "llama-hparams.cpp:2605, llama.cpp:3018)",
}

GGUF_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K",
              11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
              18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8",
              25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 39: "MXFP4"}
SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
STRING, ARRAY = 8, 9
SHORT_ARRAY = 64


class HeaderError(ValueError):
    """A header that is not a GGUF header, or is cut short."""


class RangeReader:
    """Sequential reads of a remote file through HTTP `Range` requests, one chunk at a time."""

    def __init__(self, url, chunk=8 << 20):
        self.url, self.chunk = url, chunk
        self.buf, self.start = b"", 0       # `buf` holds the file from offset `start`
        self.pos = 0

    def read(self, n):
        while self.pos + n > self.start + len(self.buf):
            got = self.start + len(self.buf)
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={got}-{got + self.chunk - 1}"})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
            except urllib.error.HTTPError as e:
                if e.code != 416:           # the range starts at or beyond the end of the file
                    raise
                data = b""
            if not data:
                raise HeaderError(f"{self.url}: the file ends before byte {self.pos + n}")
            self.buf += data
        a = self.pos - self.start
        self.pos += n
        out = self.buf[a:a + n]
        # drop what was consumed so a long header does not stay in memory
        self.buf, self.start = self.buf[a + n:], self.pos
        return out


class BytesReader:
    """The same sequential interface over bytes already in memory."""

    def __init__(self, data):
        self.data, self.pos = data, 0

    def read(self, n):
        if self.pos + n > len(self.data):
            raise HeaderError("the header ends too early")
        out = self.data[self.pos:self.pos + n]
        self.pos += n
        return out


def unpack(r, fmt):
    return struct.unpack("<" + fmt, r.read(struct.calcsize("<" + fmt)))


def read_string(r):
    (n,) = unpack(r, "Q")
    return r.read(n).decode("utf-8", "replace")


def read_value(r, vtype):
    """A scalar or a string as is; an array as a list when it is short and numeric, else as its length and a
    sha256 of its elements (strings are length-prefixed in the digest), so two vocabularies can be compared."""
    if vtype in SCALARS:
        return unpack(r, SCALARS[vtype])[0]
    if vtype == STRING:
        return read_string(r)
    if vtype == ARRAY:
        (etype,) = unpack(r, "I")
        (n,) = unpack(r, "Q")
        if etype in SCALARS and n <= SHORT_ARRAY:
            return [read_value(r, etype) for _ in range(n)]
        h = hashlib.sha256()
        if etype in SCALARS:
            h.update(r.read(n * struct.calcsize("<" + SCALARS[etype])))
        else:
            for _ in range(n):
                item = read_value(r, etype)
                if not isinstance(item, str):
                    raise HeaderError(f"arrays of arrays are not read (element type {etype})")
                raw = item.encode("utf-8")
                h.update(struct.pack("<Q", len(raw)) + raw)
        return {"array_len": n, "element_type": etype, "sha256": h.hexdigest()}
    raise HeaderError(f"unknown metadata value type {vtype}")


def read_header(r):
    """{'kv': {...}, 'tensors': [[name, dims, type, offset], ...]} of the GGUF header read from `r`."""
    if r.read(4) != b"GGUF":
        raise HeaderError("not a GGUF file")
    (version,) = unpack(r, "I")
    n_tensors, n_kv = unpack(r, "QQ")
    kv = {}
    for _ in range(n_kv):
        key = read_string(r)
        (vtype,) = unpack(r, "I")
        kv[key] = read_value(r, vtype)
    tensors = []
    for _ in range(n_tensors):
        name = read_string(r)
        (ndim,) = unpack(r, "I")
        dims = list(unpack(r, f"{ndim}Q"))
        ttype, offset = unpack(r, "IQ")
        tensors.append([name, dims, GGUF_TYPES.get(ttype, ttype), offset])
    return {"version": version, "kv": kv, "tensors": tensors}


def fetch_header(quant, no, chunk=8 << 20):
    return read_header(RangeReader(SHARD_URL.format(quant=quant, no=no), chunk))


def normalise(recorded):
    """The recorded JSON ({shard: {'kv', 'tensors'}}, keys are strings) as {int shard: header}."""
    return {int(k): v for k, v in recorded.items()}


def kv_differences(a, b):
    """[(key, value in a, value in b)] for every key whose value differs or that only one side has."""
    return [(k, a.get(k, "<absent>"), b.get(k, "<absent>")) for k in sorted(set(a) | set(b)) if a.get(k, "<absent>")
            != b.get(k, "<absent>")]


def shorten(v, n=80):
    s = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
    return s if len(s) <= n else s[:n] + "..."


def check(q2k, q80):
    """The findings on the mount, from headers {shard number: {'kv', 'tensors'}} (`kv` only on shard 1).

    Returns {'verdict', 'reasons', 'checks', 'metadata_differences', ...}; `reasons` lists every cause of an
    "écart" or an "indéterminé"."""
    gaps, unknown, checks = [], [], {}

    missing = [f"Q8_0 shard {n}" for n in FROM_Q80 if n not in q80] + \
              [f"Q8_0 shard {n}" for n in FROM_Q2K if n not in q80] + \
              [f"Q2_K shard {n}" for n in range(1, N_SHARDS + 1) if n not in q2k]
    if missing:
        unknown.append("headers missing: " + ", ".join(missing))
    present = [n for n in range(1, N_SHARDS + 1) if n in q2k and n in q80]

    # tensors by shard: names and shapes against the oracle, for every shard both sides have
    layout = {}
    for n in present:
        names2 = [(t[0], tuple(t[1])) for t in q2k[n]["tensors"]]
        names8 = [(t[0], tuple(t[1])) for t in q80[n]["tensors"]]
        ok = names2 == names8
        layout[n] = {"q2k_tensors": len(names2), "q80_tensors": len(names8), "names_and_shapes_equal": ok,
                     "mounted_from": "Q8_0" if n in FROM_Q80 else "Q2_K"}
        if not ok:
            only2 = sorted(set(names2) - set(names8))[:3]
            only8 = sorted(set(names8) - set(names2))[:3]
            gaps.append(f"shard {n}: tensor names or shapes differ between Q8_0 and Q2_K "
                        f"(Q2_K only {only2}, Q8_0 only {only8})")
    checks["layout_by_shard"] = layout

    # tensors only of the mounted shards
    mounted = {n: (q80 if n in FROM_Q80 else q2k).get(n) for n in range(1, N_SHARDS + 1)}
    if all(mounted[n] is not None for n in mounted):
        total = sum(len(mounted[n]["tensors"]) for n in mounted)
        checks["mounted_tensor_total"] = total
        names = [t[0] for n in mounted for t in mounted[n]["tensors"]]
        if len(set(names)) != len(names):
            gaps.append("a tensor name appears twice in the mounted shards (the loader refuses duplicates)")
        # shards 1 to 7: experts only
        foreign = sorted({t[0] for n in FROM_Q80 for t in mounted[n]["tensors"] if not EXPERT_TENSOR.match(t[0])})
        checks["shards_1_to_7_foreign_tensors"] = foreign
        if foreign:
            gaps.append(f"shards 1 to 7 hold tensors other than ffn_*_exps: {foreign[:5]}")
    else:
        total = None

    # metadata of the first shard (the only one the loader reads)
    kv8 = q80.get(1, {}).get("kv")
    kv2 = q2k.get(1, {}).get("kv")
    if not kv8 or not kv2:
        unknown.append("metadata of the first shard missing on one side")
    else:
        diffs = []
        for key, v2, v8 in kv_differences(kv2, kv8):
            why = HARMLESS_KEYS.get(key)
            diffs.append({"key": key, "q2k": shorten(v2), "q80": shorten(v8), "matters": why is None,
                          "reason": why or "no reason known: treated as mattering"})
        checks["metadata_differences"] = diffs
        checks["metadata_keys"] = {"q2k": len(kv2), "q80": len(kv8)}
        for d in diffs:
            if d["matters"]:
                gaps.append(f"metadata key {d['key']} differs and is not known to be harmless "
                            f"(Q2_K {d['q2k']}, Q8_0 {d['q80']})")
        placeholders = sorted(k for k, v in {**kv2, **kv8}.items() if isinstance(v, str) and v.startswith("<arr "))
        if placeholders:
            unknown.append(f"{len(placeholders)} arrays were recorded as placeholders and are compared by length "
                           f"only (e.g. {placeholders[:3]}); read the first shards with --fetch")
        split = {k: (kv2.get(k), kv8.get(k)) for k in ("split.count", "split.no", "split.tensors.count")}
        checks["split_keys_q2k_q80"] = split
        if kv8.get("split.count") != kv2.get("split.count") or kv8.get("split.count") != N_SHARDS:
            gaps.append(f"split.count differs or is not {N_SHARDS}: {split['split.count']}")
        if kv8.get("split.no") != 0:
            gaps.append(f"the Q8_0 first shard has split.no {kv8.get('split.no')}, not 0")
        if total is not None and kv8.get("split.tensors.count") != total:
            gaps.append(f"split.tensors.count is {kv8.get('split.tensors.count')} but the mounted shards hold "
                        f"{total} tensors")

    verdict = VERDICT_GAP if gaps else VERDICT_UNKNOWN if unknown else VERDICT_OK
    return {"verdict": verdict, "reasons": gaps + unknown, "checks": checks,
            "mounted": {"shards_from_q80": list(FROM_Q80), "shards_from_q2k": list(FROM_Q2K)}}


def format_report(result):
    c = result["checks"]
    lines = [f"verdict: {result['verdict']}"]
    for r in result["reasons"]:
        lines.append(f"  cause: {r}")
    lines.append(f"mounted tensor total: {c.get('mounted_tensor_total')}")
    for n, s in sorted(c["layout_by_shard"].items()):
        lines.append(f"  shard {n:>2} from {s['mounted_from']}: Q2_K {s['q2k_tensors']} / Q8_0 {s['q80_tensors']} "
                     f"tensors, names and shapes {'equal' if s['names_and_shapes_equal'] else 'DIFFER'}")
    lines.append(f"shards 1 to 7, tensors other than ffn_*_exps: {c.get('shards_1_to_7_foreign_tensors')}")
    diffs = c.get("metadata_differences")
    if diffs is not None:
        lines.append(f"metadata keys of the first shard: Q2_K {c['metadata_keys']['q2k']}, "
                     f"Q8_0 {c['metadata_keys']['q80']}; {len(diffs)} differ")
        for d in diffs:
            lines.append(f"  {d['key']}: Q2_K {d['q2k']} | Q8_0 {d['q80']} -> "
                         f"{'MATTERS' if d['matters'] else 'harmless'} ({d['reason']})")
    return "\n".join(lines)


def complete_with_fetch(q2k, q80, chunk=8 << 20):
    """Reads what the recorded headers lack: the Q8_0 shards 2 to 7, and the first shard of both quantizations
    with array digests.  Returns the two header sets and the shards fetched."""
    q2k, q80, fetched = dict(q2k), dict(q80), []
    for n in range(2, 8):
        if n not in q80:
            q80[n] = fetch_header("Q8_0", n, chunk)
            fetched.append(("Q8_0", n))
    for quant, headers in (("Q2_K", q2k), ("Q8_0", q80)):
        headers[1] = fetch_header(quant, 1, chunk)
        fetched.append((quant, 1))
    return q2k, q80, fetched


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--q2k", required=True, help="recorded headers of the Q2_K GGUF (JSON)")
    ap.add_argument("--q80", required=True, help="recorded headers of the Q8_0 GGUF (JSON)")
    ap.add_argument("--fetch", action="store_true", help="read the missing headers on Hugging Face (Range requests)")
    ap.add_argument("--out", help="JSON verdict")
    ap.add_argument("--save-headers", help="write the headers used (JSON), fetched ones included")
    args = ap.parse_args(argv)
    q2k = normalise(json.loads(Path(args.q2k).read_text()))
    q80 = normalise(json.loads(Path(args.q80).read_text()))
    fetched = []
    if args.fetch:
        q2k, q80, fetched = complete_with_fetch(q2k, q80)
    if args.save_headers:
        Path(args.save_headers).write_text(json.dumps({"q2k": q2k, "q80": q80}) + "\n")
    result = check(q2k, q80)
    result["fetched"] = [f"{q} shard {n}" for q, n in fetched]
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(format_report(result))
    return {VERDICT_OK: 0, VERDICT_GAP: 1, VERDICT_UNKNOWN: 2}[result["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
