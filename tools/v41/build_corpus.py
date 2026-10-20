#!/usr/bin/env python3
"""build_corpus.py - the two-half calibration corpus for the DeepSeek-V4.1 expert-usage measurement.

Writes `half_a.txt`, `half_b.txt` and `manifest.json` to --out-dir.  Each half holds about --tokens-per-half
tokens, half of it usage and half of it wikitext-2, with the same composition in both halves and no source
shared between them (a file, a doc or a wikitext article sits in one half only).

Usage part, rendered with DeepSeek's own `encode_messages(msgs, thinking_mode='chat')` (--encoding-py):
  review        a code-review request on a tracked file of the repository
  explain_doc   a request to explain a document of docs/ in French
  explain_code  a request to explain a source file in French
  tool_read     the chat template's tool calling: `tools` in the system message, a read_file call and its result
  tool_grep     the same with a grep call and its matches
  short         a short technical question in French
  long          one long prompt of about --long-tokens tokens per half: several files and a synthesis question
Files are read with `git show <rev>:<path>` (--repo, --rev) so only tracked files are used, or from a
`git archive` tar file (--source-tar), which holds the same bytes; the tar is hashed into the manifest and --rev is then
only a label.  Nothing from the wiki or the second brain.

Wikitext part: wikitext-2-raw-v1 (--wikitext, a jsonl of rows, fetched by --fetch-wikitext from Hugging Face),
cut at its level-1 headings into articles, which are dealt alternately to the two halves.

Tokens are counted with the real tokenizer when --tokenizer (tokenizer.json, `tokenizers` library) is given;
without it a characters-per-token estimate stands in and the manifest says so.  The split is deterministic
(--seed).

    python tools/v41/build_corpus.py --out-dir corpus --encoding-py encoding.py --wikitext wikitext.jsonl
                                     --tokenizer tokenizer.json [--repo .] [--rev HEAD]
"""
import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import random
import re
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

SEED = 41
CHARS_PER_TOKEN = 3.0     # the stand-in estimate when no tokenizer is given

CODE_DIRS = ("src/", "include/", "serve/", "tools/", "tests/", "sycl/", "metal/")
CODE_EXT = {".cpp", ".hpp", ".h", ".cu", ".cuh", ".py", ".sh", ".metal"}
DOC_DIR, DOC_EXT = "docs/", {".md"}
SKIP_PARTS = ("third_party/", "tools/v41/", "/media/", "/data/")
MIN_CHARS, MAX_BYTES = 1500, 400_000
LONG_FILE_CHARS = 24_000   # a file of the long prompt is cut here, so the prompt spans many files

KINDS = ("review", "explain_doc", "explain_code", "tool_read", "tool_grep", "short")

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Lit un fichier du dépôt et renvoie son contenu.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Chemin relatif à la racine du dépôt"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "grep", "description": "Cherche un motif dans un fichier et renvoie les lignes qui correspondent.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "Mot ou expression à chercher"},
            "path": {"type": "string", "description": "Fichier dans lequel chercher"}},
            "required": ["pattern", "path"]}}},
    {"type": "function", "function": {
        "name": "list_dir", "description": "Liste le contenu d'un dossier du dépôt.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Dossier à lister"}}, "required": ["path"]}}},
]

REVIEW_ASKS = [
    "Fais une revue de code du fichier `{path}` ci-dessous. Signale les bugs probables, les risques de "
    "concurrence ou de dépassement, et ce qui nuit à la lisibilité. Cite les lignes concernées.",
    "Relis `{path}` comme si tu devais l'approuver en revue. Quels sont les trois défauts les plus graves, "
    "et comment les corrigerais-tu ?",
    "Ce fichier (`{path}`) vient d'être modifié. Cherche les erreurs de logique, les cas limites non "
    "traités et les fuites de ressources.",
    "Peux-tu auditer `{path}` : gestion des erreurs, propriétés de mémoire, nommage, et tests qui manquent ?",
]
EXPLAIN_CODE_ASKS = [
    "Explique en français, pas à pas, comment fonctionne `{path}` et quelle est sa place dans le moteur.",
    "Je découvre `{path}`. Résume son rôle en français et décris les fonctions principales.",
    "Quel problème `{path}` résout-il ? Explique son fonctionnement à un développeur qui ne connaît pas le projet.",
]
EXPLAIN_DOC_ASKS = [
    "Explique en français, à quelqu'un qui découvre le projet, ce que dit ce document et ce qu'il faut en retenir.",
    "Résume ce document en français, puis liste les points qui demandent une action de la part d'un utilisateur.",
    "Reformule ce document en français simple. Quelles sont les limites ou les pièges qu'il signale ?",
]
CODE_ASKS = {"review": REVIEW_ASKS, "explain_code": EXPLAIN_CODE_ASKS}
TOOL_ASKS = [
    "Regarde `{path}` et dis-moi ce que fait la première fonction.",
    "Ouvre `{path}` et explique-moi son rôle en quelques phrases.",
    "J'ai un doute sur `{path}` : peux-tu le lire et me dire s'il y a un risque évident ?",
]
GREP_ASKS = [
    "Où est utilisé `{word}` dans `{path}` ? Donne-moi les lignes et ce qu'elles font.",
    "Cherche `{word}` dans `{path}` et explique comment cet identifiant est employé.",
]
SHORT_QUESTIONS = [
    "Explique en dix lignes la différence entre la RAM et la VRAM à un débutant.",
    "Quelle est la différence entre un mutex et un sémaphore ? Donne un exemple en C++.",
    "Comment fonctionne le cache L2 d'un processeur et pourquoi l'alignement des données compte-t-il ?",
    "Écris une fonction Python qui fusionne des intervalles qui se chevauchent, avec ses tests pytest.",
    "Explique ce qu'est la quantification d'un modèle de langage et ce qu'elle coûte en qualité.",
    "Qu'est-ce qu'un modèle à mélange d'experts et pourquoi n'active-t-il que quelques experts par jeton ?",
    "Pourquoi le débit d'un disque NVMe dépend-il de la profondeur de file des requêtes ?",
    "Décris la différence entre un tri stable et un tri instable, avec un cas où cela change le résultat.",
    "Comment découper un gros fichier en morceaux pour le lire sans saturer la mémoire ?",
    "Explique le rôle du cache KV dans l'inférence d'un transformeur.",
    "Quelle est la complexité d'une recherche dans une table de hachage, et dans quels cas se dégrade-t-elle ?",
    "Écris une requête SQL qui renvoie les dix clients les plus actifs du dernier mois.",
    "Qu'est-ce qu'une condition de course ? Montre un exemple et sa correction.",
    "Explique la différence entre le parallélisme de données et le parallélisme de pipeline sur plusieurs GPU.",
    "Comment vérifier qu'une sauvegarde est restaurable sans toucher à la production ?",
    "Que fait `git rebase` par rapport à `git merge`, et quand éviter le rebase ?",
    "Pourquoi un test qui passe en local peut-il échouer en intégration continue ?",
    "Explique ce qu'est un deadlock et comment l'éviter avec un ordre de verrouillage.",
    "Décris le fonctionnement d'un allocateur en arène et son avantage sur malloc.",
    "Comment mesurer proprement la latence d'un service HTTP, avec les percentiles ?",
    "Explique la différence entre un entier signé et non signé lors d'un dépassement en C.",
    "Quels sont les avantages d'un format de fichier binaire à enregistrements de taille fixe ?",
    "Écris un script shell qui archive un dossier et vérifie l'archive avant de supprimer l'original.",
    "Explique comment un compilateur décide d'inliner une fonction.",
    "Quelle est la différence entre la perplexité et la précision d'un modèle de langage ?",
    "Pourquoi le premier appel d'un noyau GPU est-il plus lent que les suivants ?",
    "Décris un plan de migration de base de données sans interruption de service.",
    "Que signifie qu'une fonction est idempotente ? Donne un exemple pour une API.",
    "Explique le principe d'une file d'attente avec accusés de réception et reprise sur erreur.",
    "Comment organiser des logs pour qu'on retrouve la cause d'un incident en quelques minutes ?",
]
LONG_ASK = ("Voici plusieurs fichiers du dépôt StrataGP. Question de synthèse : donne en une vingtaine de lignes "
            "le rôle de chacun de ces fichiers, puis dis lesquels dépendent les uns des autres et lesquels "
            "peuvent changer sans toucher aux autres.")


# ---- tokens

class Counter:
    """Token counts: the real tokenizer when tokenizer.json is given, else an estimate."""

    def __init__(self, tokenizer_path=None):
        self.exact = tokenizer_path is not None
        self.tokenizer = None
        if self.exact:
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_file(str(tokenizer_path))

    def count(self, text):
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False).ids)
        return math.ceil(len(text) / CHARS_PER_TOKEN)


def fit_prefix(counter, units, budget):
    """The longest run of leading `units` (lines) that holds at most `budget` tokens, joined."""
    lo, hi = 0, len(units)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if counter.count("".join(units[:mid])) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return "".join(units[:lo])


def head(text, n):
    """The first n characters of text, cut at a line end."""
    if len(text) <= n:
        return text
    cut = text.rfind("\n", 0, n)
    return text[:cut + 1] if cut > 0 else text[:n]


def fence(path, text):
    return f"=== {path} ===\n{text}\n=== fin de {path} ===\n"


# ---- sources

def classify(path, size):
    """'code', 'doc' or None for a tracked path."""
    if size < MIN_CHARS or size > MAX_BYTES or any(part in "/" + path for part in SKIP_PARTS):
        return None
    ext = Path(path).suffix
    if path.startswith(DOC_DIR) and ext in DOC_EXT:
        return "doc"
    if path.startswith(CODE_DIRS) and ext in CODE_EXT:
        return "code"
    return None


def clean_text(blob):
    """The text of a file, or None when it cannot be a plain source: not UTF-8, binary, or carrying the characters
    of the chat template's special tokens (the tokenizer would read them as tokens, not as text)."""
    try:
        text = blob.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if "\0" in text or "｜" in text or "▁" in text:
        return None
    return text.replace("\r\n", "\n")


def git_candidates(repo, rev):
    """[(kind, path, text)] of the tracked files at `rev`, read with `git show rev:path`."""
    listing = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "-l", "-z", rev], check=True,
                             capture_output=True).stdout.decode("utf-8", "surrogateescape")
    found = []
    for entry in filter(None, listing.split("\0")):
        meta, path = entry.split("\t", 1)
        mode, kind, _sha, size = meta.split()
        if kind != "blob" or mode == "120000":
            continue
        k = classify(path, int(size))
        if k is None:
            continue
        blob = subprocess.run(["git", "-C", str(repo), "show", f"{rev}:{path}"], check=True,
                              capture_output=True).stdout
        text = clean_text(blob)
        if text is not None and len(text) >= MIN_CHARS:
            found.append((k, path, text))
    return sorted(found, key=lambda c: c[1])


def tar_candidates(source):
    """The same list from a `git archive` tar file."""
    found = []
    with open(source, "rb") as fileobj, tarfile.open(fileobj=fileobj, mode="r|*") as tar:
        for member in tar:
            if not member.isreg():
                continue
            k = classify(member.name, member.size)
            if k is None:
                continue
            text = clean_text(tar.extractfile(member).read())
            if text is not None and len(text) >= MIN_CHARS:
                found.append((k, member.name, text))
    return sorted(found, key=lambda c: c[1])


def read_rows(path):
    """The texts of a jsonl of dataset rows, in order."""
    return [json.loads(line)["text"] for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def rows_to_lines(rows):
    """The pieces of text of the rows: an empty row becomes a bare newline."""
    return [r if r else "\n" for r in rows]


def rows_to_text(rows):
    """The plain text of the rows, joined (the text `llama-perplexity -f` reads)."""
    return "".join(rows_to_lines(rows))


def load_wikitext(path):
    """The articles [(title, text)] of the wikitext rows: cut at the level-1 headings (` = Title = `)."""
    lines = rows_to_lines(read_rows(path))
    heading = re.compile(r"^ = [^=].* = \n?$")
    articles, title, buf = [], None, []
    for line in lines:
        if heading.match(line):
            if title is not None:
                articles.append((title, "".join(buf)))
            title, buf = line.strip(" =\n"), []
        if title is not None:
            buf.append(line)
    if title is not None:
        articles.append((title, "".join(buf)))
    return articles


def fetch_wikitext(path, split="test"):
    """Download wikitext-2-raw-v1 from the Hugging Face datasets server into `path`, one JSON row per line."""
    base = ("https://datasets-server.huggingface.co/rows?dataset=Salesforce/wikitext&config=wikitext-2-raw-v1"
            f"&split={split}")
    rows, offset = [], 0
    while True:
        with urllib.request.urlopen(f"{base}&offset={offset}&length=100", timeout=60) as r:
            page = json.load(r)
        rows += [x["row"]["text"] for x in page["rows"]]
        offset += 100
        if offset >= page["num_rows_total"]:
            break
    Path(path).write_text("".join(json.dumps({"text": t}, ensure_ascii=False) + "\n" for t in rows),
                          encoding="utf-8")


def load_encoder(encoding_py):
    spec = importlib.util.spec_from_file_location("deepseek_encoding", encoding_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return lambda messages: module.encode_messages(messages, thinking_mode="chat")


# ---- usage items

def pick_word(text, rng):
    """An identifier that occurs several times in text, for a grep question."""
    words = {}
    for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]{5,}", text):
        words[w] = words.get(w, 0) + 1
    common = sorted(w for w, n in words.items() if n >= 3)
    if not words:
        raise ValueError("no identifier of 6 characters or more to ask about")
    return rng.choice(common) if common else sorted(words)[0]


def grep_result(path, text, word, limit=14):
    hits = [f"{path}:{i}:{line}" for i, line in enumerate(text.splitlines(), 1) if word in line]
    return "\n".join(hits[:limit]) if hits else f"(aucune ligne ne contient {word})"


def render_item(kind, source, rng, encode, file_chars):
    """The rendered prompt of one usage item.  `source` is (path, text) or a question."""
    if kind == "short":
        return encode([{"role": "user", "content": source}])
    path, text = source
    body = head(text, file_chars)
    if kind in CODE_ASKS:
        ask = rng.choice(CODE_ASKS[kind]).format(path=path)
        return encode([{"role": "user", "content": f"{ask}\n\n```\n{body}\n```"}])
    if kind == "explain_doc":
        ask = rng.choice(EXPLAIN_DOC_ASKS)
        return encode([{"role": "user", "content": f"{ask}\n\n{fence(path, body)}"}])
    system = {"role": "system", "content": "Tu es un assistant de développement. Tu peux appeler des outils "
                                           "pour lire le dépôt.", "tools": TOOLS}
    if kind == "tool_read":
        ask = rng.choice(TOOL_ASKS).format(path=path)
        call = {"id": "call_0", "type": "function",
                "function": {"name": "read_file", "arguments": json.dumps({"path": path})}}
        result = body
    else:
        word = pick_word(text, rng)
        ask = rng.choice(GREP_ASKS).format(path=path, word=word)
        call = {"id": "call_0", "type": "function",
                "function": {"name": "grep", "arguments": json.dumps({"pattern": word, "path": path})}}
        result = grep_result(path, text, word)
    return encode([system, {"role": "user", "content": ask},
                   {"role": "assistant", "content": "", "tool_calls": [call]},
                   {"role": "tool", "tool_call_id": "call_0", "content": result}])


def render_long(files, counter, encode, tokens):
    """One long prompt of about `tokens` tokens: whole files in turn, the last one cut, then the question."""
    overhead = counter.count(encode([{"role": "user", "content": "Voici plusieurs fichiers du dépôt StrataGP.\n\n"
                                                                  + LONG_ASK}]))
    left = tokens - overhead
    parts, used = [], []
    for path, text in files:
        whole = fence(path, text)
        n = counter.count(whole)
        if n <= left:
            parts.append(whole)
            used.append(path)
            left -= n
            continue
        cut = fit_prefix(counter, text.splitlines(keepends=True), left - 40)
        if cut:
            parts.append(fence(path, cut))
            used.append(path)
        break
    else:
        raise SystemExit("build_corpus: not enough files to reach the long prompt's token budget")
    content = "Voici plusieurs fichiers du dépôt StrataGP.\n\n" + "".join(parts) + "\n" + LONG_ASK
    return encode([{"role": "user", "content": content}]), used


# ---- the corpus

def split_dealt(items, rng):
    """Shuffle, then deal alternately: two disjoint lists whose sizes differ by at most one."""
    items = list(items)
    rng.shuffle(items)
    return items[0::2], items[1::2]


def build(candidates, articles, questions, encode, counter, *, tokens_per_half, long_tokens, file_chars, seed):
    """The two halves: ({'A': text, 'B': text}, manifest parts).  Pure: no file, no git."""
    rng = random.Random(seed)
    pools = {}
    for kind in ("code", "doc"):
        pools[kind] = split_dealt([(p, t) for k, p, t in candidates if k == kind], rng)
    shorts = split_dealt(questions, rng)
    arts = split_dealt(articles, rng)
    halves = {}
    for i, name in enumerate("AB"):
        half_rng = random.Random(f"{seed}/{name}")
        code, docs = list(pools["code"][i]), list(pools["doc"][i])
        # the long prompt takes files from the front of the half's pools, alternating code and doc
        long_files = [(p, head(t, LONG_FILE_CHARS))
                      for pair in itertools.zip_longest(code, docs) for p, t in filter(None, pair)]
        long_prompt, long_used = render_long(long_files, counter, encode, long_tokens)
        # the files the long prompt used are not reused below
        code = [c for c in code if c[0] not in long_used]
        docs = [d for d in docs if d[0] not in long_used]
        halves[name] = {"code": code, "doc": docs, "short": list(shorts[i]), "articles": list(arts[i]),
                        "long": (long_prompt, long_used), "rng": half_rng, "items": [], "tokens": 0}
    # usage items in lockstep so the halves get the same kinds in the same order
    usage_budget = tokens_per_half // 2
    for h in halves.values():
        h["items"].append({"kind": "long", "sources": h["long"][1], "text": h["long"][0]})
        h["tokens"] = counter.count(h["long"][0])
    sources_of = {"review": "code", "explain_code": "code", "tool_read": "code", "tool_grep": "code",
                  "explain_doc": "doc", "short": "short"}
    while any(h["tokens"] < usage_budget for h in halves.values()):
        for kind in KINDS:
            for h in halves.values():
                if h["tokens"] >= usage_budget:
                    continue
                pool = h[sources_of[kind]]
                if not pool:
                    raise SystemExit(f"build_corpus: the {sources_of[kind]} pool ran out before the budget")
                src = pool.pop(0)
                text = render_item(kind, src, h["rng"], encode, file_chars)
                h["items"].append({"kind": kind, "sources": [src if kind == "short" else src[0]], "text": text})
                h["tokens"] += counter.count(text)
    # wikitext fills each half to the target, whole articles then the last one cut at a line
    for h in halves.values():
        budget = tokens_per_half - h["tokens"]
        if budget <= 0:
            raise SystemExit("build_corpus: the usage part already exceeds --tokens-per-half")
        for title, text in h["articles"]:
            n = counter.count(text)
            if n <= budget:
                h["items"].append({"kind": "wikitext", "sources": [title], "text": text})
                budget -= n
                continue
            cut = fit_prefix(counter, text.splitlines(keepends=True), budget)
            if cut:
                h["items"].append({"kind": "wikitext", "sources": [title + " (cut)"], "text": cut})
            budget = 0
            break
        if budget > 0:
            raise SystemExit("build_corpus: the wikitext pool ran out before the budget")
    texts, parts = {}, {}
    for name, h in halves.items():
        h["rng"].shuffle(h["items"])
        texts[name] = "\n".join(it["text"].rstrip("\n") for it in h["items"]) + "\n"
        parts[name] = h["items"]
    return texts, parts


def manifest_for(texts, parts, counter, params):
    out = {"params": params, "tokens_exact": counter.exact, "halves": {}}
    for name in "AB":
        by_kind = {}
        for it in parts[name]:
            row = by_kind.setdefault(it["kind"], {"items": 0, "chars": 0, "tokens": 0})
            row["items"] += 1
            row["chars"] += len(it["text"])
            row["tokens"] += counter.count(it["text"])
        usage = sum(r["tokens"] for k, r in by_kind.items() if k != "wikitext")
        out["halves"][name] = {
            "file": f"half_{name.lower()}.txt",
            "chars": len(texts[name]),
            "tokens": counter.count(texts[name]),
            "tokens_usage": usage,
            "tokens_wikitext": by_kind.get("wikitext", {}).get("tokens", 0),
            "sha256": hashlib.sha256(texts[name].encode("utf-8")).hexdigest(),
            "by_kind": by_kind,
            "sources": [{"kind": it["kind"], "source": it["sources"]} for it in parts[name]],
        }
    return out


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", required=True, help="where half_a.txt, half_b.txt and manifest.json go")
    ap.add_argument("--encoding-py", required=True, help="DeepSeek-V4.1's encoding.py")
    ap.add_argument("--wikitext", required=True, help="wikitext-2-raw-v1 rows, one JSON object per line")
    ap.add_argument("--fetch-wikitext", action="store_true", help="download --wikitext from Hugging Face first")
    ap.add_argument("--tokenizer", help="tokenizer.json of DeepSeek-V4.1 (exact token counts)")
    ap.add_argument("--repo", default=".", help="git repository to read (default: the current directory)")
    ap.add_argument("--rev", default="HEAD",
                    help="revision read with git show; only a label when --source-tar, which is hashed instead")
    ap.add_argument("--source-tar", help="a `git archive` tar file to read instead of --repo (not '-': it is hashed)")
    ap.add_argument("--tokens-per-half", type=int, default=100_000)
    ap.add_argument("--long-tokens", type=int, default=30_000, help="size of each half's long prompt")
    ap.add_argument("--file-chars", type=int, default=3000, help="characters of a file kept in an ordinary item")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)
    if a.source_tar == "-":
        ap.error("--source-tar needs a file, not stdin: the archive is hashed into the manifest")

    if a.fetch_wikitext:
        fetch_wikitext(a.wikitext)
    counter = Counter(a.tokenizer)
    encode = load_encoder(a.encoding_py)
    if a.source_tar:
        candidates, origin = tar_candidates(a.source_tar), {"tar": a.source_tar, "rev": a.rev}
    else:
        rev = subprocess.run(["git", "-C", a.repo, "rev-parse", a.rev], check=True, capture_output=True,
                             text=True).stdout.strip()
        candidates, origin = git_candidates(a.repo, rev), {"git": a.repo, "rev": rev}
    texts, parts = build(candidates, load_wikitext(a.wikitext), SHORT_QUESTIONS, encode, counter,
                         tokens_per_half=a.tokens_per_half, long_tokens=a.long_tokens, file_chars=a.file_chars,
                         seed=a.seed)
    params = {"seed": a.seed, "tokens_per_half": a.tokens_per_half, "long_tokens": a.long_tokens,
              "file_chars": a.file_chars, "source": origin, "candidates": len(candidates),
              "source_tar_sha256": sha256_of(a.source_tar) if a.source_tar else None,
              "wikitext_sha256": sha256_of(a.wikitext), "encoding_py_sha256": sha256_of(a.encoding_py),
              "tokenizer_sha256": sha256_of(a.tokenizer) if a.tokenizer else None}
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in "AB":
        (out / f"half_{name.lower()}.txt").write_text(texts[name], encoding="utf-8")
    manifest = manifest_for(texts, parts, counter, params)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    m = manifest["halves"]
    for name in "AB":
        print(f"half {name}: {m[name]['tokens']} tokens (usage {m[name]['tokens_usage']}, "
              f"wikitext {m[name]['tokens_wikitext']}), {m[name]['chars']} chars", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
