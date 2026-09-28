# Paquet `server-tools` : serveur, tokeniseur et outils

Branche `perf/server-tools`, partie de `perf/base` (`b858f9b`, moteur 0.1.20). Issues traitées : #31, #32, #33, #34,
#35 (serveur, section O9 de l'audit), #49 (E6, `make_profile.py`), #50 (S1, `--spec` dans `calibrate.py`), #52 (S4,
quantification du brouillon MTP).

Aucun fichier du moteur C++ ni aucun kernel n'est modifié. Le seul ajout côté GPU est un test de parité,
`s2_grouped_parity` (voir #52), enregistré dans ctest.

Étiquettes des gains, comme dans l'audit : **MESURÉ ici** = mesuré dans ce dépôt sur le CPU de la machine de
développement (4 cœurs AVX-512, sans GPU) ; **ESTIMÉ** = calcul de l'audit ; **HYPOTHÈSE** = à confirmer sur la
RTX 5090. Aucun chiffre de ce document n'a été mesuré sur GPU.

## Résumé

| Issue | Changement | Par défaut | Retour à l'ancien comportement / activation | Gain |
|---|---|---|---|---|
| #31 O9 | Détokeniseur incrémental (décodeur UTF-8 incrémental) | **actif** (identique delta par delta) | `STRATA_OLD_DETOK=1` | 4,9 ms → ~1 µs par token à 16K (MESURÉ ici) |
| #32 O9b | Cache des pièces BPE + encodage incrémental des prompts | **actif** (ids identiques par construction) | `STRATA_OLD_PROMPT_ENCODE=1` ; contrôle : `STRATA_CHECK_PROMPT_IDS=1` | 482 ms → 3,7 ms au tour suivant, à 101K tokens (MESURÉ ici) |
| #33 O9c | Avertissement quand le niveau de réflexion change en cours de conversation | **actif** (message seulement, prompt inchangé) | — | aucun (documentation) |
| #34 O9d | `chat.py` renvoie `reasoning_content` | **actif** dans `chat.py` | `python chat.py --drop-thinking` | 0,2 à 2 s par tour (ESTIMÉ) |
| #34 O9d | Le serveur réinjecte la réflexion omise par un client | **désactivé** | `--recall-reasoning`, `"recall_reasoning": true` dans la config, ou `STRATA_RECALL_REASONING=1` | 0,2 à 2 s par tour (ESTIMÉ) |
| #35 O9e | Cache d'images indexé par les octets bruts ; lien dur pour une image seule | **actif** (mêmes embeddings) | — | 50 à 500 ms par image WebP/TIFF et par tour (ESTIMÉ) |
| #49 E6 | `make_profile.py` mélange traces et classement de base | **actif** dans l'outil (`--trace-weight 0.5`) | `--trace-weight 0` (sortie identique à l'octet près à l'ancienne) | meilleur taux de hits initial (HYPOTHÈSE) |
| #50 S1 | `calibrate.py` mesure `--spec` sur demande | **désactivé** | `--spec 4,5,6` ou `STRATA_CALIBRATE_SPEC=4,5,6` | à mesurer |
| #52 S4 | `mtp_pack.py` : recherche d'échelle élargie ou exacte, échelles négatives | **désactivé** (`grid`, octets identiques) | `--q2-search wide` ou `--q2-search exact` | erreur RMS 0,411 → 0,351 (MESURÉ ici) ; acceptation : HYPOTHÈSE |

Deux outils de mesure sont ajoutés pour les A/B : `tools/bench_turns.py` (conversation multi-tours contre un serveur)
et `tools/ab_oneshot.py` (exécutions ponctuelles du moteur, entrelacées).

## Détail par issue

### #31 (O9) : détokenisation incrémentale

**Avant.** `Detokenizer.push` ajoutait le token à la liste et **re-décodait toute la réponse** à chaque token
(`self.tok.decode(self.ids)`), soit un coût quadratique.

**Après.** Chaque token passe ses octets (`Tokenizer.token_bytes`, mis en cache) dans
`codecs.getincrementaldecoder("utf-8")("replace")`. La règle de rétention est celle de l'ancien code, à
l'identique : on retient la sortie tant qu'elle se termine par un caractère incomplet (octets gardés par le décodeur,
que l'ancien décodage complet affichait comme un U+FFFD) ou par un vrai U+FFFD. Les deltas émis sont donc les mêmes,
un pour un. Les tests le vérifient sur 400 flux aléatoires d'octets (octets invalides, caractères coupés, U+FFFD
réels, tokens spéciaux) et sur le vocabulaire qwen35. Un mutant qui ne retient pas les U+FFFD de fin est détecté
(334 flux sur 400 diffèrent).

- **Défaut :** actif. **Ancien chemin :** `STRATA_OLD_DETOK=1` (ou un tokeniseur sans `token_bytes`).
- **MESURÉ ici** (vocabulaire qwen35 de llama.cpp, réponse de 16 384 tokens) : 43,9 s au total et 4,9 ms par token
  entre 15K et 16K avant ; 0,02 s au total et ~1 µs par token après.
- **Gain sur la 5090 (HYPOTHÈSE) :** un cœur libéré pendant les longues réponses, soit quelques % de décodage quand le
  CPU limite. Rien ne change pour les réponses courtes.

### #32 (O9b) : tokenisation incrémentale des prompts

1. **Cache des pièces.** `Tokenizer._piece_ids` : `functools.lru_cache` (131 072 entrées) sur les ids d'une pièce du
   pré-tokeniseur de 64 caractères au plus. Il couvre le BPE et la conversion en octets. Les pièces plus longues
   (longues suites d'espaces, base64) sont recalculées. La valeur en cache est un tuple, donc non modifiable.
2. **`PromptEncoder`** (`tools/strata_tokenizer.py`). Il garde les 4 derniers prompts avec leurs ids et leurs **points
   de reprise**, c'est-à-dire la fin de chaque token spécial et le nombre d'ids à cet endroit (`encode_marked`). Pour un
   nouveau prompt, il calcule le préfixe commun L avec chaque prompt gardé. Il prend le dernier point de reprise c tel
   que `c + max_special_len - 1 <= L`, et n'encode que `texte[c:]`.

   **Pourquoi c'est exact.** L'encodage coupe le texte aux tokens spéciaux et encode chaque tronçon séparément. La
   décision « un token spécial commence en q » ne lit que `texte[q : q + max_special_len]`. Avec cette marge, toutes les
   décisions prises avant c sont donc identiques pour les deux textes. Sans la marge, un littéral plus long qui
   commence avant la coupure et la dépasse casserait l'égalité. Les tests construisent exactement ce cas : un
   encodeur sans marge donne 308 résultats faux sur 1 800 ; celui livré n'en donne aucun.
- **Défaut :** actif (ids identiques par construction et par test). **Ancien chemin :** `STRATA_OLD_PROMPT_ENCODE=1`.
  **Contrôle en service :** `STRATA_CHECK_PROMPT_IDS=1` encode aussi chaque prompt en entier et compare. En cas
  d'écart, le serveur affiche `PROMPT IDS DIFFER` et utilise l'encodage complet.
- **MESURÉ ici** (conversation de 62 messages, 365 476 caractères, 101 332 ids, vocabulaire qwen35 de llama.cpp) :

  | Encodage du prompt complet | Temps |
  |---|---:|
  | ancien code, sans cache | 482 ms (4,8 µs/token) |
  | nouveau, cache des pièces froid | 154 ms |
  | nouveau, cache chaud | 98 ms |
  | **nouveau, incrémental, tour suivant** (353 397 caractères repris) | **3,7 ms** |

  Les ids sont identiques dans les quatre cas.
- **Gain sur la 5090 (ESTIMÉ) :** 0,5 à 2 s de moins avant le premier token à 100K, à chaque tour. Le CPU de la
  machine de l'utilisateur est différent ; l'audit estimait 8 à 15 µs par token.

### #33 (O9c) : niveau de réflexion et relecture

Le gabarit écrit le niveau de réflexion dans le bloc système, en position 0. Changer de niveau au milieu d'une
conversation invalide donc tous les points de reprise, et tout est relu.

- **Fait :**
  - le serveur affiche `the thinking level changed (medium -> low) in the middle of a conversation ...` quand une
    requête prolonge la conversation précédente avec un autre niveau ;
  - `chat.py` le signale à `/think` si une conversation est en cours ;
  - l'application web ajoute une ligne sous le réglage « Thinking » et un avertissement quand on applique un nouveau
    niveau à une conversation en cours.
- **Non fait, volontairement :** déplacer l'indication plus loin dans le prompt. Cela change le conditionnement du
  modèle. C'est l'HYPOTHÈSE à évaluer de l'issue, avec une mesure de qualité, pas seulement de vitesse.
- **Gain :** aucun par défaut. L'utilisateur sait pourquoi le tour est lent.

### #34 (O9d) : la réflexion des tours passés

Le gabarit rend chaque réponse passée sous la forme `<think>\nRÉFLEXION\n</think>\n\nRÉPONSE`. Un client qui ne
renvoie que la réponse produit `<think>\n\n</think>` : le prompt quitte la séquence vivante du moteur juste après
`<think>`, et la réponse précédente est relue à chaque tour.

1. **`chat.py`** renvoie `reasoning_content` (actif par défaut ; `--drop-thinking` pour l'ancien comportement).
2. **Serveur, opt-in** (`--recall-reasoning`, `"recall_reasoning": true`, `STRATA_RECALL_REASONING=1`) :
   `ReasoningRecall` garde la réflexion écrite par le modèle pour chaque réponse. La clé est le hash SHA-256 de la
   conversation jusqu'à cette réponse incluse (rôles, contenus sans les espaces de bord, appels d'outils ; jamais la
   réflexion). Une réponse identique dans une autre conversation ne reçoit donc rien, et un tour antérieur modifié ne
   trouve plus rien.
   La table est bornée à 1 024 réponses et 16 millions de caractères (LRU). Une réflexion envoyée par le client est
   toujours prioritaire.
   - **Pourquoi opt-in :** cela change le prompt des clients qui omettent la réflexion. Leurs tours passés portent
     désormais leur réflexion, comme le gabarit le prévoit, et le contexte se remplit plus vite.
- **Tests :** avec la table, le prompt du tour 2 commence exactement par le prompt du tour 1 suivi des tokens
  produits (réponse simple et appel d'outil). Sans la table (défaut), il diverge. `chat.py` seul, contre un serveur par
  défaut, garde aussi le préfixe vivant.
- **Gain (ESTIMÉ, audit) :** 0,2 à 2 s par tour.

### #35 (O9e) : images

- La clé du cache d'embeddings est le hash des **octets bruts**. La normalisation par Pillow (WebP, TIFF, AVIF → PNG)
  ne tourne plus qu'en cas d'absence du cache. Les mêmes octets donnent toujours le même PNG, donc les embeddings sont
  inchangés. Un accès au cache remet l'entrée en tête (LRU au lieu de FIFO).
- Une requête avec **une seule image** crée un lien dur (`os.link`) vers le fichier d'embeddings en cache, au lieu d'en
  copier ~10 Mo (copie de repli si le système de fichiers refuse). Le fichier de la requête est créé pendant que la
  FIFO est tenue, pour qu'aucun autre encodage ne puisse évincer les fichiers dont il est fait.
- **Défaut :** actif. **Gain (ESTIMÉ) :** 50 à 500 ms avant le premier token par image WebP/TIFF et par tour, plus
  quelques ms de copie évitées par requête à une image.

### #49 (E6) : `make_profile.py`

Le profil livré classe les 24 576 paires, et le classement de base passait en premier. Une trace ne changeait donc
rien : vérifié ici, la sortie de l'ancien outil avec une trace est identique à l'octet près au profil livré.

- **Nouveau classement :** score = `w × fréquence dans la trace + (1 − w) × fréquence de la trace au rang de base de
  la paire`. Les deux termes sont ainsi sur la même échelle. Les égalités sont départagées par le rang de base.
  - `--trace-weight` vaut 0,5 par défaut ;
  - avec 1 : la trace d'abord, puis le rang de base ;
  - avec 0 : l'ancien ordre.
- Sans trace, avec `--no-base` ou avec `--trace-weight 0`, la sortie est **identique à l'octet près** à l'ancienne
  (vérifié).
- **Gain (HYPOTHÈSE) :** meilleur taux de hits initial pour un usage ciblé (code, appels d'outils), à mesurer.

### #50 (S1) : `--spec` dans la calibration

`tools/calibrate.py` mesure maintenant, sur demande, la fenêtre de vérification :
- activation : `--spec 4,5,6`, ou `STRATA_CALIBRATE_SPEC=4,5,6` pour `setup --calibrate` ;
- un redémarrage du moteur par valeur, avec la part PCIe et le plancher de brouillon déjà choisis ;
- un tour de chauffe, puis 3 exécutions des 3 prompts ;
- une fenêtre n'est gardée que si sa médiane bat `--spec 4` de plus de 3 % (même règle `MIN_GAIN`) ;
- le nombre de workers est ensuite mesuré avec la fenêtre choisie.

`apply()` remet `--spec` à 4 (la valeur de `setup.py`) quand la calibration ne l'a pas choisi, pour qu'une ancienne
valeur ne reste pas. Il n'ajoute jamais `--spec` à une config qui ne spécule pas. Désactivé par défaut, parce que
chaque valeur coûte un redémarrage. **Gain :** à mesurer sur la 5090.

### #52 (S4) : quantification du brouillon MTP

- **Prérequis vérifié :**
  - les 512 experts du brouillon tournent tous sur `moe_grouped_s2` (`core/mtp.cpp`) ;
  - ce kernel multiplie l'échelle fp16 comme un flottant signé (`dw * dx * (s - hx)`), donc une échelle négative est
    prise en charge par construction ;
  - la référence ggml (`((q) - 1) * d`) et le noyau CPU ont la même propriété ;
  - `mtp_rt.py` recopie les blocs sans les interpréter.
- **Nouveau test GPU `s2_grouped_parity`** (ctest) :
  - `moe_grouped_s2` contre une référence en double sur 16 experts synthétiques, dont environ 63 % des échelles sont
    négatives, avec 8 tokens en top-10 ;
  - la même chose avec toutes les échelles positives, dont la sortie doit différer ;
  - vérifié ici contre une émulation CPU des kernels (marge : 0,016 de la tolérance) et contre une émulation qui perd le
    signe (échec, comme attendu) ;
  - à exécuter sur la 5090.
- **`mtp_pack.py --q2-search`** :
  - `grid` (défaut, octets identiques à l'ancien code, testé) ;
  - `wide` : 0,15 à 1,0 × amax en 35 pas, dans les deux signes ;
  - `exact` : l'optimum des moindres carrés sur d de signe quelconque. Il y a au plus 128 changements d'arrondi par
    bloc, parcourus dans l'ordre. L'optimalité est testée contre un balayage dense.
- **MESURÉ ici** (erreur RMS relative, blocs synthétiques de 64 poids) :

  | Loi | grid | wide | exact |
  |---|---:|---:|---:|
  | normale | 0,411 | 0,351 | 0,351 |
  | Laplace | 0,485 | 0,399 | 0,399 |
  | Student t3 | 0,517 | 0,430 | 0,429 |

  Temps CPU par expert : 0,8 / 1,4 / 1,8 s, soit ~7 / 12 / 15 min pour les 512 experts.
- **Défaut :** `grid`. Le taux d'acceptation du brouillon est une HYPOTHÈSE à mesurer.
- **Non fait :** les brouillons Q4/Q8. `mtp_rt.py` et le moteur (`cpu::BLOB`, `moe_grouped_s2`) ne connaissent que la
  disposition Q2_0 ; il faut d'abord porter le moteur.

## Vérifié ici (sans GPU)

- Build complet (`cmake --build build-wp -j 3`) : vert. `s2_grouped_parity` compile.
- ctest CPU : `ple_reader_selftest`, `platform_memory_test`, `pool_stress`, `expert_multi_test`,
  `suffix_drafter_test`, `draft_policy_test`, `controller_test`, `conv_cache_test` : tous passent.
- Python :

  | Suite | Résultat |
  |---|---|
  | `serve/test_server.py` | 41 tests OK |
  | `serve/test_mcp.py` | 24 OK |
  | `tools.test_calibrate` | 17 OK |
  | `tools.test_iq_pack` | 8 OK |
  | `tools.test_strata_tokenizer` | 12 OK, dont les 50 vecteurs qwen35 de llama.cpp |
  | `tools.test_make_profile` | 7 OK |
  | `tools.test_mtp_pack` | 6 OK |
  | `tools.test_ab_oneshot` | 2 OK |

## À exécuter sur la RTX 5090

Depuis la racine du dépôt, sous Linux (sous Windows, mêmes commandes avec `set VAR=...` et `python`).

```bash
git fetch origin && git checkout perf/server-tools
export PATH=/usr/local/cuda/bin:$PATH
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON \
      -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_GGML_DIR=<le llama.cpp de votre build habituel>
cmake --build build -j

# 1. Tests
ctest --test-dir build -R s2_grouped_parity --output-on-failure        # #52 : échelles négatives sur le GPU
python3 serve/test_server.py && python3 serve/test_mcp.py
python3 -m unittest tools.test_calibrate tools.test_make_profile tools.test_ab_oneshot
STRATA_GGUF_PY=third_party/llama.cpp/gguf-py python3 -m unittest tools.test_iq_pack tools.test_mtp_pack
# le tokeniseur du pack, plutôt que celui de llama.cpp (#32) :
STRATA_TOKENIZER="$(python3 -c "import json;print(json.load(open('strata-q2_0.json',encoding='utf-8-sig'))['tokenizer'])")" \
  python3 -m unittest tools.test_strata_tokenizer -v
```

### 2. A/B du serveur (#31, #32, #34)

Mesures faites avec `bench_turns.py` (réglages par défaut : glouton, 256 tokens, 6 tours) :
- document d'environ 100K tokens au premier tour ;
- 3 exécutions par configuration ;
- chaque exécution est une nouvelle conversation (`--salt` par défaut) ;
- colonnes : temps avant le premier texte, `prompt ms` et `reused` du serveur.

Lancer le serveur dans un terminal et le client dans un autre ; arrêter le serveur entre deux configurations.

```bash
# A : anciens chemins
STRATA_OLD_DETOK=1 STRATA_OLD_PROMPT_ENCODE=1 python3 -m serve.server --engine strata --config strata-q2_0.json --port 8080
for i in 1 2 3; do python3 tools/bench_turns.py --port 8080 --context-chars 400000 --json A$i.json; done

# B : défaut de la branche, avec contrôle des ids (aucune ligne "PROMPT IDS DIFFER" ne doit apparaître)
STRATA_CHECK_PROMPT_IDS=1 python3 -m serve.server --engine strata --config strata-q2_0.json --port 8080
for i in 1 2 3; do python3 tools/bench_turns.py --port 8080 --context-chars 400000 --json B$i.json; done
#   puis sans STRATA_CHECK_PROMPT_IDS (le contrôle refait un encodage complet et fausse le temps)

# C : + réinjection de la réflexion (#34)
python3 -m serve.server --engine strata --config strata-q2_0.json --port 8080 --recall-reasoning
for i in 1 2 3; do python3 tools/bench_turns.py --port 8080 --context-chars 400000 --json C$i.json; done

# D : client qui renvoie sa réflexion (comme chat.py maintenant), serveur par défaut
for i in 1 2 3; do python3 tools/bench_turns.py --port 8080 --context-chars 400000 --send-reasoning --json D$i.json; done
```

Attendu :
- B contre A : même `prompt tokens`, `first text s` plus court à chaque tour (le gain de #32), et moins de CPU pendant
  les longues réponses (#31) ;
- C et D contre B : `reused` proche de `prompt tokens` dès le tour 2, et `prompt ms` bien plus court.

Pour #35, envoyer la même image WebP deux tours de suite dans l'application web : le second tour ne doit plus
passer par Pillow ni par l'encodeur d'image.

### 3. A/B du moteur (#50, #52)

`ab_oneshot.py` fait tourner le moteur en exécutions ponctuelles :
- glouton, 256 tokens, `--adapt-every 100000` ;
- les 3 prompts de calibration, 3 exécutions, variantes entrelacées ;
- il affiche la médiane des tok/s, les tokens par tour, l'acceptation des brouillons, et vérifie que la sortie
  gloutonne est identique token pour token.

```bash
# #50 : fenêtre spéculative
python3 tools/ab_oneshot.py strata-q2_0.json --runs 3 --variant spec4="--spec 4" --variant spec5="--spec 5" \
        --variant spec6="--spec 6" --json spec.json
#   ou la calibration complète, qui garde la fenêtre seulement au-delà de +3 % :
python3 tools/calibrate.py strata-q2_0.json --spec 4,5,6

# #52 : brouillon MTP avec échelles négatives (le dossier mtp est le parent du --mtp de la config)
MTP="$(python3 -c "import json;a=json.load(open('strata-q2_0.json',encoding='utf-8-sig'))['args'];print(a[a.index('--mtp')+1])")/.."
python3 tools/mtp_pack.py --src "$MTP" --experts q2_0 --q2-search exact --out "$MTP/mtp-q2_0-exact.gguf"
python3 tools/mtp_rt.py --gguf "$MTP/mtp-q2_0-exact.gguf" --out "$MTP/rt-exact"
cp data/draft_vocab.bin "$MTP/rt-exact/"
python3 tools/ab_oneshot.py strata-q2_0.json --runs 3 --variant grid= --variant exact="--mtp '$MTP/rt-exact'" \
        --json mtp.json
#   (idem avec --q2-search wide si exact gagne : les deux sont à 0,3 % l'un de l'autre en erreur RMS)
```

Attendu :
- `same greedy output` = `yes` pour toutes les variantes ;
- une variante n'est retenue que si elle gagne plus de 3 % de tok/s ;
- pour #52, regarder d'abord l'acceptation et les tokens par tour.

### 4. Profil d'experts (#49)

```bash
# une trace de routage typique de l'usage visé (prompt de code, d'appels d'outils, ...)
build/strata <arguments de la config> --tokens-file p.ids --max-new 256 --greedy --dump-routing trace.bin
python3 tools/make_profile.py trace.bin --out data/expert-profile-mine.bin                   # w = 0,5
python3 tools/make_profile.py trace.bin --trace-weight 1 --out data/expert-profile-mine1.bin
python3 tools/ab_oneshot.py strata-q2_0.json --runs 3 --variant base= \
        --variant mine="--expert-profile data/expert-profile-mine.bin" \
        --variant mine1="--expert-profile data/expert-profile-mine1.bin"
```

`--adapt-every 100000` fige le tier adaptatif : l'écart de tok/s vient donc du profil initial. En mode serveur, le
taux de hits est dans la colonne `hit_rate` de `/metrics` (onglet Monitor) et dans la ligne
`decode expert cache hit rate` du journal du moteur.

## Limites et reste à faire

- **#33 :** seules l'information et la documentation sont faites. Déplacer le niveau de réflexion hors du bloc système
  demande une évaluation de qualité (HYPOTHÈSE de l'issue) ; rien n'est changé dans le prompt.
- **#34 :**
  - la réinjection côté serveur est opt-in ;
  - l'application web ne renvoie toujours pas sa réflexion. Avec plusieurs tours d'outils MCP, elle concatène les
    réflexions et ne peut pas les redécouper par tour. `--recall-reasoning` couvre aussi ce cas.
- **#52 :** les brouillons Q4 et Q8 demandent un portage du moteur (`mtp_rt.py`, `cpu::BLOB`, `moe_grouped_s2`) ; non
  fait. `s2_grouped_parity` n'a tourné que contre une émulation CPU ici.
- **#50 et #52 :** aucune mesure sur GPU. Les réglages ne changent pas par défaut.
- **#31 et #32 :** mesures faites avec le vocabulaire qwen35 de llama.cpp (151 936 tokens), pas avec celui du pack
  (248 320). Le test `tools.test_strata_tokenizer` avec `STRATA_TOKENIZER` refait les vérifications d'identité avec le
  vrai tokeniseur.
