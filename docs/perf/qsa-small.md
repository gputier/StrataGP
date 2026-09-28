# qsa-small : moins de lancements dans la couche QSA, pas de table RoPE inutile (issues #24, #25, #47)

Branche `perf/qsa-small`, partie de `perf/base` (b858f9b, moteur 0.1.20). **Aucun chiffre GPU n'a été mesuré** :
la machine de développement n'a pas de GPU. Tout a été compilé (CUDA 13.0, sm_120) et les tests CPU ont été lancés
(ils passent, sauf `pool_test` qui exige un pack de modèle, comme sur `perf/base`) ; le nouveau test GPU
`qsa_prep_parity` compile mais reste à lancer sur la RTX 5090. **À faire avant de fusionner : `qsa_prep_parity`
doit afficher `OK`, puis les tokens de l'A/B doivent être identiques** (voir « Validation »).

## Ce qui change

| # | Issue | Changement | Par défaut | Retour à l'ancien chemin |
|---|---|---|---|---|
| 1 | #24 (O6d) | Le pas (`step`) et les positions RoPE (`pos_dev`) sont téléversés **une fois par token**, par la première couche QSA, en **un** kernel (`copy_i32x2_from_mapped`) ; les 11 autres couches lisent ses tampons. Avant : 2 kernels `copy_i32_from_mapped` par couche QSA, 24 par token. | **activé** | `STRATA_OLD_QSA_STEP=1` (un tampon et deux copies par couche, comme avant) |
| 2 | #25 (O6e) | Côté K : norme + RoPE (+ les deux FWHT en Q4) + ajout au cache KV en **un** lancement (`kv_append{,_q8,_q4}_prep_step`). Côté requêtes : q lu directement dans `q_full` (plus de `cudaMemcpy2DAsync`), norme + RoPE (+ FWHT en Q4) de q **et** norme + RoPE de la requête de l'indexeur en **un** lancement (`qsa_q_prep`). **Identique au bit près** (voir 2). | **activé** | `STRATA_OLD_QSA_PREP=1` (les lancements séparés) |
| 3 | #47 (E4) | La table RoPE FP64→FP32 (`max_cells × 32 × 2` floats) n'est plus allouée ni construite quand rien ne la lit : `--native-rope` **et** `--native-qsa-indexer` actifs, ce que fait `--native`. | **activé** | `STRATA_OLD_ROPE_TABLE=1` (la table est toujours allouée) |

Les trois sont identiques au bit près par construction (1 et 3 ne touchent aucune arithmétique ; 2 refait exactement
les mêmes opérations flottantes, voir plus bas), d'où l'activation par défaut, avec l'ancien chemin sélectionnable pour
l'A/B.

**Portée : où ces gains se voient.** `qsa_layer` (fichier `src/core/layer.cpp`) est la couche QSA du **décodage token
par token** : `session_run_token` / `session_loop` / `session_token`. Elle tourne :
- pour tout le décodage **sans** `--spec` avec un pack canonique (Q2_0 réempaqueté AVX-512) ;
- pour le dernier token du prompt avec un pack canonique, même avec `--spec`.

Avec `--spec 4 --mtp …` (la configuration écrite par `setup.py`), le décodage passe par les **fenêtres de
vérification** (`src/core/verify.cpp`), que cette branche ne modifie pas ; un pack natif (IQ) ne décode que par
fenêtres. Les points 1 et 2 n'accélèrent donc que le décodage non spéculatif. Le point 3 s'applique à toute
configuration `--native` (VRAM et démarrage). Les kernels du point 2 acceptent `n_tok` cellules (pas et positions
propres à chaque token) précisément pour que `verify.cpp` puisse les adopter : voir « Suite possible ».

### 1. Un seul téléversement du pas par token (`layer.cpp`, `session.cpp`, `elementwise.cu`)

- Les 12 couches QSA dérivent `step` (`qsa_step_fill(pos)`) et `pos_dev` (`pos_base + pos` × `n_head`) de la même
  position : les valeurs sont identiques d'une couche à l'autre.
- `session_init` fait pointer `step`/`pos_dev` des états 1 à 11 vers ceux de l'état 0 (`qsa_state_share_step`,
  drapeau `QsaState::step_shared`). Dans `qsa_layer`, un état partagé ne téléverse rien ; l'état 0 copie les deux
  plages en un kernel (`copy_i32x2_from_mapped`, depuis la mémoire hôte mappée comme avant).
- **Invariant** : la couche propriétaire (la première couche QSA, couche 3) tourne avant les autres dans chaque
  token. C'est le cas de tous les chemins qui appellent `qsa_layer` : graphe « token », graphes par couche
  (`session_loop`, `session_replay*`, y compris `--gpu-stages` et ses préfixes, qui rejouent les couches dans
  l'ordre) et `session_token` (`--no-capture`). Le drafter MTP et les fenêtres de vérification ont leurs propres
  tampons de pas et ne sont pas concernés. Les tampons hôtes épinglés (`host_step`, `host_pos`) restent remplis
  pour les 12 états, sur tous les chemins : par `stage_token` pour les graphes, et par `qsa_layer` lui-même sur le
  chemin direct, où un état partagé ne saute que le téléversement (le dernier élément de `host_step` porte toujours
  le statut d'attention par couche).
- Kernels par token : 24 → 1.

### 2. Préparation de K et des requêtes en deux lancements (`qsa_prep.hpp`, `qsa_prep.cu`, `kv_q8.cu`, `kv_q4.cu`)

Nouvelles fonctions (déclarées dans `include/strata/kernels/qsa_prep.hpp`) :
- `kv_append_prep_step` (FP16), `kv_append_q8_prep_step` (INT8), `kv_append_q4_prep_step` (Q4_0 + Hadamard) : grille
  `(n_head_kv, K|V, n_tok)`, 256 threads par ligne. Bloc K : réduction RMS, poids, rotation NEOX en mémoire partagée,
  (Q4) FWHT par le warp 0, puis quantification et écriture (VRAM si la page est résidente, copie hôte si streaming).
  Bloc V : (Q4) FWHT, puis quantification. `kcur` (et `vcur` en Q4) sont réécrits avec les lignes préparées, comme
  les lancements séparés les laissaient (les vidages `--dump-*` restent identiques).
- `qsa_q_prep` : grille `(n_head + idx_n_head, n_tok)`. Les 24 lignes de q sont lues **directement** dans la première
  moitié de chaque bloc `2 × head_dim` de `q_full` et écrites normalisées/tournées (et en Q4 transformées) dans
  `qcur` ; les 4 lignes de la requête de l'indexeur sont traitées en place. Pour n'avoir qu'un lancement, la
  projection BF16 de la requête de l'indexeur est faite **avant** (elle ne dépend que de `x` : ordre sans effet).
  La FWHT de q, qui était faite après la sélection, est faite ici (q ne sert qu'à l'attention) ; l'étape 8 la saute.
- Les lignes de la requête de l'indexeur lisent leur position dans les mêmes `n_head` entrées par token que q :
  `qsa_prep_supported` exige donc `idx_n_head <= n_head` (4 ≤ 24 ici) ; au-delà, l'appelant garde les lancements séparés.
- Lancements par couche QSA : FP16/INT8 **8 → 2**, Q4_0 **11 → 2** (dont un nœud de copie `memcpy2D` en moins). Par
  token : −72 (FP16/INT8) ou −108 (Q4_0) nœuds, plus les −23 du point 1.
- Registres (ptxas, sm_120) : 28 à 36 selon le format et la variante, **0 spill**.
- Repli automatique sur les lancements séparés si `qsa_prep_supported` refuse la géométrie (head_dim ≠ 256, etc.).

**Pourquoi c'est identique au bit près.** Chaque opération flottante est celle des kernels séparés, sur les mêmes
valeurs, dans le même ordre, fusionnée (FMA) ou non exactement comme ptxas la fusionne dans ces kernels. Vérifié en
lisant le SASS (`cuobjdump -sass`, CUDA 13.0, sm_120) des anciens et des nouveaux kernels :
- norme canonique (`rms_norm_weighted`, fichier précis) : chaîne FFMA par voie avec pas de 32, arbre `shfl_down`,
  division IEEE et `rsqrtf` (avec la mise à l'échelle des dénormaux) par la voie 0, puis `(x × w) × inv` ;
- norme native (`native_qsa_rms_norm_weighted`, fichier compilé en `--use_fast_math`) : carré en FFMA.FTZ sur +0,
  deux papillons XOR sur 256 threads, puis `FFMA.FTZ(somme, RCP(n), eps)` — c'est ainsi que ptxas abaisse la division
  approchée suivie de l'ajout d'epsilon — et `MUFU.RSQ`, puis `(scale × x) × gamma` en FMUL.FTZ ;
- RoPE sur table (`rope_neox_apply`) : `oa = fma(a, c, −(b·s))`, `ob = fma(b, c, a·s)` ;
- RoPE native (`native_rope_apply`) : `theta = pos × ex2(lg2(theta_scale) × paire)`, `sin`/`cos` approchés
  (`FMUL.RZ` par 1/2π puis `MUFU.SIN/COS`), `oa = fma(c, a, −(s·b))`, `ob = fma(s, a, c·b)`, tout en FTZ ;
  `theta_scale` vient de la même fonction hôte (`native_rope_theta_scale`, extraite de `native_rope_apply`) ;
- FWHT : le même papillon que `fwht256_kernel` (FMUL par 1/16, FSEL + FADD) ;
- quantificateurs : ceux de `kv_append_q8_kernel` (même code), `q4_group`/`q4_store` (la même fonction), et
  `f16_from_f32` pour le FP16.

L'arithmétique « fast-math » est écrite dans le fichier précis (`qsa_prep.cu`, etc.) sous forme du PTX que ces
fichiers produisent (`fma.rn.ftz`, `add.rn.ftz`, `mul.rn.ftz`, `rcp/rsqrt/lg2/ex2/sin/cos.approx.ftz`) ; le suffixe
`.rn` empêche ptxas de fusionner quoi que ce soit d'autre. **Limite honnête** : l'égalité dépend donc aussi de la
façon dont ptxas compile les *anciens* kernels (ses choix de fusion). C'est vérifié pour CUDA 13.0 / sm_120 (la chaîne
de `setup.py`) ; après tout changement de toolkit, relancer `qsa_prep_parity`, qui est l'arbitre.

`qsa_prep_parity` compare octet par octet, pour les 3 formats × 4 variantes (norme canonique/native × RoPE
table/native) × KV résident/streamé (pages non résidentes comprises) × positions texte/image (table M-RoPE) × 1 ou 3
tokens × 4 tirages = **384 configurations** : les pools VRAM, les copies hôtes, `kcur`, `vcur`, `qcur` et `q_idx`.
Les lignes incluent des cas adverses : carrés sous `FLT_MIN` (la norme native les met à zéro, pas la canonique),
entrées dénormales, lignes nulles, un seul non-zéro, sommes qui débordent, magnitudes mêlées, poids nuls ou minuscules.
Il vérifie aussi `copy_i32x2_from_mapped` (point 1).

### 3. Pas de table RoPE en mode natif (`layer.cpp`, `session.cpp`)

- Lecteurs de la table : la RoPE sur table (`--native-rope` inactif : `qsa_layer`, `verify.cpp`, `mtp.cpp`) et
  l'indexeur canonique (`indexer_key_append`, `--native-qsa-indexer` inactif). Le prefill a sa propre rotation et
  les fenêtres de vérification utilisent toujours l'indexeur natif.
- `qsa_rope_table_needed()` = `STRATA_OLD_ROPE_TABLE` ou `!native_rope` ou `!native_qsa_indexer`. `session_bytes` et
  `session_init` dimensionnent et initialisent sans table quand c'est faux ; `qsa_state_init(…, with_rope)` ne la
  découpe ni ne la construit (plus de calcul FP64 sur l'hôte au démarrage). Les commutateurs sont posés dans
  `generate.cpp` avant `session_bytes`.
- Un lecteur qui trouverait la table absente échoue avec un message (`no RoPE table (E4 …)`) au lieu de lire un
  pointeur nul.
- VRAM rendue : `max_cells × 256 o`, soit 8 Mio à 32K, 32 Mio à 128K, **64 Mio à 262K** ; le cache d'experts
  (`--expert-cache auto`, dimensionné après la session sur la VRAM libre) la récupère.

## Gains attendus

- **(1) ESTIMÉ** : 23 nœuds de graphe de moins par token, à ~1–2 µs pièce : **25 à 50 µs par token** (audit : 30 à
  60 µs).
- **(2) ESTIMÉ** : 6 nœuds de moins par couche QSA (9 en Q4_0), 72 (108) par token, dont 12 nœuds de copie `memcpy2D` :
  **70 à 150 µs par token** (audit), un peu plus en Q4_0. **HYPOTHÈSE** : davantage sous Windows, où un nœud de copie
  coupe la soumission WDDM.
- Ensemble (1)+(2) : **~0,1 à 0,2 ms par token** en décodage non spéculatif (HYPOTHÈSE, à mesurer : `qsa_prep_parity
  --bench` et `--gpu-stages` ci-dessous). Rien avec `--spec`, sauf le dernier token du prompt (voir « Portée »).
- **(3)** : 64 Mio de VRAM à 262K (~46 experts de plus selon l'audit ; 23 à 128K) ; **HYPOTHÈSE** : 0,3 à 0,6 s de
  moins au démarrage à 262K (table FP64 non construite).

## Validation sur la RTX 5090

```bash
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON \
      -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_GGML_DIR=/opt/llama.cpp
cmake --build build -j

# 1. la préparation fusionnée contre les lancements séparés, octet par octet (GPU, sans modèle)
./build/qsa_prep_parity           # attendu : "qsa_prep_parity: OK (384 configurations compared byte for byte)"
./build/qsa_prep_parity --bench   # MESURE : µs par couche QSA, séparé / fusionné, par format ; et le pas par token

# 2. les tests voisins, inchangés, doivent toujours passer
(cd build && ctest --output-on-failure -R "qsa_prep_parity|kv_q8_parity|kv_q4_parity|rope_parity|qsa_parity|kv_stream_parity|elementwise_parity")
```

Si `qsa_prep_parity` échoue, le message donne la configuration et le premier octet différent : garder
`STRATA_OLD_QSA_PREP=1` et me le transmettre ; ne pas fusionner le point 2 en l'état.

**A/B de bout en bout.** Il faut une configuration qui passe par `qsa_layer` : un **pack canonique Q2_0**, **sans**
décodage spéculatif. `ARGS` = les arguments écrits par `setup.py` **sans** `--spec 4 --spec-min-p 0.5 --mtp …`
(`--pack … --native … --ple-gguf … --expert-profile … --expert-cache auto --prefill auto --max-context …`).
Trois prompts tokenisés `p1.txt p2.txt p3.txt` (`--tokens-file`).

```bash
OLD="STRATA_OLD_QSA_STEP=1 STRATA_OLD_QSA_PREP=1 STRATA_OLD_ROPE_TABLE=1"
RUN="./build/strata $ARGS --max-new 256 --greedy --adapt-every 100000"

# 1. nouveau / ancien, 3 prompts x 3 passages ; les tokens doivent être identiques
for p in p1 p2 p3; do
  for run in 1 2 3; do
    $RUN --tokens-file $p.txt > new_${p}_$run.log 2> new_${p}_$run.err
    env $OLD $RUN --tokens-file $p.txt > old_${p}_$run.log 2> old_${p}_$run.err
  done
  diff <(grep '^output' new_${p}_1.log) <(grep '^output' old_${p}_1.log) && echo "$p: tokens identiques"
done
grep -H '^decode' new_*.log old_*.log          # tok/s : comparer les moyennes par prompt

# 2. chaque point seul (tokens identiques attendus aussi)
for v in STRATA_OLD_QSA_STEP STRATA_OLD_QSA_PREP; do
  env $v=1 $RUN --tokens-file p1.txt > only_$v.log 2> only_$v.err
  diff <(grep '^output' only_$v.log) <(grep '^output' new_p1_1.log) && echo "$v: tokens identiques"
done

# 2b. le partage du pas sur les trois chemins qui appellent qsa_layer : graphe « token » (défaut), graphes par
#     couche (--no-token-graph) et appels directs (--no-capture) ; les 11 autres couches QSA doivent lire le pas téléversé
#     par la première couche QSA, donc des tokens identiques à STRATA_OLD_QSA_STEP=1 sur chaque chemin
for path in "" "--no-token-graph" "--no-capture"; do
  tag=${path:-token}
  $RUN $path --tokens-file p1.txt --max-new 64 > step_new_$tag.log 2>/dev/null
  STRATA_OLD_QSA_STEP=1 $RUN $path --tokens-file p1.txt --max-new 64 > step_old_$tag.log 2>/dev/null
  diff <(grep '^output' step_new_$tag.log) <(grep '^output' step_old_$tag.log) && echo "$tag: tokens identiques"
done

# 3. le temps GPU pur de la couche QSA (graphes rejoués sans l'hôte) : ligne "QSA layers ... ms/token"
$RUN --tokens-file p1.txt --max-new 1 --gpu-stages 2>&1 | grep -E "QSA layers|GDN layers"
env $OLD $RUN --tokens-file p1.txt --max-new 1 --gpu-stages 2>&1 | grep -E "QSA layers|GDN layers"
$RUN --tokens-file p1.txt --max-new 1 --gpu-only-full | grep "GPU floor"
env $OLD $RUN --tokens-file p1.txt --max-new 1 --gpu-only-full | grep "GPU floor"

# 4. les formats KV Q4_0 et INT8 (le chemin Q4_0 fusionne aussi les trois FWHT)
for kv in int8 q4_0; do
  $RUN --kv $kv --tokens-file p1.txt > kv_new_$kv.log 2>/dev/null
  env $OLD $RUN --kv $kv --tokens-file p1.txt > kv_old_$kv.log 2>/dev/null
  diff <(grep '^output' kv_new_$kv.log) <(grep '^output' kv_old_$kv.log) && echo "--kv $kv: tokens identiques"
done
```

**E4 (VRAM)** : avec la configuration complète de `setup.py` (`SETUP_ARGS`, `--spec` et `--max-context` compris ;
l'effet est proportionnel au contexte : 64 Mio à 262K, 8 Mio à 32K), comparer le nombre d'emplacements d'experts et
vérifier que les tokens ne changent pas :

```bash
FULL="./build/strata $SETUP_ARGS --max-new 256 --greedy --adapt-every 100000"
$FULL --tokens-file p1.txt > e4_new.log 2> e4_new.err
STRATA_OLD_ROPE_TABLE=1 $FULL --tokens-file p1.txt > e4_old.log 2> e4_old.err
grep "expert cache auto" e4_new.err e4_old.err   # attendu à 262K : ~64 Mio libres de plus, ~46 emplacements
diff <(grep '^output' e4_new.log) <(grep '^output' e4_old.log) && echo "E4: tokens identiques"
```

Avec `--spec` (configuration par défaut), les points 1 et 2 ne doivent rien changer au débit du décodage, seulement au
dernier token du prompt d'un pack canonique : les tokens doivent rester identiques.

## Suite possible (hors de ce lot)

Les fenêtres de vérification (`verify.cpp`, le décodage réel avec `--spec`) refont les mêmes lancements **par token**
(norme + RoPE de K, FWHT, ajout, `memcpy2D` de q, norme + RoPE de q, FWHT, requête de l'indexeur). Les kernels de ce
lot prennent déjà `n_tok` cellules avec leurs propres pas (`step_ + t × kStepCount`) et positions
(`pos_ + t × NH`, `pos_stride = NH`) : les boucles par token de `verify.cpp` se remplacent par un appel
`kv_append*_prep_step(…, step_ + tb × kStepCount, kcur_ + tb × NKV × HD, vcur_ + …, wkn, pos_ + tb × NH, NH, n, …)`
et, après les projections BF16 des requêtes de l'indexeur, un appel `qsa_q_prep(qfull_ + …, qcur_ + …, wqn, qidx_ + …,
wiqn, pos_ + tb × NH, NH, n, st.kv_q4, …)`. Ce serait ~6 à 9 nœuds de moins par token de fenêtre et par couche QSA,
de l'ordre de 300 à 600 nœuds par fenêtre de 4 à 6 tokens (ESTIMÉ). Ce fichier est modifié par le lot O4 (fenêtres)
et n'a donc pas été touché ici ; `qsa_prep_parity` couvre déjà le cas `n_tok = 3`.
