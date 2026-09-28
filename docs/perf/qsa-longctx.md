# Paquet `qsa-longctx` : indexeur QSA, top-k de blocs et attention au long contexte

Branche `perf/qsa-longctx`, partie de `perf/base` (b858f9b, moteur 0.1.20). Issues traitées : **#21** (O6),
**#22** (O6b), **#23** (O6c) et **#12** (B14).

Rien n'a pu être exécuté sur GPU : la machine de développement n'en a pas. Tout compile (CUDA 13.0, sm_120), les tests
CPU passent, et chaque kernel modifié a été vérifié au niveau du PTX (voir « Ce qui a été vérifié ici »). **Aucun gain
n'est mesuré** : les chiffres ci-dessous sont des ESTIMATIONS ou des HYPOTHÈSES, à confirmer sur la RTX 5090 avec les
commandes de la dernière section.

## Résumé

| Issue | Changement | Par défaut | Sélecteur | Résultat |
|---|---|---|---|---|
| #12 (B14) | cellule dont la page vaut -1 masquée dans `qsa_decode_attn` ; refus d'un balayage CLOCK avec moins de 1 024 emplacements | **activé** | aucun (correctif) | identique quand toutes les pages sont résolues |
| #22 (O6b) | `block_topk2_kernel` : lectures coalescées, histogrammes par warp agrégés, balayages parallèles | **activé** | `STRATA_OLD_TOPK=1` rend l'ancien kernel | **même sélection, par construction** |
| #23 (O6c) | `attn_chunk_pf_kernel` : V en `cp.async` vers la mémoire partagée, lignes K préchargées | **activé** | `STRATA_OLD_QSA_ATTN=1` rend l'ancien kernel | **bit à bit, par construction** |
| #23 (O6c) | CHUNK = 32 (deux fois plus de blocs) | désactivé | `STRATA_QSA_CHUNK=32` | change l'arrondi (autre découpage du softmax) |
| #23 (O6c) | porte sigmoïde intégrée à `attn_merge` (sauf `--kv q4_0`) | désactivé | `STRATA_QSA_MERGE_GATE=1` | bit à bit attendu, à confirmer par `qsa_decode_attn_parity` |
| #21 (O6) | copie FP16 des clés d'indexeur, notation depuis cette copie | désactivé | `--idx-fp16` ou `STRATA_IDX_FP16=1` | **change la sélection** des cellules |

Sous Windows (`cmd`), remplacer `VAR=1 commande` par `set VAR=1` puis la commande.

## #12 (B14) : page non résolue, balayage CLOCK

- **`qsa_decode_attn.cu`**. Si la résolution du streaming KV déborde (`ctl[3]`), la table de pages garde -1 et
  l'ancien code lisait avant le début du pool. Désormais `srow = -1` pour une telle cellule, dans l'ancien kernel comme
  dans le nouveau :
  - score `-FLT_MAX` et poids 0 dans le softmax ;
  - ligne V jamais lue ;
  - un morceau sans aucune cellule résolue a `m = -FLT_MAX`, `l = 0`, et la fusion l'ignore déjà ;
  - une requête sans aucune cellule résolue donne une sortie nulle.

  Quand toutes les pages sont résolues (le cas normal), les conditions ajoutées sont toujours vraies. La séquence
  d'opérations flottantes de l'ancien kernel est identique à celle d'avant (vérifié sur le PTX, voir plus bas).
- **`kv_stream.cu`**. `kv_stream_resolve` refuse (message et sortie) un `n_slots < 1024`. Avec moins d'emplacements
  que de threads du bloc de résolution, deux threads verraient le même emplacement dans une étape du balayage et
  pourraient le prendre deux fois. Le moteur ne descend jamais sous 5 120 emplacements : c'est un garde-fou.

**Gain :** aucun (correctif). **Risque :** nul pour les exécutions normales.

## #22 (O6b) : `block_topk_kernel`

Nouveau kernel `block_topk2_kernel` (512 threads, un bloc par requête), l'ancien reste disponible :

- **passes d'histogramme** (4 × 8 bits, comme avant) : chaque warp lit 32 blocs consécutifs, avec 4 chargements en vol
  par thread ;
- **histogramme par warp** (16 × 256 entiers). Les lanes d'un warp qui tombent dans la même case n'ajoutent qu'une
  fois (`__match_any_sync`), avec le poids exact : R par bloc plein, et le poids propre du bloc de queue ;
- **choix du chiffre** : l'ancien kernel faisait jusqu'à 256 itérations en série sur le thread 0, à chaque passe. Il
  est remplacé par un balayage préfixe parallèle des 256 cases. Le thread t tient le chiffre 255 − t, donc le préfixe
  exclusif vaut « cellules dans les chiffres au-dessus ». Avec S[d] = cellules dont le chiffre est ≥ d, on garde
  le d unique tel que `above + S[d] >= width` (ou d = 0) et `above + S[d+1] < width` : c'est exactement la règle de
  la boucle série ;
- **comptage et émission** : chaque warp possède une plage contiguë de blocs, parcourue 32 blocs à la fois
  (lectures coalescées). Les positions viennent de balayages de warp, et le budget de cellules égales au seuil est
  appliqué dans l'ordre croissant des cellules. Les deux préfixes série de 256 éléments du thread 0 disparaissent.

**Identité :** tous les comptes sont des sommes d'entiers, donc l'ordre des additions ne change rien. Le seuil, le
budget d'égalités et les cellules émises (ordre croissant, égalités vers l'indice le plus bas) sont ceux de l'ancien
kernel. Défaut **activé**, et `STRATA_OLD_TOPK=1` rend l'ancien kernel (lu une fois, avant la capture).

**Gain (HYPOTHÈSE) :** 3 à 5× sur ce kernel à 128K, soit 0,1 à 0,5 ms par token (audit). Les parcours série supprimés
coûtaient à eux seuls de l'ordre de 10 à 20 µs par appel dès que le contexte dépasse 2 051 cellules. Si c'est
confirmé, le gain existe donc aussi à 4K–32K (12 couches QSA par token, plus les fenêtres de vérification).

## #23 (O6c) : `qsa_decode_attn`

Nouveau kernel `attn_chunk_pf_kernel<KV_MODE, CH>`, l'ancien `attn_chunk_kernel` est conservé :

- dès que `srow` est connu, les lignes V du morceau partent en `cp.async` vers la mémoire partagée, par copies de
  16 octets (plus les échelles INT8, 8 octets). Taille : 32 Ko en FP16, 16 Ko en INT8, 9 Ko en Q4_0 pour CHUNK = 64 ;
- chaque warp charge ses 8 lignes K en registres avant son premier produit scalaire : 8 chargements en vol au lieu
  d'un ;
- `cp.async.wait_group 0` + `__syncthreads()` seulement avant la boucle V, qui lit la mémoire partagée. QK et le
  softmax recouvrent donc les lectures V.

**Identité :** mêmes expressions dans le même ordre que l'ancien kernel (même `load8`, même produit, mêmes arbres de
warp, même chaîne de `fmaf` sur les cellules croissantes, même décodage de V depuis les mêmes bits). Le PTX le
confirme : même motif `mul` + 7 `fma.rn` par produit, mêmes `shfl`. Défaut **activé**, et `STRATA_OLD_QSA_ATTN=1`
rend l'ancien kernel. Un pool V non aligné sur 16 octets repasse automatiquement sur l'ancien kernel.

- **CHUNK = 32** (`STRATA_QSA_CHUNK=32`, opt-in) : ~132 blocs au lieu de 66 pour 170 SM. Le softmax est découpé
  autrement, ce qui change l'arrondi final, d'où l'opt-in. La taille du scratch suit la variante, lue une fois au
  démarrage.
- **Porte intégrée à la fusion** (`STRATA_QSA_MERGE_GATE=1`, opt-in) : `attn_merge_gate_kernel` écrit directement
  `attn32` (décodage `layer.cpp` et fenêtres de vérification `verify.cpp`), ce qui fait un lancement de moins par
  couche QSA :
  - porte FP64 (`qsa_gate_apply_f32`) : même source, même TU sans fast-math ; même séquence d'opérations FP64 dans le
    PTX ;
  - porte native (`native_qsa_gate_apply`, TU fast-math) : les instructions que ce TU émet (`mul.ftz` par −log2 e,
    `ex2.approx.ftz`, `add.ftz`, `rcp.approx.ftz`, `mul.ftz`) sont reproduites en PTX inline, en `.rn`, pour
    qu'aucune fusion ne soit possible ;
  - pas avec `--kv q4_0` : la FWHT inverse s'intercale entre l'attention et la porte ;
  - opt-in tant que `qsa_decode_attn_parity` ne l'a pas vu bit à bit sur GPU. Si c'est le cas, on peut l'activer par
    défaut.
- **Registres (ptxas, sm_120)** : ancien kernel 40 registres, 15,9 Ko de mémoire partagée. Nouveau, CHUNK 64 :
  - FP16 : 94 registres, 48,6 Ko ;
  - INT8 : 80 registres, 32,8 Ko ;
  - Q4_0 : 80 registres, 25,1 Ko, avec **8 octets de spill** (négligeable, lus 2 fois).

  En CHUNK 32 : 48 à 62 registres, sans spill. La fusion est inchangée : 40 registres (38 avec porte). À 48,6 Ko, le
  noyau FP16 tient 2 blocs par SM, ce qui ne limite pas une grille de 66 blocs.
- **Lots (prompt, vérification, MTP)** : le même `launch()` sert `qsa_decode_attn_batch`, dont la grille compte `n_q`
  fois plus de blocs ; là, 2 blocs par SM (contre ~5 pour l'ancien kernel à 15,9 Ko) peuvent coûter. Non mesuré.
  `STRATA_QSA_ATTN_BATCH_OLD=1` garde l'ancien kernel pour `n_q > 1` seulement et le nouveau pour le décodage : même
  découpe de 64 cellules, même scratch, donc bit à bit neutre. Si le prompt régresse (commande 3 ci-dessous), ce
  choix par `n_q` deviendra le défaut.
- **Scratch** : sa taille dépend du variant, figé au premier usage. `qsa_decode_attn_set_variant` doit précéder tout
  `qsa_decode_attn_scratch_floats` (seuls les tests changent de variant). Si un variant à chunks plus petits que le
  plus petit jamais dimensionné est demandé après coup, `launch()` repasse en chunks de 64 (un avertissement) au lieu
  de déborder du scratch.

**Gain (ESTIMÉ, audit) :** ~10 µs par couche QSA, soit 0,1 à 0,2 ms par token. La sélection sature à 2 051 cellules,
donc le gain vaut pour tout contexte supérieur à ~2K, pas seulement à 128K. Porte intégrée : 12 lancements de moins
par token (HYPOTHÈSE : 10 à 30 µs par token).

## #21 (O6) : clés d'indexeur en FP16

- `QsaIndexerBuffers` gagne deux pointeurs optionnels, `pooled16` et `dead16`. Les deux kernels de pooling
  (`indexer_key_append` dans `qsa.cu`, `native_qsa_indexer_append`) écrivent `f16_from_f32` de chaque ligne FP32
  qu'ils écrivent (bloc complété, emplacement de réserve déplacé, `dead`). L'arrondi est entier et RNE, donc exact
  même dans le TU fast-math. La copie FP32 reste la référence, et son arithmétique est inchangée (même séquence
  d'opérations flottantes dans le PTX).
- `qsa_block_scores_f16` note les blocs depuis la copie (lecture `uint2` : 256 octets par bloc au lieu de 512), avec
  exactement l'arithmétique FP32 sur les valeurs élargies. Son score est donc celui du chemin FP32 sur des clés
  arrondies en FP16.
- Branchement : `qsa_set_idx_fp16` (layer.hpp) alloue la copie dans l'état QSA (+64 octets par cellule et par couche
  QSA : 96 Mo à 128K, 192 Mo à 262K pour les 12 couches) et la remet à zéro avec l'état. `qsa_indexer_buffers(st)`
  et `qsa_index_scores(st, …)` servent le décodage (`layer.cpp`), la fenêtre de vérification (`verify.cpp`) et le
  prompt (`prefill.cpp`), qui voient ainsi la même clé.
- **Opt-in** : `--idx-fp16` ou `STRATA_IDX_FP16=1` (message au démarrage). **La sélection change** près de la frontière
  (`qsa.hpp`, INDEXER KEYS). Sans l'option, rien n'est alloué ni écrit, et le résultat est identique.
- Non concerné : le chemin `--no-fast-select` (scores FP64 par cellule), qui lit toujours la copie FP32. Avec
  `--no-fast-select`, `--idx-fp16` est donc ignoré (avertissement au démarrage, rien n'est alloué) : sinon le prompt
  et la vérification noteraient en FP16 et le décodage en FP32.

**Gain (ESTIMÉ, audit) :** ~0,15 ms par token à 128K, rien à 4K. **Mesures à faire avant tout passage par défaut** :
% de cellules sélectionnées différentes à 32K et 128K, KL forcé par l'enseignant, aiguilles 5/5.

## Tests ajoutés (enregistrés dans ctest, GPU, sans modèle)

- **`qsa_select_parity`** :
  1. `qsa_block_scores` contre une référence hôte en double (relu par tête, `dead` et +1e9 pour la queue), à 1e-6 de
     la magnitude des termes ;
  2. `qsa_block_topk`, nouveau et ancien kernel, contre une sélection hôte par définition (tri des cellules par
     (clé décroissante, indice croissant)). Égalité exacte, les deux kernels bit à bit égaux, et rien d'écrit au-delà
     de `width`. Contextes de 1 à 131 072 cellules, toutes les longueurs de queue. Scores du GPU et scores
     adversariaux : égalités massives, tout à zéro, −0/+0, NaN, négatifs, sous-normaux, +1e9, pics sur un plancher,
     clés qui ne diffèrent que par le dernier octet ;
  3. la copie FP16 : `qsa_block_scores_f16` égale bit à bit `qsa_block_scores` sur les clés arrondies. Les deux kernels
     de pooling écrivent `pooled16`/`dead16` = `f16_from_f32` de chaque ligne, sans changer un bit des lignes FP32.
     Le taux d'accord de sélection FP16/FP32 est affiché (clés aléatoires : ce n'est pas une mesure du modèle).
- **`qsa_decode_attn_parity`** : pools FP16, INT8 et Q4_0 derrière une table de pages permutée, lot de 12 requêtes
  (largeurs 1 à 2 051) :
  1. les trois variantes contre un softmax hôte en double (2e-4 de max|V|) ;
  2. le nouveau kernel **bit à bit** égal à l'ancien, en lot et requête seule ;
  3. B14 : des pages à −1 (dont un morceau entier et une requête entière) sont masquées dans toutes les variantes ;
  4. un pool V non aligné repasse sur l'ancien kernel (bit à bit) ;
  5. la porte intégrée (FP64 et native) égale bit à bit l'attention suivie du kernel de porte séparé.

## Ce qui a été vérifié ici (sans GPU)

- `cmake --build build-wp` : vert (CUDA 13.0, sm_120). Les deux nouveaux tests compilent. Lancés ici, ils échouent
  proprement : pas de pilote CUDA.
- Tests CPU :
  - réussis : `platform_memory_test`, `pool_stress`, `expert_multi_test`, `suffix_drafter_test`,
    `draft_policy_test`, `controller_test`, `conv_cache_test` ;
  - `pool_test` échoue comme prévu : il faut un pack ;
  - `ple_reader_selftest` est instable ici : les worktrees parallèles partagent `/tmp/ple_reader_selftest.bin`. Il
    passe lancé seul, et ce paquet n'y touche pas.
- **PTX comparé à b858f9b** :
  - identiques : `block_scores_kernel<float>`, l'ancien `block_topk_kernel` et la fusion (aux noms de registres près) ;
  - même séquence d'opérations flottantes : l'ancien `attn_chunk_kernel` gardé (B14), les deux kernels de pooling ;
  - nouveau kernel d'attention : même motif par produit scalaire que l'ancien ;
  - porte FP64 intégrée : même séquence d'opérations FP64 que `qsa_gate_apply_f32_kernel`.
- **Émulation CPU de `block_topk2_kernel`** (lane par lane, même arithmétique d'indices, hors dépôt) contre la
  sélection par définition : 140/140 cas. Sept familles de scores, dont les cas adversariaux ci-dessus, et 20
  longueurs de contexte jusqu'à 131 072 cellules.

## Ce qui reste, et pourquoi

- **Rien n'a tourné sur GPU.** Il faut d'abord les deux tests de parité, puis les mesures.
- **#21** : la copie INT8 avec échelle par ligne n'est pas faite. Le passage par défaut attend les mesures de
  précision (sélection, KL, aiguilles).
- **#22** : chiffres de 11 bits (3 passes) non faits. Il faudrait 2 048 cases, soit 128 Ko d'histogrammes par warp
  ou un histogramme partagé contesté. Le vrai levier restant au décodage (une seule requête, donc un seul SM) est un
  top-k multi-blocs, hors de ce paquet.
- **#23** :
  - FWHT inverse de Q4_0 non intégrée à la fusion : la porte n'y est donc pas intégrée non plus ;
  - la porte n'est pas intégrée dans le brouillon MTP (`mtp.cpp`, hors périmètre) ni dans le prompt (`gate_attn` de
    `prefill`, autre kernel) ;
  - le spill de 8 octets en Q4_0 / CHUNK 64 reste.
- **B14** : pas de test du refus `n_slots < 1024`, car il termine le processus.

## Commandes à lancer sur la RTX 5090

Construire, puis passer les tests (depuis la racine du dépôt) :

```bash
export CCACHE_BASEDIR=$PWD CCACHE_NOHASHDIR=true
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON \
      -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j
cd build
ctest -R "qsa_select_parity|qsa_decode_attn_parity|kv_stream_parity|qsa_parity|kv_q8_parity|kv_q4_parity" \
      --output-on-failure
```

Tout doit afficher PASS. En particulier : « prefetch = previous kernel, bitwise », « the two kernels are bitwise
equal » et « folded into the merge = separate kernel, bitwise ».

Pour la suite, `ARGS` contient les arguments habituels de votre `strata-*.json` (`--pack`, `--native`, `--ple-gguf`,
`--expert-profile`, `--expert-cache auto`, `--prefill auto`, `--spec 4 --spec-min-p 0.5`, `--mtp`, `--kv int8`,
`--kv-resident 32768` à partir de 64K…) et `P1`, `P2`, `P3` sont trois prompts pré-tokenisés (`.ids`). Pour le long
contexte, prendre par exemple 4K, 32K et 128K, avec `--max-context` assez grand.

**1. Temps par kernel au long contexte** (`nsys`, prompt de 128K). `--gpu-only-full` ne convient pas ici : il rejoue
les graphes à la position 0, donc avec une seule cellule sélectionnée.

```bash
RUN="./build/strata $ARGS --tokens-file P3 --max-new 64 --greedy --adapt-every 100000"
nsys profile --cuda-graph-trace=node -o qsa_old --force-overwrite true \
     env STRATA_OLD_TOPK=1 STRATA_OLD_QSA_ATTN=1 $RUN                      # avant
nsys profile --cuda-graph-trace=node -o qsa_new --force-overwrite true $RUN    # après (défaut)
nsys profile --cuda-graph-trace=node -o qsa_c32 --force-overwrite true env STRATA_QSA_CHUNK=32 $RUN
nsys profile --cuda-graph-trace=node -o qsa_gate --force-overwrite true env STRATA_QSA_MERGE_GATE=1 $RUN
nsys profile --cuda-graph-trace=node -o qsa_f16 --force-overwrite true $RUN --idx-fp16
for r in qsa_old qsa_new qsa_c32 qsa_gate qsa_f16; do
  echo "== $r"
  nsys stats --report cuda_gpu_kern_sum $r.nsys-rep | grep -E "topk|attn_chunk|attn_merge|block_scores|gate"
done
```

À comparer :
- `block_topk_kernel` (avant) contre `block_topk2_kernel` (après) ;
- `attn_chunk_kernel` contre `attn_chunk_pf_kernel` ;
- `block_scores_kernel<float>` contre `<unsigned short>` avec `--idx-fp16` ;
- le nombre de lancements de `gate` et de `qsa_gate_apply_f32_kernel` avec la porte intégrée.

**2. A/B de bout en bout**, 3 exécutions × 3 prompts, glouton, 256 tokens, résidence statique :

```bash
for run in 1 2 3; do for P in P1 P2 P3; do
  STRATA_OLD_TOPK=1 STRATA_OLD_QSA_ATTN=1 ./build/strata $ARGS --tokens-file $P --max-new 256 --greedy \
      --adapt-every 100000 > old_${P}_${run}.txt
  ./build/strata $ARGS --tokens-file $P --max-new 256 --greedy --adapt-every 100000 > new_${P}_${run}.txt
done; done
```

Comparer les tokens/s et `tokens_per_round`. **Les tokens générés doivent être identiques** entre `old_*` et `new_*`
(les deux changements par défaut sont bit à bit). Refaire la même boucle avec `STRATA_QSA_CHUNK=32`, puis avec
`STRATA_QSA_MERGE_GATE=1` :
- CHUNK 32 : les tokens peuvent différer très rarement ;
- porte intégrée : ils doivent être identiques.

**3. Prompt** (le top-k et l'attention en lots) :

```bash
STRATA_PREFILL_TIMING=1 STRATA_OLD_TOPK=1 STRATA_OLD_QSA_ATTN=1 ./build/strata $ARGS --tokens-file P3 --max-new 1
STRATA_PREFILL_TIMING=1 ./build/strata $ARGS --tokens-file P3 --max-new 1
STRATA_PREFILL_TIMING=1 STRATA_QSA_ATTN_BATCH_OLD=1 ./build/strata $ARGS --tokens-file P3 --max-new 1
```

Comparer les phases de sélection et d'attention QSA. Si la 3e ligne est plus rapide que la 2e sur l'attention QSA,
le kernel à préchargement (48,6 Ko de mémoire partagée) coûte en occupation sur les lots : le signaler, et garder
`STRATA_QSA_ATTN_BATCH_OLD=1` (bit à bit, les tokens ne changent pas) en attendant d'en faire le défaut.

**4. #21, précision avant tout passage par défaut** (32K et 128K) :

```bash
# accord de sélection FP16 contre FP32, sans --idx-fp16 (le moteur calcule les deux) :
STRATA_IDX_FP16_CHECK=1 ./build/strata $ARGS --tokens-file P_32K --max-new 1
STRATA_IDX_FP16_CHECK=1 ./build/strata $ARGS --tokens-file P_128K --max-new 1
# avec --idx-fp16, le même contrôle doit afficher 100 % : la copie écrite par le pooling
# est bien l'arrondi de la copie FP32
STRATA_IDX_FP16_CHECK=1 ./build/strata $ARGS --tokens-file P_32K --max-new 1 --idx-fp16
# aiguilles : serveur lancé avec --idx-fp16, puis
python tools/needle_bench.py --lengths 32k,128k --depths 10,50,90
```

Ajouter le KL forcé par l'enseignant avec l'outil habituel (`kv_precision_compare.py`, non publié dans ce dépôt) : même
texte, `--idx-fp16` contre défaut.
