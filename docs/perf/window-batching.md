# window-batching : moins de nœuds dans le graphe de fenêtre, combinaison lue en place (#19, #44)

*29/09/2026 : ce document a été écrit sur une machine sans GPU ; la première exécution sur une RTX 5090 est dans
[PERF-CHANGES.md, section 6](../PERF-CHANGES.md#6-première-exécution-sur-gpu-rtx-5090-29092026).*

Branche `perf/window-batching`, partie de `perf/base` (b858f9b, moteur 0.1.20). Aucune mesure GPU n'a été faite :
la machine de développement n'a pas de GPU. Tout a été compilé (CUDA 13.0, sm_120, 0 spill) ; les tests CPU ont
été lancés, le nouveau test GPU `window_batch_parity` compile et reste à lancer sur la RTX 5090. Les gains sont
**ESTIMÉ** (calcul) ou **HYPOTHÈSE** (à confirmer au profileur).

Principe : chaque boucle « par token » du graphe de fenêtre (`Verifier::record_window`) devient **un lancement par
groupe de tokens** (la fenêtre entière, ou chaque moitié avec `--spec-split`). Chaque nouveau chemin est **identique
au bit près, token par token**, à celui qu'il remplace : il est donc actif par défaut, et l'ancien chemin reste
sélectionnable par variable d'environnement pour l'A/B. Rien de ce paquet ne change l'arithmétique.

## Ce qui change

| # | Issue | Avant (par groupe de n tokens, par couche) | Après | Par défaut | Retour à l'ancien chemin |
|---|---|---|---|---|---|
| 1 | #19 routeur | n × (GEMV BF16 2,6 Mo + top-10) | 1 GEMV multi-colonnes (poids lus une fois) + 1 top-10 pour n lignes | activé | `STRATA_OLD_WINDOW_ROUTE=1` |
| 2 | #19 indexeur | n × GEMV `k_proj` + n × GEMV `q_proj` (couches QSA) | 1 + 1 GEMV multi-colonnes | activé | `STRATA_OLD_WINDOW_INDEXER=1` |
| 3 | #19 q/porte QSA | n × (copie 2D q/porte + norme + RoPE [+ FWHT]) et n × porte de sortie | 1 copie 2D, 1 norme, 1 RoPE [1 FWHT], 1 porte pour les n × 24 têtes | activé | `STRATA_OLD_WINDOW_QSA=1` |
| 4 | #19 expert partagé | n × conversion BF16, SwiGLU puis quantification Q8_1, n × (GEMV de porte + sigmoïde `<<<1,1>>>`), mise à l'échelle | conversion BF16 seulement si la porte BF16 est utilisée (1 lancement), SwiGLU+Q8_1 fusionnés, 1 GEMV de porte pour n tokens, sigmoïde et échelle **repliées dans la combinaison** | activé | `STRATA_OLD_WINDOW_SHARED=1` |
| 5 | #44 (E1) combinaison | copie de **toutes** les k lignes depuis `y_miss` mappé (PCIe), `moe_hit_add`, n × combinaison | 1 kernel : lignes CPU lues directement dans `y_miss` (seulement celles qui ne sont pas dans la liste des hits), lignes GPU lues dans `hit_out`, porte partagée appliquée, somme | activé | `STRATA_OLD_WINDOW_COMBINE=1` |
| 6 | #44 (E1) graphe « un token » | copie + `moe_hit_add` + combinaison | le même kernel, n = 1 | activé | `STRATA_OLD_TOKEN_COMBINE=1` |
| 7 | #19 `layer.cpp:398` | `f32_to_bf16_bulk` inconditionnel dans `moe_shared` | sauté quand la porte native est active (personne ne lit la sortie) | activé | — (travail mort) |
| 8 | #44 HYPOTHÈSE | `y_miss` en mémoire épinglée ordinaire | `cudaHostAllocWriteCombined` (le CPU ne fait qu'y écrire) | **opt-in** | `STRATA_YMISS_WC=1` pour l'activer |

`STRATA_OLD_WINDOW=1` rétablit d'un coup 1 à 5 (la fenêtre entière comme en 0.1.20). Les variables sont lues une
fois ; un graphe déjà capturé garde son choix.

Garde-fou de la ligne 5 : la combinaison en place n'est prise que si 2 ≤ K ≤ 15 et (taille du plus grand groupe) × K
≤ 128 (plages des deux kernels, comme le graphe « un token » de `session.cpp`). Sinon (modèle avec K = 1 ou K > 15 dans
le GGUF) la fenêtre reprend d'elle-même l'ancienne suite copie + `moe_hit_add` + combinaison par token, et l'expert
partagé applique alors sa propre porte (`defer_gate = false`), exactement comme avec `STRATA_OLD_WINDOW_COMBINE=1`.
Les modèles livrés (K = 10, fenêtre ≤ 8 tokens) prennent toujours le nouveau chemin.

### Nouveaux kernels et pourquoi ils sont identiques au bit près

- **`bf16_gemv_fp32_mmvf_multi`** (`native_bf16.cu`) : un bloc par ligne de sortie comme `bf16_gemv_fp32_mmvf`, un
  accumulateur par colonne. Même taille de bloc, même attribution des paires aux threads, les deux FMA par paire
  sont les mêmes `__fmaf_rn` explicites, puis les mêmes réductions (warp XOR puis `partials`). La ligne de poids est
  lue une fois pour les n colonnes au lieu de n fois.
- **`native_router_top10_multi`** (`native_router.cu`) : un warp par token dans le bloc 32×8 d'origine. Le kernel
  à un token appelle maintenant le même corps (`route_warp`) ; son PTX est **inchangé** (vérifié par diff). Le
  routeur non natif (`router_top10`) acceptait déjà n lignes : un bloc par ligne, même code.
- **`native_swiglu_quantize_q8_1`** (`native_mmvq.cu`) : la quantification Q8_1 est le même corps que
  `native_quantize_q8_1` (fonction partagée, PTX de l'ancien kernel inchangé). La SwiGLU, elle, vient de
  `shared_expert.cu`, qui n'est **pas** compilé avec `--use_fast_math` : ses `__expf`/`__fdividef` ne mettent pas
  les dénormaux à zéro, alors que les mêmes intrinsèques dans `native_mmvq.cu` le feraient. Elle est donc écrite en
  PTX (`mul.rn`, `ex2.approx`, `add.rn`, `div.approx`, `mul.rn`, non-FTZ) : ce sont exactement les instructions que
  nvcc 13.0 produit pour `native_swiglu_kernel` (comparé au PTX). Le `.rn` interdit seulement la contraction en
  FMA, qui n'avait pas lieu non plus dans l'ancien kernel (le produit y était stocké).
- **`native_moe_combine_window`** (`native_moe.cu`) / **`moe_combine_window`** (`shared_expert.cu`, somme en double) :
  - une ligne GPU vaut `+0 + hit` : le pool met à zéro la ligne de `y_miss` d'un expert calculé sur le GPU, puis
    `moe_hit_add` lui ajoutait `hit_out` (addition non-FTZ, écrite telle quelle en PTX) ; `-0`, dénormaux et NaN
    des lignes non lues sont couverts par le test ;
  - une ligne CPU est lue dans `y_miss` (`__ldcv`, float4) : mêmes octets que la copie ;
  - la porte partagée, quand elle est repliée ici : `sigmoïde(dot)` avec la séquence non-FTZ de
    `native_scalar_sigmoid_kernel`, puis `out × g` arrondi seul (`mul.rn` non-FTZ, comme `scale_rows_kernel`) ;
  - la combinaison : premier produit, FMA dans l'ordre des experts, ajout du terme partagé — en intrinsèques à
    arrondi unique explicites (`__fmul_rn`, `__fmaf_rn`, `__fadd_rn` en FTZ comme le reste de `native_moe.cu`),
    ce qui est exactement l'arithmétique de `combine` pour k ≥ 2 (k = 10 ici ; k = 1 est refusé par ce kernel
    parce que l'ancien pouvait y contracter produit et ajout en une FMA).
- **`shared_expert_multi_batched`** (`shared_expert.cu`) : même ordre de lancements que `shared_expert_multi`, avec
  les fusions ci-dessus ; la porte BF16 (non native) utilise un bloc par token (`scalar_gate_rows_kernel`, même corps
  que `scalar_gate_kernel`, PTX de ce dernier inchangé). Le mode de porte rendu (`SharedGate::Applied/Value/Logit`)
  dit à la combinaison ce qui reste à appliquer.
- Groupes QSA (point 3) : les têtes d'un groupe sont des lignes de même pas dans `qfull_` et `qcur_`, `pos_`
  contient une position **par ligne** (NH par token), et la norme (native ou non), la RoPE (native ou table), la
  FWHT et la porte de sortie sont par ligne ou par élément : un seul appel avec n × NH lignes calcule les mêmes
  valeurs. La norme et la RoPE de **k** ne sont pas groupées : `pos_` n'a pas le pas NKV qu'il leur faudrait.

Registres (ptxas -v, sm_120), aucun spill : `combine_window` 78–79, `moe_combine_window_kernel` 80,
`bf16_f32_mmvf_multi_kernel` 46 (contre 37 pour le kernel à une colonne), `route_multi` 59 (54),
`native_swiglu_quantize_q8_1_kernel` 16, `scalar_gate_rows_kernel` 22, `sigmoid_scale_rows_kernel` 10.

### Fichiers

`src/kernels/cuda/{native_bf16,native_router,native_mmvq,native_moe,shared_expert}.cu` et leurs en-têtes ;
`src/core/layer.cpp` (`moe_route_multi`, `moe_combine_rows`, `block_layer_post(…, rows)`, `moe_shared`) ;
`src/core/verify.cpp` (`record_window` : routeur, indexeur, q/porte, expert partagé, combinaison ; allocation de
`y_miss`) ; `src/core/session.cpp` (graphe « un token ») ; test `src/kernels/window_batch_parity.cpp`. Les parties
PLE, synchronisation du commit et `fetch_dma` de `verify.cpp` ne sont pas touchées.

## Gains attendus

Nœuds retirés du graphe, par couche et par groupe de n tokens (préréglage `--native`) : routeur 2n−2, expert
partagé 3n+1, combinaison n+1 ; en plus sur les 12 couches QSA : indexeur 2(n−1), q/porte 3(n−1) (4(n−1) avec le
KV Q4) et porte de sortie n−1.

| fenêtre | nœuds en moins (**ESTIMÉ**, décompte du code) |
|---|---|
| T = 4 | 1 368 |
| T = 5 | 1 728 |
| T = 6 (`--spec 4` + brouillon suffixe) | 2 088 |
| T = 8 | 2 808 |

(360·T − 72 ; avec `--spec-split`, 360·T − 144.)

- **HYPOTHÈSE** (audit : 1 à 2 µs par nœud, et les kernels groupés font le même travail en parallèle au lieu
  d'en série) : **1 à 3 ms par tour**, soit 3 à 8 % sur la 5070 et 6 à 15 % sur la 5090.
- **ESTIMÉ** pour E1 : l'ancienne copie lisait T × 4,9 Mo par fenêtre sur PCIe (k lignes de 10 Ko par couche et par
  token, y compris les lignes nulles des experts GPU), sur le chemin critique de chaque couche. Seules les lignes CPU
  sont lues maintenant : avec ~93 % des experts en VRAM (5090), ~0,9 ms de moins par fenêtre de 5 tokens à
  ~25 Go/s ; ~0,5 ms sur la 5070 (~50 % en VRAM).
- `STRATA_YMISS_WC=1` : **HYPOTHÈSE** seulement (lecture PCIe sans espionnage des caches CPU, écritures CPU en
  combinaison d'écriture). Valeurs inchangées ; à garder seulement si l'A/B le montre.

## Tests

Sur la machine de développement (CPU seulement) :

- construction complète `-DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON` (sm_120) : **OK**, aucun nouvel
  avertissement ;
- `ple_reader_selftest`, `platform_memory_test`, `pool_stress`, `expert_multi_test`, `suffix_drafter_test`,
  `draft_policy_test`, `controller_test`, `conv_cache_test` : **OK** ; `pool_test` échoue comme attendu (il lui faut
  un pack de modèle) ; `python3 serve/test_server.py` : **OK** (26 tests) ;
- PTX : les anciens kernels (`route`, `native_quantize_q8_1_kernel`, `combine`, `scalar_gate_kernel`,
  `scale_rows_kernel`, `moe_combine_kernel`, `native_swiglu_kernel`, `native_scalar_sigmoid_kernel`…) sont
  **identiques** avant/après ; les nouveaux ont la même suite d'opérations flottantes que ceux qu'ils remplacent
  (comparé instruction par instruction).

Nouveau test GPU, **compilé, pas exécuté** : `window_batch_parity` (ctest). Tout est comparé par `memcmp` :

1. GEMV multi-colonnes contre une GEMV par colonne (formes du routeur, de l'indexeur, de la porte, pas de ligne
   non tassés, 1 à 8 colonnes) ;
2. les deux routeurs, groupés contre un appel par token (égalités de logits incluses, 1 à 20 tokens) ;
3. la combinaison en place contre copie + `moe_hit_add` + combinaison par token, native et double, 1 à 8 tokens,
   0 à 100 % de lignes GPU, liste de hits mélangée, `-0` et dénormaux, NaN dans les lignes qui ne doivent pas être
   lues, sans liste de hits ; puis par groupe de tokens comme `--spec-split` l'enregistre (lignes décalées du
   premier token du groupe, liste de hits relative au groupe, 2, 7 et 8 tokens) ;
4. l'expert partagé groupé contre `shared_expert_multi` (porte native et BF16) : sortie, et Q8_1 de la SwiGLU ;
   puis la chaîne complète porte différée + combinaison contre l'ancienne chaîne, pour les deux combinaisons ;
5. copie q/porte, norme, RoPE, FWHT et porte de sortie groupées contre par token (noyaux natifs et non natifs).

## À valider sur la RTX 5090

```bash
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON \
      -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j

# tests : le nouveau, puis ceux des kernels voisins
ctest --test-dir build -R "window_batch_parity|shared_expert_parity|router_top10_parity|bf16_gemv_parity|elementwise_parity|native_expert_parity|qsa_parity" --output-on-failure
```

Benchmark A/B : 3 prompts × 3 exécutions, glouton, 256 tokens, résidence statique (`--adapt-every 100000`).
Remplacer `<OPTS>` par la ligne habituelle (`--pack`, `--ple-gguf`, `--native`, `--expert-profile`,
`--expert-cache`, `--spec`, …).

```bash
OPTS="<OPTS> --greedy --max-new 256 --adapt-every 100000"
mkdir -p ab
for p in p1 p2 p3; do for r in 1 2 3; do
  # ancien graphe de fenêtre (et ancien graphe « un token »)
  STRATA_OLD_WINDOW=1 STRATA_OLD_TOKEN_COMBINE=1 ./build/strata $OPTS --tokens-file prompts/$p.ids > ab/old_${p}_$r.txt 2>&1
  # nouveau (défaut)
  ./build/strata $OPTS --tokens-file prompts/$p.ids > ab/new_${p}_$r.txt 2>&1
  # nouveau + y_miss write-combined (opt-in)
  STRATA_YMISS_WC=1 ./build/strata $OPTS --tokens-file prompts/$p.ids > ab/wc_${p}_$r.txt 2>&1
done; done
# 1) mêmes tokens partout (identique au bit près par construction) :
for p in p1 p2 p3; do for v in new wc; do
  diff <(grep '^output' ab/old_${p}_1.txt) <(grep '^output' ab/${v}_${p}_1.txt) && echo "$p $v: mêmes tokens"
done; done
# 2) débit et temps d'attente du GPU par tour :
grep -H "^decode\|^verify window\|^speculation" ab/*.txt
```

Sous PowerShell : `$env:STRATA_OLD_WINDOW="1"; $env:STRATA_OLD_TOKEN_COMBINE="1"` avant la commande, puis
`Remove-Item Env:STRATA_OLD_WINDOW, Env:STRATA_OLD_TOKEN_COMBINE`.

Critères :

- les lignes `output` sont **identiques** entre `old`, `new` et `wc` pour chaque prompt (sinon : lancer
  `window_batch_parity`, puis isoler la partie fautive avec `STRATA_OLD_WINDOW_ROUTE/INDEXER/QSA/SHARED/COMBINE=1`
  une par une) ;
- `decode … tok/s` plus haut en `new` ; `verify window wait for rings … ms/round` plus bas (le GPU atteint chaque
  couche plus tôt) ;
- `wc` : à garder seulement s'il est mesurablement meilleur que `new`.

Profil pour compter les nœuds et le temps entre nœuds :

```bash
nsys profile --cuda-graph-trace=node -o win_old -e STRATA_OLD_WINDOW=1 ./build/strata $OPTS --tokens-file prompts/p1.ids
nsys profile --cuda-graph-trace=node -o win_new ./build/strata $OPTS --tokens-file prompts/p1.ids
nsys stats --report cuda_gpu_kern_sum win_old.nsys-rep win_new.nsys-rep
```

## Ce qui reste (et pourquoi)

- **Programmatic Dependent Launch** (O4) : non fait. Il faut changer la façon de lancer presque chaque kernel du
  graphe (attributs de lancement + `cudaGridDependencySynchronize()` dans les kernels) ; c'est transversal et hors de
  ce paquet.
- **Norme + RoPE de k, écritures KV et ajout à l'indexeur par token** : restent par token. La RoPE de k demanderait
  un tableau de positions au pas NKV (préparation hôte en plus) ; les écritures KV et l'ajout à l'indexeur prennent
  une fiche de pas par token et demanderaient de nouveaux kernels (`kv_q8.cu`, `kv_q4.cu`,
  `native_qsa_indexer.cu`).
- **Un seul kernel BF16 + Q8_0 + Q8_1** : non fait. Les quantifications Q8_1 de `native_mmvq.cu` (fast-math) et de
  `iq_kernels.cu` (IEEE) n'ont pas la même arithmétique ; les fusionner changerait des valeurs (il faudrait un opt-in).
  La conversion BF16, elle, a disparu du chemin natif.
- **Séparation q/porte du graphe « un token »** (`layer.cpp:887`) : faire lire la moitié q par le kernel de norme
  demande une variante à pas d'entrée de `native_qsa_rms_norm_weighted` (`native_qsa.cu`) ; non fait. Dans la
  fenêtre de vérification, la copie est désormais une seule par groupe (point 3).
- La tête (`lm_head_mix` par token) et le `gr_write` par token de la dernière couche restent par token.
- Le chemin `session_loop` (graphes par couche, copie `cudaMemcpyAsync` de `y_miss`) n'est pas modifié.
