# Changements de performance et de précision (branche `perf/all`)

`perf/all` part de `perf/base` (b858f9b, moteur 0.1.20) et réunit les seize lots de l'audit
([`docs/AUDIT-PERF.md`](AUDIT-PERF.md)), fusionnés un par un dans cet ordre : `ple-io`, `round-sync`,
`grouped-experts`, `iq-kernels`, `window-batching`, `qsa-small`, `sampler`, `qsa-longctx`, `grids`, `cpu`,
`server-tools`, `prefill-kernels`, `prefill-host`, `prefill-dense`, `correctness`, `research`. Chaque lot a son
document dans [`docs/perf/`](perf/) :

| Lot | Document | Issues |
|---|---|---|
| `ple-io` | [perf/ple-io.md](perf/ple-io.md) | #15 |
| `round-sync` | [perf/round-sync.md](perf/round-sync.md) | #16 |
| `grouped-experts` | [perf/grouped-experts.md](perf/grouped-experts.md) | #17 |
| `iq-kernels` | [perf/iq-kernels.md](perf/iq-kernels.md) | #4, #18 |
| `window-batching` | [perf/window-batching.md](perf/window-batching.md) | #19, #44 |
| `qsa-small` | [perf/qsa-small.md](perf/qsa-small.md) | #24, #25, #47 |
| `sampler` | [perf/sampler.md](perf/sampler.md) | #20 |
| `qsa-longctx` | [perf/qsa-longctx.md](perf/qsa-longctx.md) | #12, #21, #22, #23 |
| `grids` | [perf/grids.md](perf/grids.md) | #26 |
| `cpu` | [perf/cpu.md](perf/cpu.md) | #11, #27, #28, #29, #30 |
| `server-tools` | [perf/server-tools.md](perf/server-tools.md) | #31 à #35, #49, #50, #52 |
| `prefill-kernels` | [perf/prefill-kernels.md](perf/prefill-kernels.md) | #6, #7, #37, #38, #43 |
| `prefill-host` | [perf/prefill-host.md](perf/prefill-host.md) | #36, #40, #42, #45, #46, #48 |
| `prefill-dense` | [perf/prefill-dense.md](perf/prefill-dense.md) | #39, #41 |
| `correctness` | [perf/correctness.md](perf/correctness.md) | #2, #3, #5, #8, #9, #10, #13 |
| `research` | [perf/research.md](perf/research.md) | #51, #53 |

**Rien n'a tourné sur un GPU.** La machine de développement n'en a pas : tout le code GPU est compilé (CUDA 13.0,
sm_120) mais aucun kernel n'a été exécuté. Les tests CPU et Python passent (voir la fin). Règle suivie par tous les
lots : ce qui est identique au bit près par construction est **actif par défaut** avec une variable
`STRATA_OLD_*` (ou équivalente) pour l'A/B ; ce qui change des valeurs est **désactivé par défaut** (opt-in). La
procédure de la section 3 est donc à faire sur la RTX 5090 avant de se fier aux défauts.

## 1. Les issues, une par ligne

Statut : **fait** = traité comme demandé ; **partiel** = une partie seulement, le reste est dans la section 4.
Dernière colonne : ce qui est actif sans rien passer, et l'option exacte pour revenir en arrière ou pour activer.

| # | Titre | Lot | Statut | Par défaut / option exacte |
|---|---|---|---|---|
| #2 | [B2] s_gemv_q8_split_kernel : return avant __syncthreads (comportement indéfini latent) | correctness | fait | actif (correctif, même kernel), aucune option |
| #3 | [B4] RoPE natif en fast-math : précision de phase aux grandes positions (non testée) | correctness | fait | test `rope_parity` ; angle FP64 **désactivé**, `STRATA_ROPE_F64=1` |
| #4 | [B5] Deux quantificateurs Q8_1 compilés différemment (IEEE vs fast-math) | iq-kernels | partiel | **désactivé** : `STRATA_IQ_FASTDIV=1` (l'écart reste tant que ce n'est pas le défaut) |
| #5 | [B6] Softplus différent entre le GDN fusionné et le pré-traitement natif | correctness | fait | vérifié et documenté, aucun changement de calcul |
| #6 | [B7] Prefill : conversions FP16 sans écrêtage (risque d'inf/NaN) | prefill-kernels | fait | **désactivé** : `STRATA_PREFILL_F16_SAT=1` |
| #7 | [B8] Routeur : expf précis au prefill, fast-math au décodage ; NaN masqués | prefill-kernels | fait | **désactivé** : `STRATA_PREFILL_ROUTE_DECODE=1` ; comptage NaN avec `STRATA_DBG_NAN=1` |
| #8 | [B9] Tests de parité manquants sur des kernels de production | correctness | fait | tests ctest (voir section 3) |
| #9 | [B10] Étalon de qualité du prefill suspect : KL 0,33 entre morceaux de 6144 et 8192 | correctness | fait | outil `tools/prefill_chunk_check.py` |
| #10 | [B11] --spec sans --mtp remplit les fenêtres de token 0 | correctness | fait | **désactivé** : `STRATA_SPEC_T1=1` (fenêtres d'un token sans brouillon) |
| #11 | [B13] expert_pool_dispatch_multi : tableaux fixes kind[128]/distinct[128]/dma_src[64] sans garde | cpu | fait | actif (gardes et `static_assert`), aucune option |
| #12 | [B14] qsa_decode_attn : pas de garde page < 0 si le streaming KV déborde | qsa-longctx | fait | actif (correctif), aucune option |
| #13 | [B15] bf16_from_f32 transforme NaN en -0 | correctness | fait | actif (identique hors NaN), aucune option |
| #14 | [T1] Les tests de parité ne compilent pas : add_subdirectory(tests) sur un dossier absent | perf/base (b858f9b) | fait | `-DSTRATA_BUILD_TESTS=ON` configure sans `tests/` |
| #15 | [O1] Sortir les lectures n-gram (PLE) du chemin critique de chaque fenêtre | ple-io | partiel | actif ; `STRATA_OLD_PLE_STAGING=1`, `STRATA_PLE_IO_BACKEND=uring\|threads\|sync`, `STRATA_PLE_IO_THREADS=N` |
| #16 | [O2] Supprimer les allers-retours hôte↔GPU du tour spéculatif | round-sync | fait | actif ; `STRATA_OLD_MTP_CHAIN=1`, `STRATA_OLD_COMMIT_SYNC=1`, `STRATA_OLD_RES_UPLOAD=1`, `STRATA_OLD_ADAPT_THREAD=1`, `STRATA_OLD_DMA_FLAG=1` |
| #17 | [O3] Kernel groupé des experts VRAM : conflits de banques 8-way + somme hx recalculée | grouped-experts | partiel | actif ; `STRATA_OLD_GROUPED=1` ; seuil `STRATA_GROUPED_PAIR_MIN_HITS=N` (défaut 0) |
| #18 | [O3b] native_expert_grouped / iq_mmvq : grilles IQ re-décodées pour chaque colonne | iq-kernels | fait | actif ; `STRATA_OLD_IQ_MMVQ=1` |
| #19 | [O4] Réduire le nombre de nœuds du graphe de fenêtre (lots, fusions, PDL) | window-batching | partiel | actif ; `STRATA_OLD_WINDOW=1` ou `STRATA_OLD_WINDOW_ROUTE\|INDEXER\|QSA\|SHARED\|COMBINE=1` |
| #20 | [O5] Sampler non glouton : top-k en O(k²·V) sur un SM + fin FP64 calculée par 1024 threads | sampler | fait | actif ; `STRATA_OLD_SAMPLER=1`, `STRATA_SAMPLER_ONE_BLOCK=1` |
| #21 | [O6] Contexte long : indexeur QSA en FP32 relu à chaque requête | qsa-longctx | fait | **désactivé** : `--idx-fp16` ou `STRATA_IDX_FP16=1` |
| #22 | [O6b] block_topk_kernel : un seul bloc, lectures non coalescées, atomiques disputées | qsa-longctx | fait | actif ; `STRATA_OLD_TOPK=1` |
| #23 | [O6c] qsa_decode_attn : lectures V sérielles, pas de recouvrement, 66 blocs | qsa-longctx | fait | actif ; `STRATA_OLD_QSA_ATTN=1`, `STRATA_QSA_ATTN_BATCH_OLD=1` ; opt-in `STRATA_QSA_CHUNK=32`, `STRATA_QSA_MERGE_GATE=1` |
| #24 | [O6d] Pas/position QSA recopiés depuis l'hôte à chaque couche (22 petits kernels par token) | qsa-small | fait | actif ; `STRATA_OLD_QSA_STEP=1` |
| #25 | [O6e] QSA : norme, RoPE, FWHT, quantification et ajout KV en lancements séparés | qsa-small | fait | actif ; `STRATA_OLD_QSA_PREP=1` |
| #26 | [O7] Grilles calibrées pour 48 SM : sous-remplissage sur RTX 5090 (170 SM) | grids | fait | actif ; `STRATA_OLD_GRIDS=1`, `STRATA_OLD_GDN_AB=1`, `STRATA_OLD_GDN_STEP=1` |
| #27 | [O8] Pool CPU : run_split déséquilibré (20 tâches sur 6 threads) | cpu | fait | actif ; `STRATA_OLD_RUN_SPLIT=1` |
| #28 | [O8b] Arène d'experts Linux en pages de 4 Ko sans MADV_HUGEPAGE | cpu | fait | actif ; `STRATA_NO_THP=1` ; `STRATA_THP_REPORT=0` saute la mesure |
| #29 | [O8c] Noyau CPU Q2_0 : prélecture des échelles et surcoût par ligne de la projection down | cpu | partiel | prélecture active : `STRATA_OLD_CPU_PREFETCH=1` (ou `STRATA_CPU_PREFETCH=0`, `STRATA_CPU_PREFETCH_CODES=0`) ; correction entière **désactivée** : `STRATA_CPU_INT_CORR=1` |
| #30 | [O8d] Noyaux CPU i-quant limités par le calcul (~5 Go/s par cœur) | cpu | fait | actif ; `STRATA_OLD_IQ512=1`, `STRATA_IQ512_NOPACK=1` ; opt-in `STRATA_IQ512_ONE=1` |
| #31 | [O9] Serveur : détokenisation quadratique (MESURÉ 2,2 ms/token à 16K) | server-tools | fait | actif ; `STRATA_OLD_DETOK=1` |
| #32 | [O9b] Serveur : conversation entière re-tokenisée à chaque requête (BPE Python sans cache) | server-tools | fait | actif ; `STRATA_OLD_PROMPT_ENCODE=1` ; contrôle `STRATA_CHECK_PROMPT_IDS=1` |
| #33 | [O9c] Changer le niveau de réflexion relit toute la conversation | server-tools | partiel | actif (message seulement), aucune option |
| #34 | [O9d] chat.py et clients OpenAI : la réutilisation « live » échoue (réflexion non renvoyée) | server-tools | fait | `chat.py` renvoie la réflexion (`--drop-thinking` pour l'ancien comportement) ; côté serveur **désactivé** : `--recall-reasoning`, `"recall_reasoning": true` ou `STRATA_RECALL_REASONING=1` |
| #35 | [O9e] Images WebP/TIFF re-normalisées à chaque tour | server-tools | fait | actif (mêmes embeddings), aucune option |
| #36 | [P1] Prefill MoE : synchronisation hôte par couche + ~6 appels CUDA par expert | prefill-host | fait | actif ; `STRATA_OLD_MOE_GROUP=1`, `STRATA_PREFILL_COALESCE=0` |
| #37 | [P2] Prefill gdn_conv : filtre FIR exécuté comme une récurrence série sur 8192 tokens | prefill-kernels | fait | actif ; `STRATA_OLD_PREFILL_GDN_CONV=1` |
| #38 | [P3] Prefill gdn_rec_kernel : 48 blocs, ~5 barrières par token | prefill-kernels | fait | actif ; `STRATA_OLD_PREFILL_GDN_REC=1` |
| #39 | [P4] Prefill : projections denses en FP16 cuBLAS au lieu de MMQ int8 | prefill-dense | fait | **désactivé** : `--prefill-dense-mmq` ou `STRATA_PREFILL_DENSE_MMQ=1` |
| #40 | [P5] Prefill : native_qsa_indexer_append lancé une fois par token (~98 000 lancements par morceau) | prefill-host | fait | actif ; `STRATA_OLD_IDX_APPEND=1` |
| #41 | [P6] Prefill : attention QSA faite par le kernel de décodage, par lots de 32 requêtes | prefill-dense | partiel | mesure seulement : `STRATA_PREFILL_TIMING=1` |
| #42 | [P7] Re-remplissage bloquant des emplacements prêtés après chaque prompt | prefill-host | fait | **désactivé** : `STRATA_ASYNC_REFILL=1` |
| #43 | [P8] Prefill : chaîne GR élément par élément (~290 Go de trafic par morceau) | prefill-kernels | fait | actif ; `STRATA_OLD_PREFILL_GR=1` |
| #44 | [E1] Combinaison MoE : k lignes relues sur PCIe dont les lignes GPU nulles, 3 kernels | window-batching | fait | actif ; `STRATA_OLD_WINDOW_COMBINE=1`, `STRATA_OLD_TOKEN_COMBINE=1` ; opt-in `STRATA_YMISS_WC=1` |
| #45 | [E2] Points de reprise de conversation : copies synchrones vers des vecteurs paginables | prefill-host | fait | actif ; `STRATA_OLD_CKPT=1` |
| #46 | [E3] Chercheur de suffixe reconstruit de zéro à chaque requête | prefill-host | fait | actif ; `STRATA_OLD_SFX_RESET=1` |
| #47 | [E4] Table RoPE FP64 allouée (64 Mio à 262K) mais inutilisée en mode --native | qsa-small | fait | actif ; `STRATA_OLD_ROPE_TABLE=1` |
| #48 | [E5] Démarrage : verify_slot relit chaque emplacement du profil en entier | prefill-host | fait | actif ; `STRATA_OLD_PROFILE_FILL=1` |
| #49 | [E6] make_profile.py : les traces ne changent rien au profil livré | server-tools | fait | actif dans l'outil (`--trace-weight 0.5`) ; `--trace-weight 0` = ancienne sortie |
| #50 | [S1] Mesurer : fenêtre spéculative plus longue (--spec 5/6) sur RTX 5090 | server-tools | fait | **désactivé** : `calibrate.py --spec 4,5,6` ou `STRATA_CALIBRATE_SPEC=4,5,6` |
| #51 | [S3] Évaluer : prédire les experts de la couche suivante pour lancer leurs copies en avance | research | fait | outil `tools/routing_locality.py` ; `--dump-routing` trace aussi les fenêtres |
| #52 | [S4] Évaluer : meilleure quantification du brouillon MTP (taux d'acceptation) | server-tools | fait | **désactivé** : `mtp_pack.py --q2-search wide\|exact` ; A/B `tools/ab_oneshot.py --variant` |
| #53 | [S5] Évaluer : état GDN en BF16 (moitié moins de trafic) | research | fait | **désactivé** : `--gdn-state-bf16` |
| #54 | [S6] Évaluer : prefill couche par couche (chaque expert diffusé une fois par prompt) | — | non traité | — |

## 2. Tous les interrupteurs

Sauf mention contraire, une variable est lue une fois par processus ; `=1` (toute valeur non vide autre que `0`)
l'active ; un graphe CUDA déjà capturé garde son choix. Sous Windows (`cmd`) : `set VAR=1` puis la commande.

### 2.1 Retour à l'ancien chemin (A/B des défauts identiques au bit près)

| Interrupteur | Défaut | Effet | Lot |
|---|---|---|---|
| `STRATA_OLD_PLE_STAGING=1` | absent | lecture PLE bloquante avant le lancement de la fenêtre, copie en tête du graphe | ple-io |
| `STRATA_PLE_IO_BACKEND=uring\|threads\|sync` | `uring` (repli `threads` si le noyau refuse) | backend des lectures PLE sous Linux ; `sync` = l'ancienne lecture sous verrou | ple-io |
| `STRATA_PLE_IO_THREADS=N` | `min(--ple-inflight, 16)` | taille du pool du backend `threads` (1..64) | ple-io |
| `STRATA_OLD_MTP_CHAIN=1` | absent | chaîne MTP pas à pas (un graphe + une synchro par pas) au lieu du nœud `WHILE` | round-sync |
| `STRATA_OLD_COMMIT_SYNC=1` | absent | l'hôte attend chaque `commit` | round-sync |
| `STRATA_OLD_RES_UPLOAD=1` | absent | `d_res` par `cudaMemcpy` synchrone | round-sync |
| `STRATA_OLD_ADAPT_THREAD=1` | absent | un `std::thread` par tour d'adaptation au lieu du thread persistant | round-sync |
| `STRATA_OLD_DMA_FLAG=1` | absent | drapeau B par fonctions hôtes (`--pcie-mode dma`), la nouvelle règle d'ordre est gardée | round-sync |
| `STRATA_OLD_GROUPED=1` | absent | anciens kernels Q2_0 groupés et par hit | grouped-experts |
| `STRATA_GROUPED_PAIR_MIN_HITS=N` | 0 | garde les anciens kernels par hit sous N hits | grouped-experts |
| `STRATA_OLD_IQ_MMVQ=1` | absent | anciens kernels IQ (décodage par colonne/entrée) | iq-kernels |
| `STRATA_OLD_WINDOW=1` | absent | toute la fenêtre de vérification comme en 0.1.20 (les cinq parties ci-dessous) | window-batching |
| `STRATA_OLD_WINDOW_ROUTE=1`, `_INDEXER=1`, `_QSA=1`, `_SHARED=1`, `_COMBINE=1` | absents | une partie seulement : routeur, projections de l'indexeur, q/porte QSA, expert partagé, combinaison | window-batching |
| `STRATA_OLD_TOKEN_COMBINE=1` | absent | combinaison du graphe « un token » en trois kernels | window-batching |
| `STRATA_OLD_QSA_STEP=1` | absent | pas/positions QSA téléversés par chaque couche | qsa-small |
| `STRATA_OLD_QSA_PREP=1` | absent | préparation QSA en lancements séparés | qsa-small |
| `STRATA_OLD_ROPE_TABLE=1` | absent | table RoPE toujours allouée | qsa-small |
| `STRATA_OLD_SAMPLER=1` | absent | `sampler_kernel` de 0.1.20 (échantillonnage seulement ; le glouton ne change pas) | sampler |
| `STRATA_SAMPLER_ONE_BLOCK=1` | absent | nouveau seuil et fin sur un warp, mais un bloc par ligne (pas de top-k réparti) | sampler |
| `STRATA_OLD_TOPK=1` | absent | ancien `block_topk_kernel` | qsa-longctx |
| `STRATA_OLD_QSA_ATTN=1` | absent | ancien kernel d'attention QSA (décodage et lots) | qsa-longctx |
| `STRATA_QSA_ATTN_BATCH_OLD=1` | absent | ancien kernel pour les lots (`n_q > 1`, prefill et fenêtres) seulement | qsa-longctx |
| `STRATA_OLD_GRIDS=1` | absent | `fetch_blobs`/`gather_rows` sur `48 × 8` blocs | grids |
| `STRATA_OLD_GDN_AB=1` | absent | `gdn_ab` un warp par ligne (12 blocs) | grids |
| `STRATA_OLD_GDN_STEP=1` | absent | pas GDN un bloc par tête (pas de cluster) | grids |
| `STRATA_OLD_RUN_SPLIT=1` | absent | ancien découpage des tâches du pool CPU | cpu |
| `STRATA_NO_THP=1` | absent | pas de `madvise(MADV_HUGEPAGE)` sur l'arène | cpu |
| `STRATA_THP_REPORT=0` | actif | ne lit pas `smaps` pour la note `AnonHugePages` | cpu |
| `STRATA_OLD_CPU_PREFETCH=1` | absent | coupe la prélecture logicielle | cpu |
| `STRATA_CPU_PREFETCH=<octets>`, `STRATA_CPU_PREFETCH_CODES=<octets>` | 2048 | distance de prélecture (échelles, codes) ; 0 = coupé | cpu |
| `STRATA_OLD_IQ512=1` | absent | anciens noyaux i-quant AVX-512 | cpu |
| `STRATA_IQ512_NOPACK=1` | absent | `gather` sans la copie alignée des activations | cpu |
| `STRATA_OLD_DETOK=1` | absent | serveur : re-décodage complet à chaque token | server-tools |
| `STRATA_OLD_PROMPT_ENCODE=1` | absent | serveur : encodage complet du prompt | server-tools |
| `STRATA_CHECK_PROMPT_IDS=1` | absent | serveur : encode des deux façons, compare, repli sur l'encodage complet | server-tools |
| `STRATA_OLD_PREFILL_GDN_CONV=1`, `_GDN_REC=1`, `_GR=1` | absents | anciens kernels du prompt (`gdn_conv`, `gdn_recurrence`, chaîne GR + tampon FP32) | prefill-kernels |
| `STRATA_OLD_IDX_APPEND=1` | absent | ajout de l'indexeur QSA token par token au prefill | prefill-host |
| `STRATA_OLD_MOE_GROUP=1` | absent | regroupement des paires sur l'hôte, experts un par un | prefill-host |
| `STRATA_PREFILL_COALESCE=0` | 1 | copies d'experts une par une (pas de fusion des contigus) | prefill-host |
| `STRATA_OLD_SFX_RESET=1` | absent | chercheur de suffixe reconstruit à chaque requête | prefill-host |
| `STRATA_OLD_CKPT=1` | absent | points de reprise synchrones en mémoire paginable | prefill-host |
| `STRATA_OLD_PROFILE_FILL=1` | absent | remplissage du profil au démarrage synchrone, relecture complète | prefill-host |

### 2.2 Options qui changent les valeurs (désactivées par défaut)

| Interrupteur | Défaut | Effet | Lot |
|---|---|---|---|
| `STRATA_IQ_FASTDIV=1` | absent | `__fdividef` dans `quantize_q8_1_rows` et le SwiGLU groupé (#4) | iq-kernels |
| `STRATA_YMISS_WC=1` | absent | `y_miss` en mémoire write-combined (valeurs inchangées, hypothèse de vitesse) | window-batching |
| `STRATA_QSA_CHUNK=32` | 64 | attention QSA en morceaux de 32 cellules (autre découpage du softmax) | qsa-longctx |
| `STRATA_QSA_MERGE_GATE=1` | absent | porte sigmoïde dans `attn_merge` (bit à bit attendu, à confirmer ; pas avec `--kv q4_0`) | qsa-longctx |
| `--idx-fp16` / `STRATA_IDX_FP16=1` | absent | notation de l'indexeur depuis une copie FP16 des clés (change la sélection) ; ignoré avec `--no-fast-select` | qsa-longctx |
| `STRATA_CPU_INT_CORR=1` | absent | correction Q2_0 des lignes down en entier | cpu |
| `STRATA_IQ512_ONE=1` | absent | experts IQ à un token sur le noyau AVX-512 au lieu de `vec_dot` | cpu |
| `STRATA_PREFILL_F16_SAT=1` | absent | images FP16 du prompt écrêtées à ±65 504 | prefill-kernels |
| `STRATA_PREFILL_ROUTE_DECODE=1` | absent | routeur du prompt = celui du décodage | prefill-kernels |
| `STRATA_ASYNC_REFILL=1` | absent | re-remplissage asynchrone des emplacements prêtés (experts sur CPU en attendant) | prefill-host |
| `--prefill-dense-mmq` / `STRATA_PREFILL_DENSE_MMQ=1` | absent | projections denses du prompt en MMQ int8 | prefill-dense |
| `STRATA_ROPE_F64=1` | absent | angle RoPE natif en FP64 au décodage (la préparation QSA fusionnée cède alors la place aux lancements séparés) | correctness |
| `STRATA_SPEC_T1=1` | absent | `--spec` sans brouillon : fenêtres d'un token | correctness |
| `--gdn-state-bf16` | absent | état récurrent GDN stocké en BF16 (exige le pas GDN natif fusionné) | research |

### 2.3 Mesure, contrôle et outils

| Interrupteur | Défaut | Effet | Lot |
|---|---|---|---|
| `STRATA_MTP_CHAIN_CHECK=N` | 0 | compare la chaîne en un graphe et l'ancien chemin sur N tours (double le coût du brouillon) | round-sync |
| `STRATA_PREFILL_TIMING=1` | absent | temps GPU du prompt par phase (dont « dense proj ») et par morceau | prefill-dense |
| `--dump-routing FICHIER` | absent | trace du routage, fenêtres de vérification comprises (tokens validés seulement) | research |
| `--recall-reasoning` / `"recall_reasoning": true` / `STRATA_RECALL_REASONING=1` | absent | le serveur réinjecte la réflexion qu'un client n'a pas renvoyée | server-tools |
| `chat.py --drop-thinking` | absent | `chat.py` ne renvoie plus la réflexion (ancien comportement) | server-tools |
| `make_profile.py --trace-weight W` | 0,5 | poids des traces dans le classement ; 0 = ancienne sortie | server-tools |
| `calibrate.py --spec 4,5,6` / `STRATA_CALIBRATE_SPEC=4,5,6` | absent | mesure aussi la taille de fenêtre | server-tools |
| `mtp_pack.py --q2-search grid\|wide\|exact` | `grid` | recherche d'échelle Q2_0 du brouillon MTP (échelles négatives avec `wide`/`exact`) | server-tools |
| `tools/ab_oneshot.py`, `tools/bench_turns.py` | — | A/B entrelacés du moteur ; conversation multi-tours contre un serveur | server-tools |
| `tools/logits_kl.py`, `tools/routing_locality.py`, `tools/prefill_chunk_check.py` | — | KL entre deux `--dump-logits` ; localité du routage ; comparaison de tailles de morceau | research, correctness |

## 3. Validation sur la RTX 5090

### 3.1 Build

```bash
export PATH=/usr/local/cuda/bin:$PATH
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON \
      -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_GGML_DIR=/opt/llama.cpp
cmake --build build -j
```

### 3.2 Tests CPU (déjà passés sur la machine de développement)

```bash
(cd build && TMPDIR=$(mktemp -d) ctest --output-on-failure -R \
  "platform_memory_test|ple_reader_selftest|suffix_drafter_test|controller_test|draft_policy_test|conv_cache_test|round_sync_test|bf16_bits_test|iq512_test|expert_multi_test|pool_test_synthetic|pool_stress|routing_locality_test|logits_kl_test|gdn_state_bf16_drift_cpu")
for t in serve/test_*.py tools/test_*.py; do STRATA_GGUF_PY=/opt/llama.cpp/gguf-py python3 $t || echo "ÉCHEC $t"; done
```

`pool_test` et `expert_parity` demandent le pack du modèle (`pack/full/experts.bin`) : à lancer sur la machine qui l'a.

### 3.3 Tests de parité GPU, avant tout A/B

Tous doivent afficher `OK`/`PASS` et sortir en 0. Dans l'ordre (les kernels les plus utilisés par la configuration
par défaut d'abord) :

```bash
cd build
ctest --output-on-failure -R "window_batch_parity|s2_expert_grouped_parity|iq_multi_parity|grids_parity|qsa_prep_parity|qsa_select_parity|qsa_decode_attn_parity|mtp_chain_parity|ple_stage_parity|sampler_parity|qsa_indexer_chunk_parity|moe_group_parity|prefill_kernels_parity"
ctest --output-on-failure -R "gdn_fused_parity|fused_gr_parity|qsa_decode_parity|router_top10_parity|bf16_gemv_parity|quantize_act_parity|kv_q4_parity|s_gemv_q8k_parity|rope_parity|elementwise_parity|gdn_parity|s2_grouped_parity|gdn_state_bf16_parity|gdn_state_bf16_parity_old_step|prefill_dense_mmq_parity"
ctest --output-on-failure    # le reste (parités existantes)
./iq_parity && ./native_expert_parity   # hors ctest (native_expert_parity lit le pack natif)
```

Points à lire dans la sortie :
- `grids_parity` : l'en-tête doit dire `clusters yes` et `cluster: ... yes` ; sinon les comparaisons de pas GDN
  n'opposent que l'ancien kernel à lui-même (le repli sur l'ancien kernel est automatique).
- `mtp_chain_parity` : `PASS` ; si le pilote refuse le nœud `WHILE`, le moteur repasse seul à l'ancien chemin.
- `rope_parity` imprime l'erreur de phase réelle du RoPE natif jusqu'à 262 144.
- `gdn_state_bf16_parity --selftest` couvre aussi le pas en cluster sous `--gdn-state-bf16` (voir 4.2) ;
  `gdn_state_bf16_parity_old_step` refait les mêmes vérifications avec `STRATA_OLD_GDN_STEP=1` (un bloc par tête).
- `qsa_indexer_chunk_parity` couvre l'ombre FP16 de `--idx-fp16` (cas « fp16 shadow »).

Sous `compute-sanitizer` (memcheck, puis racecheck et synccheck pour les kernels à mémoire partagée, clusters,
`cp.async` ou barrières réécrites) :

```bash
for t in s2_expert_grouped_parity grids_parity qsa_decode_attn_parity qsa_select_parity prefill_kernels_parity \
         prefill_dense_mmq_parity sampler_parity window_batch_parity qsa_prep_parity moe_group_parity \
         qsa_indexer_chunk_parity; do
  for tool in memcheck racecheck synccheck; do
    compute-sanitizer --tool $tool --error-exitcode 1 ./$t --selftest > san_${t}_$tool.log 2>&1 || echo "ÉCHEC $t $tool"
  done
done
compute-sanitizer --tool memcheck ./mtp_chain_parity && compute-sanitizer --tool memcheck ./ple_stage_parity
```

(Un test qui ne connaît pas `--selftest` ignore l'argument ou le refuse : relancer sans. Les tests « bench » ne sont
pas à passer sous le sanitizer.)

### 3.4 A/B de bout en bout : sortie identique entre le défaut et tous les anciens chemins

`ARGS` = la configuration écrite par `setup.py` (`--pack … --native … --ple-gguf … --expert-profile …
--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp … --max-context …`). Trois prompts tokenisés
`p1.ids p2.ids p3.ids` (court, moyen, long ≥ 16K pour le prefill et la QSA).

```bash
OLD="STRATA_OLD_PLE_STAGING=1 STRATA_PLE_IO_BACKEND=sync \
 STRATA_OLD_MTP_CHAIN=1 STRATA_OLD_COMMIT_SYNC=1 STRATA_OLD_RES_UPLOAD=1 STRATA_OLD_ADAPT_THREAD=1 STRATA_OLD_DMA_FLAG=1 \
 STRATA_OLD_GROUPED=1 STRATA_OLD_IQ_MMVQ=1 STRATA_OLD_WINDOW=1 STRATA_OLD_TOKEN_COMBINE=1 \
 STRATA_OLD_QSA_STEP=1 STRATA_OLD_QSA_PREP=1 STRATA_OLD_ROPE_TABLE=1 STRATA_OLD_SAMPLER=1 \
 STRATA_OLD_TOPK=1 STRATA_OLD_QSA_ATTN=1 STRATA_OLD_GRIDS=1 STRATA_OLD_GDN_AB=1 STRATA_OLD_GDN_STEP=1 \
 STRATA_OLD_RUN_SPLIT=1 STRATA_NO_THP=1 STRATA_OLD_CPU_PREFETCH=1 STRATA_OLD_IQ512=1 \
 STRATA_OLD_PREFILL_GDN_CONV=1 STRATA_OLD_PREFILL_GDN_REC=1 STRATA_OLD_PREFILL_GR=1 \
 STRATA_OLD_IDX_APPEND=1 STRATA_OLD_MOE_GROUP=1 STRATA_PREFILL_COALESCE=0 STRATA_OLD_SFX_RESET=1 \
 STRATA_OLD_CKPT=1 STRATA_OLD_PROFILE_FILL=1"
RUN="./build/strata $ARGS --max-new 256 --greedy --adapt-every 100000 --stats"
for p in p1 p2 p3; do for r in 1 2 3; do
  $RUN --tokens-file $p.ids > new_${p}_$r.log 2> new_${p}_$r.err
  env $OLD $RUN --tokens-file $p.ids > old_${p}_$r.log 2> old_${p}_$r.err
done
  for r in 1 2 3; do
    diff <(grep '^output' new_${p}_$r.log) <(grep '^output' old_${p}_$r.log) > /dev/null \
      && echo "$p run $r : tokens identiques" || echo "$p run $r : DIFFÉRENTS"
  done
done
grep -H "^decode\|^verify window\|^prefill\|ple io" new_*.err old_*.err   # débit (tok/s), attente GPU, E/S PLE
```

Attendu : **les 9 paires ont exactement les mêmes tokens de sortie** (tous ces défauts sont identiques au bit près
par construction). Le débit se compare sur la moyenne des 3 passages de chaque prompt. Si une paire diffère, refaire
l'A/B lot par lot avec les seules variables du lot (tableau 2.1) pour trouver le coupable ; chaque document de lot
donne son A/B détaillé. À refaire aussi avec `--spec-split`, avec `--pcie-mode dma`, avec `--ple-delay-us 3000`
(force l'attente du drapeau P de `ple-io`), et en mode serveur avec `STRATA_STATE_HASH=1` (hash d'état après chaque
requête : points de reprise de `prefill-host`, `commit` asynchrone de `round-sync`) ; les commandes exactes sont dans
[perf/ple-io.md](perf/ple-io.md), [perf/round-sync.md](perf/round-sync.md) et [perf/prefill-host.md](perf/prefill-host.md).

Contrôles complémentaires sur le vrai modèle :
- `STRATA_MTP_CHAIN_CHECK=50` : `chain check: 50 rounds compared, 0 differ` ;
- sampler (le glouton ne passe pas par le nouveau code) : même A/B avec `--temperature 0.8 --seed 1` au lieu de
  `--greedy`, défaut contre `STRATA_OLD_SAMPLER=1` puis `STRATA_SAMPLER_ONE_BLOCK=1` : mêmes tokens ;
- serveur (`server-tools`) : `STRATA_CHECK_PROMPT_IDS=1` ne doit jamais signaler d'écart ; `tools/bench_turns.py`
  contre `STRATA_OLD_DETOK=1 STRATA_OLD_PROMPT_ENCODE=1` pour le temps par tour.

### 3.5 Options qui changent les valeurs, une par une

Chacune contre le défaut, même protocole (3 prompts × 3 passages, glouton, 256 tokens, `--adapt-every 100000`), avec
son contrôle de précision. Ne pas en activer deux à la fois.

| Option | Contrôle de précision | Garder si |
|---|---|---|
| `--idx-fp16` | `qsa_select_parity` ; accord top-1 et KL par `--dump-logits` + `tools/logits_kl.py` sur un prompt ≥ 64K ; aiguilles à 128K | KL négligeable et gain en contexte long |
| `STRATA_QSA_MERGE_GATE=1` | `qsa_decode_attn_parity` doit dire bit à bit ; tokens identiques attendus | tokens identiques |
| `STRATA_QSA_CHUNK=32` | KL (`tools/logits_kl.py`) | gain net sur la phase `qsa attn` et KL négligeable |
| `--prefill-dense-mmq` | `prefill_dense_mmq_parity` (sous memcheck) ; KL forcé sur les logits du prompt ; `STRATA_PREFILL_TIMING=1` pour « dense proj » | KL négligeable et prefill plus rapide |
| `STRATA_PREFILL_F16_SAT=1` | seules les anciennes valeurs ±inf changent : `STRATA_DBG_NAN=1` ne doit plus rien signaler | toujours sûr ; à passer par défaut après vérification |
| `STRATA_PREFILL_ROUTE_DECODE=1` | `tools/prefill_chunk_check.py`, KL prompt | KL plus bas entre prefill et décodage |
| `STRATA_ASYNC_REFILL=1` | TTFT en mode serveur ; tokens peuvent différer (experts sur CPU le temps des copies) | TTFT meilleur, qualité inchangée (KL) |
| `STRATA_IQ_FASTDIV=1` | `iq_multi_parity` ; KL | écart de #4 résorbé sans perte |
| `STRATA_CPU_INT_CORR=1`, `STRATA_IQ512_ONE=1` | `pool_test`, KL | gain CPU mesuré (`pool_test --bench`) sans perte |
| `STRATA_YMISS_WC=1` | tokens identiques (valeurs inchangées) | débit meilleur |
| `STRATA_ROPE_F64=1` | `rope_parity` ; KL forcé et aiguilles à 128K (voir [perf/correctness.md](perf/correctness.md)) | meilleure précision aux grandes positions |
| `STRATA_SPEC_T1=1` (sans `--mtp`) | tokens identiques attendus (T = 1 contre T = 4) | tokens identiques et plus rapide |
| `--gdn-state-bf16` | `gdn_state_bf16_parity` ; `tools/logits_kl.py` sur 4K à 32K tokens ; gain au-delà du bruit | KL négligeable **et** gain mesurable (attendu ≤ 1 %) |
| `calibrate.py --spec 4,5,6` | — (mesure) | la fenêtre la plus rapide est écrite dans la config |
| `mtp_pack.py --q2-search wide` | `s2_grouped_parity` (échelles négatives) ; taux d'acceptation par `tools/ab_oneshot.py --variant` | acceptation meilleure |

## 4. Limites connues et reste à faire

### 4.1 Général

- **Aucune exécution GPU.** Toutes les affirmations « identique au bit près » reposent sur la construction du code,
  l'examen du PTX/SASS de CUDA 13.0 et des émulations sur CPU. Un autre `nvcc` peut contracter une expression
  différemment : la parité GPU (3.3) et l'A/B (3.4) sont le vrai contrôle ; chaque défaut garde son interrupteur.
- Gains : tous **ESTIMÉ** ou **HYPOTHÈSE** (chiffres dans chaque document de lot), sauf les mesures CPU de `cpu` et
  `server-tools`.
- `ple-io` : io_uring par appels système bruts, testé seulement sur Linux 6.18/ext4 (repli automatique sur le pool
  de threads si `io_uring_setup` est refusé) ; la ligne `ple io` se termine maintenant par `backend <nom>`.
- `round-sync` : le `commit` n'est plus attendu dans les boucles de décodage ; tout code ajouté plus tard qui lit
  l'état GDN, l'indexeur ou la PLE depuis un autre flux dans ces boucles doit d'abord appeler `ver.sync_commit()`.
  La ligne `verify window` affiche `commit launch` (lancement seul) au lieu de `commit`.
- `qsa-small` : les gains de #24/#25 ne portent que sur le décodage token par token (sans `--spec`) ; la
  configuration par défaut (`--spec 4 --mtp`) décode par fenêtres.
- `qsa-longctx` : le kernel d'attention préchargé prend 48,6 Ko de mémoire partagée (2 blocs par SM) ; si le prefill
  ou les fenêtres ralentissent, `STRATA_QSA_ATTN_BATCH_OLD=1`.
- `grouped-experts`, `iq-kernels` : plus de registres (occupation un peu plus basse) ; surveiller les petits cas dans
  `--bench`.

### 4.2 Corrections faites à l'intégration

Cinq interactions entre lots, dont trois que la fusion git ne signalait pas du tout (3 à 5) :

1. **`research` × `grids`** (conflit) : le pas GDN en cluster (#26) et l'état BF16 (#53) modifiaient les mêmes
   kernels. Les kernels en cluster sont maintenant des templates sur le type de stockage, comme ceux de `research` :
   `--gdn-state-bf16` garde le lancement en cluster, et l'instanciation FP32 est le kernel de `grids` inchangé.
2. **`research` × `prefill-host`** (conflit) : les points de reprise dans le pool épinglé (#45) copient l'état GDN
   selon la disposition de `research` (les seules parties vivantes avec `--gdn-state-bf16`), sur le flux du pool.
3. **`correctness` × `qsa-small`** : la préparation QSA fusionnée refait l'angle f32 du RoPE natif ; avec
   `STRATA_ROPE_F64=1`, `qsa_prep_supported` répond non et la couche garde les lancements séparés, que l'option atteint.
4. **`qsa-longctx` × `prefill-host`** : l'ajout de l'indexeur par morceau (#40) n'écrivait pas l'ombre FP16 de
   `--idx-fp16` (#21) ; il l'écrit maintenant comme le kernel par cellule (nouveaux cas dans
   `qsa_indexer_chunk_parity`).
5. **`round-sync` × `prefill-host`** : avec `STRATA_ASYNC_REFILL=1`, un téléversement asynchrone de `d_res` (#16)
   pouvait encore être en vol quand le prêt ou le re-remplissage des emplacements réécrit `d_res` de façon
   synchrone ; les deux attendent maintenant ce téléversement.

Et, sans changement de code : `window-batching` et `qsa-longctx` touchent tous deux la porte QSA de la fenêtre (la
porte repliée dans la fusion avec `STRATA_QSA_MERGE_GATE=1` saute la boucle de porte groupée) ; `ple-io` (drapeau P
avant la couche 1) et `window-batching` touchent des parties différentes du graphe de fenêtre ; `prefill-kernels`,
`prefill-host` et `prefill-dense` modifient des parties distinctes de `prefill.cpp` (le minuteur par phase garde
l'index de morceau de `prefill-dense` et le repli partiel de `prefill-host`).

### 4.3 Issues partielles

- **#4** (iq-kernels) : l'interrupteur `STRATA_IQ_FASTDIV=1` existe mais reste désactivé ; l'écart IEEE contre
  fast-math entre les deux quantificateurs Q8_1 subsiste par défaut. À passer par défaut après `iq_multi_parity` et un
  KL.
- **#15** (ple-io) : pas de test au niveau `Verifier` de la poignée de main P (il faut le modèle et un GPU) ;
  l'A/B de [perf/ple-io.md](perf/ple-io.md) doit passer avant de s'y fier. Les soumissions io_uring ne sont pas
  regroupées.
- **#17** (grouped-experts) : le ralentissement possible des petits cas n'a pas pu être mesuré ; d'où le seuil
  `STRATA_GROUPED_PAIR_MIN_HITS`, à régler après `s2_expert_grouped_parity --bench`.
- **#19** (window-batching) : les lots et fusions sûrs sont faits ; PDL (lancement dépendant programmatique) et les
  fusions plus larges proposées par l'issue ne le sont pas.
- **#29** (cpu) : la correction entière des lignes down change les derniers bits : opt-in (`STRATA_CPU_INT_CORR=1`).
- **#33** (server-tools) : message seulement ; déplacer le niveau de réflexion plus loin dans le prompt changerait
  le conditionnement du modèle et demande d'abord une mesure de qualité.
- **#41** (prefill-dense) : seulement la mesure (`STRATA_PREFILL_TIMING=1`, phase « dense proj » et temps par
  morceau) ; le kernel d'attention creuse dédié au prefill n'est pas écrit : décider sur la mesure de la part QSA.

### 4.4 Non traité

- **#54 (S6), prefill couche par couche** : pas commencé. Gain attendu faible en PCIe 5.0 x16, important en x8 ou
  via le chipset ; effort L (~1,3 Go de VRAM pour le résidu à 32K).
- Les évaluations **#51 (S3)** et **#53 (S5)** livrent les outils de mesure et l'option ; la décision (prédiction
  des experts, état BF16 par défaut) attend les mesures sur la 5090.
- `pool_test` et `expert_parity` n'ont pas tourné ici (il faut le pack du modèle).
