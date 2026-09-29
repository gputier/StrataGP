# StrataGP : audit de performance et de précision du moteur

*28 septembre 2026. Base : `upstream/main` au commit `b38c183` (moteur 0.1.18), identique à `gputier/StrataGP`. Les
numéros de ligne renvoient à ce commit.*

> **Mise à jour, moteur 0.1.19 et 0.1.20 (fusionnés depuis Strata).** Déjà corrigés en amont par `a49e810` :
> - **B1** : historique de pénalités, une ligne par token de fenêtre ;
> - **B3** : `top_k` 0 ou > 64 vaut 64, toutes les lignes sont écrites ;
> - l'échantillonnage hors `--serve` (section B11).
>
> Couvert en partie par `setup --calibrate` (`09bff3f`) et la PR #44 : le réglage de `--pcie-frac` et de
> `--spec-min-p` (section 7). Tout le reste de ce rapport s'appliquait toujours à la 0.1.20 ; ce qui en a été fait
> depuis est dans la section suivante. Le texte du rapport, lui, décrit le code de 0.1.18 et n'est pas réécrit.

## État des constats (branche `perf/all`, fusionnée dans `main`)

*Mis à jour après la fusion du moteur amont 0.1.21.* Chaque constat de ce rapport a été ouvert comme issue sur
[gputier/StrataGP](https://github.com/gputier/StrataGP/issues) (#2 à #54, titre préfixé par son code), puis traité
dans l'un des seize lots décrits dans [`docs/PERF-CHANGES.md`](PERF-CHANGES.md) (un document par lot dans
[`docs/perf/`](perf/)). **53 issues, 52 traitées** (45 faites, 7 partielles), 1 non faite. « Fait » veut dire écrit,
compilé et testé sur CPU ; la première exécution sur une RTX 5090, le 29/09/2026, est dans
[`PERF-CHANGES.md`, section 6](PERF-CHANGES.md#6-première-exécution-sur-gpu-rtx-5090-29092026). La dernière colonne de
[`PERF-CHANGES.md`, section 1](PERF-CHANGES.md#1-les-issues-une-par-ligne) donne pour chacun ce qui est actif par
défaut et l'interrupteur exact ; la [section 4.3](PERF-CHANGES.md#43-issues-partielles) dit ce qui manque aux
partiels.

**Corrigé par l'amont lui-même (Niko1221/Strata), donc sans issue ni lot ici :**
- **0.1.19** (`a49e810`) : **B1** (historique de pénalités, une ligne par token de fenêtre), **B3** (`top_k` 0 ou
  supérieur à 64 vaut 64, toutes les lignes écrites) et l'échantillonnage hors `--serve` (section B11, deuxième
  point). `setup --calibrate` (`09bff3f`) règle `--pcie-frac` et `--spec-min-p`, ce qui couvre la piste « re-régler
  `--pcie-frac` » de la section 7 (S2).
- **0.1.21** (`3801f86`) : sa propre correction de CMake pour T1 (`-DSTRATA_BUILD_TESTS=ON` sans le dossier
  `tests/`), qui remplace celle que `perf/base` (`b858f9b`) avait faite ; c'est la version de l'amont qui est gardée.

Restent sans issue, par choix : le balayage CLOCK de `kv_stream.cu` (section B11 : cas impossible aujourd'hui) et le
non-déterminisme du glouton avec le cache adaptatif (déjà documenté par le projet).

| Code | Issue | Constat | État | Lot |
|---|---|---|---|---|
| B2 | [#2](https://github.com/gputier/StrataGP/issues/2) | s_gemv_q8_split_kernel : return avant __syncthreads (comportement indéfini latent) | fait | `correctness` |
| B4 | [#3](https://github.com/gputier/StrataGP/issues/3) | RoPE natif en fast-math : précision de phase aux grandes positions (non testée) | fait | `correctness` |
| B5 | [#4](https://github.com/gputier/StrataGP/issues/4) | Deux quantificateurs Q8_1 compilés différemment (IEEE vs fast-math) | partiel | `iq-kernels` |
| B6 | [#5](https://github.com/gputier/StrataGP/issues/5) | Softplus différent entre le GDN fusionné et le pré-traitement natif | fait | `correctness` |
| B7 | [#6](https://github.com/gputier/StrataGP/issues/6) | Prefill : conversions FP16 sans écrêtage (risque d'inf/NaN) | fait | `prefill-kernels` |
| B8 | [#7](https://github.com/gputier/StrataGP/issues/7) | Routeur : expf précis au prefill, fast-math au décodage ; NaN masqués | fait | `prefill-kernels` |
| B9 | [#8](https://github.com/gputier/StrataGP/issues/8) | Tests de parité manquants sur des kernels de production | fait | `correctness` |
| B10 | [#9](https://github.com/gputier/StrataGP/issues/9) | Étalon de qualité du prefill suspect : KL 0,33 entre morceaux de 6144 et 8192 | fait | `correctness` |
| B11 | [#10](https://github.com/gputier/StrataGP/issues/10) | --spec sans --mtp remplit les fenêtres de token 0 | fait | `correctness` |
| B13 | [#11](https://github.com/gputier/StrataGP/issues/11) | expert_pool_dispatch_multi : tableaux fixes kind[128]/distinct[128]/dma_src[64] sans garde | fait | `cpu` |
| B14 | [#12](https://github.com/gputier/StrataGP/issues/12) | qsa_decode_attn : pas de garde page < 0 si le streaming KV déborde | fait | `qsa-longctx` |
| B15 | [#13](https://github.com/gputier/StrataGP/issues/13) | bf16_from_f32 transforme NaN en -0 | fait | `correctness` |
| T1 | [#14](https://github.com/gputier/StrataGP/issues/14) | Les tests de parité ne compilent pas : add_subdirectory(tests) sur un dossier absent | fait | amont 0.1.21 (`3801f86`), après `perf/base` (`b858f9b`) |
| O1 | [#15](https://github.com/gputier/StrataGP/issues/15) | Sortir les lectures n-gram (PLE) du chemin critique de chaque fenêtre | partiel | `ple-io` |
| O2 | [#16](https://github.com/gputier/StrataGP/issues/16) | Supprimer les allers-retours hôte↔GPU du tour spéculatif | fait | `round-sync` |
| O3 | [#17](https://github.com/gputier/StrataGP/issues/17) | Kernel groupé des experts VRAM : conflits de banques 8-way + somme hx recalculée | partiel | `grouped-experts` |
| O3b | [#18](https://github.com/gputier/StrataGP/issues/18) | native_expert_grouped / iq_mmvq : grilles IQ re-décodées pour chaque colonne | fait | `iq-kernels` |
| O4 | [#19](https://github.com/gputier/StrataGP/issues/19) | Réduire le nombre de nœuds du graphe de fenêtre (lots, fusions, PDL) | partiel | `window-batching` |
| O5 | [#20](https://github.com/gputier/StrataGP/issues/20) | Sampler non glouton : top-k en O(k²·V) sur un SM + fin FP64 calculée par 1024 threads | fait | `sampler` |
| O6 | [#21](https://github.com/gputier/StrataGP/issues/21) | Contexte long : indexeur QSA en FP32 relu à chaque requête | fait | `qsa-longctx` |
| O6b | [#22](https://github.com/gputier/StrataGP/issues/22) | block_topk_kernel : un seul bloc, lectures non coalescées, atomiques disputées | fait | `qsa-longctx` |
| O6c | [#23](https://github.com/gputier/StrataGP/issues/23) | qsa_decode_attn : lectures V sérielles, pas de recouvrement, 66 blocs | fait | `qsa-longctx` |
| O6d | [#24](https://github.com/gputier/StrataGP/issues/24) | Pas/position QSA recopiés depuis l'hôte à chaque couche (22 petits kernels par token) | fait | `qsa-small` |
| O6e | [#25](https://github.com/gputier/StrataGP/issues/25) | QSA : norme, RoPE, FWHT, quantification et ajout KV en lancements séparés | fait | `qsa-small` |
| O7 | [#26](https://github.com/gputier/StrataGP/issues/26) | Grilles calibrées pour 48 SM : sous-remplissage sur RTX 5090 (170 SM) | fait | `grids` |
| O8 | [#27](https://github.com/gputier/StrataGP/issues/27) | Pool CPU : run_split déséquilibré (20 tâches sur 6 threads) | fait | `cpu` |
| O8b | [#28](https://github.com/gputier/StrataGP/issues/28) | Arène d'experts Linux en pages de 4 Ko sans MADV_HUGEPAGE | fait | `cpu` |
| O8c | [#29](https://github.com/gputier/StrataGP/issues/29) | Noyau CPU Q2_0 : prélecture des échelles et surcoût par ligne de la projection down | partiel | `cpu` |
| O8d | [#30](https://github.com/gputier/StrataGP/issues/30) | Noyaux CPU i-quant limités par le calcul (~5 Go/s par cœur) | fait | `cpu` |
| O9 | [#31](https://github.com/gputier/StrataGP/issues/31) | Serveur : détokenisation quadratique (MESURÉ 2,2 ms/token à 16K) | fait | `server-tools` |
| O9b | [#32](https://github.com/gputier/StrataGP/issues/32) | Serveur : conversation entière re-tokenisée à chaque requête (BPE Python sans cache) | fait | `server-tools` |
| O9c | [#33](https://github.com/gputier/StrataGP/issues/33) | Changer le niveau de réflexion relit toute la conversation | partiel | `server-tools` |
| O9d | [#34](https://github.com/gputier/StrataGP/issues/34) | chat.py et clients OpenAI : la réutilisation « live » échoue (réflexion non renvoyée) | fait | `server-tools` |
| O9e | [#35](https://github.com/gputier/StrataGP/issues/35) | Images WebP/TIFF re-normalisées à chaque tour | fait | `server-tools` |
| P1 | [#36](https://github.com/gputier/StrataGP/issues/36) | Prefill MoE : synchronisation hôte par couche + ~6 appels CUDA par expert | fait | `prefill-host` |
| P2 | [#37](https://github.com/gputier/StrataGP/issues/37) | Prefill gdn_conv : filtre FIR exécuté comme une récurrence série sur 8192 tokens | fait | `prefill-kernels` |
| P3 | [#38](https://github.com/gputier/StrataGP/issues/38) | Prefill gdn_rec_kernel : 48 blocs, ~5 barrières par token | fait | `prefill-kernels` |
| P4 | [#39](https://github.com/gputier/StrataGP/issues/39) | Prefill : projections denses en FP16 cuBLAS au lieu de MMQ int8 | fait | `prefill-dense` |
| P5 | [#40](https://github.com/gputier/StrataGP/issues/40) | Prefill : native_qsa_indexer_append lancé une fois par token (~98 000 lancements par morceau) | fait | `prefill-host` |
| P6 | [#41](https://github.com/gputier/StrataGP/issues/41) | Prefill : attention QSA faite par le kernel de décodage, par lots de 32 requêtes | partiel | `prefill-dense` |
| P7 | [#42](https://github.com/gputier/StrataGP/issues/42) | Re-remplissage bloquant des emplacements prêtés après chaque prompt | fait | `prefill-host` |
| P8 | [#43](https://github.com/gputier/StrataGP/issues/43) | Prefill : chaîne GR élément par élément (~290 Go de trafic par morceau) | fait | `prefill-kernels` |
| E1 | [#44](https://github.com/gputier/StrataGP/issues/44) | Combinaison MoE : k lignes relues sur PCIe dont les lignes GPU nulles, 3 kernels | fait | `window-batching` |
| E2 | [#45](https://github.com/gputier/StrataGP/issues/45) | Points de reprise de conversation : copies synchrones vers des vecteurs paginables | fait | `prefill-host` |
| E3 | [#46](https://github.com/gputier/StrataGP/issues/46) | Chercheur de suffixe reconstruit de zéro à chaque requête | fait | `prefill-host` |
| E4 | [#47](https://github.com/gputier/StrataGP/issues/47) | Table RoPE FP64 allouée (64 Mio à 262K) mais inutilisée en mode --native | fait | `qsa-small` |
| E5 | [#48](https://github.com/gputier/StrataGP/issues/48) | Démarrage : verify_slot relit chaque emplacement du profil en entier | fait | `prefill-host` |
| E6 | [#49](https://github.com/gputier/StrataGP/issues/49) | make_profile.py : les traces ne changent rien au profil livré | fait | `server-tools` |
| S1 | [#50](https://github.com/gputier/StrataGP/issues/50) | Mesurer : fenêtre spéculative plus longue (--spec 5/6) sur RTX 5090 | fait | `server-tools` |
| S3 | [#51](https://github.com/gputier/StrataGP/issues/51) | Évaluer : prédire les experts de la couche suivante pour lancer leurs copies en avance | fait | `research` |
| S4 | [#52](https://github.com/gputier/StrataGP/issues/52) | Évaluer : meilleure quantification du brouillon MTP (taux d'acceptation) | fait | `server-tools` |
| S5 | [#53](https://github.com/gputier/StrataGP/issues/53) | Évaluer : état GDN en BF16 (moitié moins de trafic) | fait | `research` |
| S6 | [#54](https://github.com/gputier/StrataGP/issues/54) | Évaluer : prefill couche par couche (chaque expert diffusé une fois par prompt) | non fait | — |

## Ce qu'il faut retenir

- **Sur votre RTX 5090, les priorités s'inversent par rapport à la carte de référence (RTX 5070).** Avec 32 Go, environ
  16 000 à 17 500 des 24 576 experts tiennent en VRAM (estimé). Le calcul des experts sur le CPU, qui occupe 40 % d'un
  tour de décodage sur la 5070, devient marginal. Ce qui domine alors : le temps fixe de chaque tour côté hôte
  (lectures SSD, synchronisations, brouillon MTP : 6,3 ms mesurés sur 34,3 ms sur la 5070) et l'efficacité des kernels
  GPU des experts en cache.
- **Trois changements à faire en premier** (section 6) :
  1. Sortir les lectures n-gram (PLE) du chemin critique de chaque fenêtre.
  2. Corriger le kernel groupé des experts en VRAM : conflits de banques mémoire, et une somme recalculée à chaque
     ligne alors qu'elle ne dépend que du token.
  3. Supprimer les allers-retours hôte↔GPU du tour spéculatif.

  Les deux premiers donnent un résultat identique au bit près.
- **Un bug de correction à corriger avant tout** (section 5, B1) : avec des pénalités de répétition, les lignes 1 et
  suivantes d'une fenêtre de vérification lisent un historique non initialisé, voire hors du tampon.
- **Rien n'a été mesuré sur GPU** : ce conteneur n'en a pas. Les gains sont des estimations, et chaque piste indique
  le test de parité ou le benchmark qui la validera.

---

## 0. Méthode et limites

**Ce qui a été lu, en entier :**
- les 204 fichiers suivis hors `third_party/` (53 000 lignes de C++, CUDA et Python) ;
- `README.md`, `docs/DETAILS.md`, l'article `docs/paper/Strata-Paper.pdf` (9 pages) et `bench/results/*` ;
- les 98 commits de `upstream/main` et les diffs des 32 PR récupérées par `refs/pull/*/head`.

J'ai relu une seconde fois les passages de code à l'origine de
chaque constat majeur, notamment `generate.cpp:2340-2352` et `3038-3058`, `sampler.cu:90-305`, `verify.cpp:752-860`,
`s2_expert_grouped.cu:515-600`, `qsa_select.cu:24-51`, `pool.cpp:430-445`, `pinned.cu:76-92`,
`direct_file.cpp:186-230`, `ngram.cpp:277-290`, `prefill/kernels.cu:118-135`, `prefill.cpp:1104-1112`,
`gemm.cu:88-113`, `native_rope.cu:48-58` et `mtp.cpp:568-592`.

**Ce qui n'a pas pu être lu :**
- **Descriptions, commentaires et reviews des PR, et les issues.** L'API et les pages GitHub sont refusées à cette
  session pour un dépôt qui ne vous appartient pas. Seuls les commits et les diffs des PR sont disponibles.
- **Des fichiers cités par le code mais absents de l'arbre publié** : `bench/micro/*` (les tests « oracle » contre
  llama.cpp, dont `native_mmvq_multi`, `moe_hit_parity` et `act_quant_parity`), `bench/prompts/*`, `Memory/LEDGER.md`
  et plusieurs dossiers de résultats (`2026-09-27-spec`, `2026-09-23-ngram-io`, etc.).
- **Le dossier `tests/`** : `CMakeLists.txt:593-596` l'inclut sans qu'il existe. La configuration échoue donc avec
  `-DSTRATA_BUILD_TESTS=ON`, alors que les 35 `add_test` de parité sont déclarés dans le CMakeLists principal. Il
  faut entourer `add_subdirectory(tests)` d'un `if(EXISTS ...)` pour pouvoir compiler les tests de parité.

**Compilation :** le moteur compile pour `sm_120` avec CUDA 13.0 dans le conteneur, en 2 min 30 et sans erreur. Il
n'y a pas de GPU pour l'exécuter.

**Étiquettes utilisées :**
- **MESURÉ** : un chiffre qui figure dans le dépôt (article, `bench/results`, message de commit ou commentaire de code).
- **ESTIMÉ** : un calcul fait pour ce rapport.
- **HYPOTHÈSE** : un mécanisme plausible qu'il faut profiler pour le confirmer.

---

## 1. Carte de l'architecture

### 1.1 Le modèle et le placement des données

Qwen3.8-Flash-Next compte 48 couches : 36 Gated DeltaNet (GDN) et 12 d'attention clairsemée Qwen Sparse Attention
(QSA, une couche sur quatre). Les autres dimensions :
- `n_embd` 2 560 et un résidu en hyper-connexions (« GR », 4 flux × 2 560) ;
- MoE de 512 experts avec `n_ff` 640, top-10, plus un expert partagé ;
- une table n-gram (« PLE », 28,8 Go) adressée par les trois derniers tokens (`layout.hpp:28-53`, `qsa.hpp:94-112`).

Un expert Q2_0 fait 1 382 400 octets (`expert.hpp:34-51`, `pack_layer.py:40-51`).

| Où | Quoi |
|---|---|
| VRAM | poids denses (~3,5 Go ; 4,52 Gio sous forme moteur, `weights.hpp:17`), tête de sortie Q6_K (~0,52 Go lus par passe), couche MTP (~0,9 Go, tous ses experts en VRAM), états GDN (36 × 3 Mio en FP32), cache KV (FP16, INT8 au-delà de 8K, Q4_0 en option), clés d'indexeur FP32, cache d'experts (tout le reste) |
| RAM épinglée | les 24 576 experts (34 Go en Q2_0), `mmap` avec `MAP_HUGETLB` sinon pages de 4 Ko, puis `cudaHostRegister` (`pinned.cu:76-92, 132-197`) |
| SSD | table PLE, lue ligne par ligne en `O_DIRECT` avec un cache de lignes en RAM (`ple_reader.cpp`, `direct_file.cpp`) |

### 1.2 Décodage en production : la fenêtre de vérification

C'est le chemin de presque tous les tokens générés. Configuration écrite par `setup.py:1303-1324` :
`--native ... --spec 4 --spec-min-p 0.5 --mtp ...`. Le premier token après le prompt passe par le chemin « token »
(section 1.3). Tous les suivants passent par des fenêtres de T ≤ 4 tokens : le dernier token accepté plus jusqu'à 3
brouillons. T peut monter à 6 quand la recherche de suffixe propose un brouillon (`generate.cpp:990-993`).

**Un tour** (`generate.cpp:3685-3780` hors serveur, `3015-3106` en serveur) :

1. **Choisir T et les brouillons (CPU).** Brouillons MTP ou recherche de suffixe, arbitrés par `DraftPolicy`
   (`draft_policy.cpp:55-82`).
2. **`apply_pending` (hôte→GPU).** Si une vague d'échanges du cache adaptatif est arrivée, `cudaMemcpy` **synchrone**
   de la table de résidence `d_res` (98 Ko, `generate.cpp:3598-3606`).
3. **Historique de pénalités, si la requête en a (hôte→GPU).** `cudaMemcpy` synchrone (`generate.cpp:3043-3055`).
4. **`Verifier::run`, préparation (CPU puis SSD).**
   - Calcul des T×16 lignes n-gram, puis **`gather_batch`, qui bloque jusqu'à ce que toutes les lignes soient lues
     sur le SSD** (`verify.cpp:760-769`, `ngram.cpp:277-290`).
   - Ce n'est qu'ensuite que `cudaGraphLaunch` lance la fenêtre (`verify.cpp:781`). La PLE n'est pourtant consommée
     qu'à la couche 1 (`verify.cpp:358`).
5. **Le graphe de fenêtre, 48 couches.** Pour chaque couche l (`verify.cpp:286-645`) :
   - **Partie `pre` (GPU).** Lecture GR fusionnée, mixeur (GDN multi-token ou QSA avec ajout KV, indexeur, top-k de
     blocs, attention split-K), écriture GR, lecture GR côté FFN.
   - **Routeur (GPU).** Une GEMV BF16 plus un top-10 **par token**, en boucle (`verify.cpp:512-516`).
   - **Sonnette (GPU vers mémoire hôte mappée).** Écriture des identifiants d'experts, avec un compteur de séquence
     (`verify.cpp:517`).
   - **Expert partagé (GPU).**
   - **Répartition (CPU).** Le thread hôte, qui scrute la sonnette, classe les experts en trois groupes
     (`expert_source.cpp:251-438`) : cache VRAM ; part PCIe (`pcie_frac`, 0,2 pour Q2_0 et 0,55 pour les packs
     natifs, `generate.cpp:1087`) ; CPU.
   - **Partie `post`.**
     - Le GPU attend le drapeau A (le plan), calcule les experts en VRAM (`moe_grouped_s2`, `verify.cpp:574`) ;
     - attend le drapeau B (experts copiés par PCIe), calcule ceux-là ;
     - attend le drapeau CPU, puis `copy_from_mapped` des sorties CPU (lecture directe sur PCIe), `moe_hit_add` et
       une combinaison par token.
   - **Pendant ce temps, côté CPU.** 5 ou 6 threads calculent les experts manqués directement dans la RAM
     (`pool.cpp`, AVX-512 VNNI). L'hôte lève le drapeau CPU une fois fini.
6. **Tête et échantillonnage (GPU).** La tête multi-colonnes et l'argmax glouton sont dans le graphe. Pour une
   requête échantillonnée ou pénalisée, `sample_tokens` est relancé hors du graphe (`verify.cpp:843-856`).
7. **Fin de fenêtre (synchronisation).** `cudaStreamSynchronize(cs_)` puis `cudaStreamSynchronize(copy_)`
   (`verify.cpp:839-842`).
8. **Acceptation (CPU).** Plus long préfixe correct (`generate.cpp:3731`).
9. **`ver.commit`.** Rejoue les tokens gardés dans l'état GDN, puis `cudaStreamSynchronize` (`verify.cpp:943-945`).
10. **`mtp.draft`.** Un graphe « round » puis `cudaStreamSynchronize`, puis **un graphe et une synchronisation par
    brouillon supplémentaire**, pour tester `p >= min_p` sur l'hôte (`mtp.cpp:573, 586`).
11. **Cache adaptatif, tous les 4 tours.** Un `std::thread` est créé ; il classe l'usage et lance jusqu'à 96 copies
    asynchrones sur un flux séparé (`generate.cpp:3610-3656, 3739-3742`).

**Synchronisations hôte↔GPU par tour :** au moins 4 (fenêtre, copies, commit, brouillon), plus 1 par brouillon
au-delà du premier, plus 1 en échantillonnage, plus les `cudaMemcpy` synchrones des étapes 2 et 3. Par couche, la
synchronisation passe par la mémoire mappée (sonnette et drapeaux), sans appel au pilote.

### 1.3 Chemin « token » (premier token, ou sans `--spec`)

- Un graphe capturé couvre les 48 couches (`session.cpp:798-870`). Par couche, les experts VRAM tournent
  (`moe_hit_select`, `quantize_q8_0_scaled`, `moe_hit_grouped_s2_dev`), puis le GPU attend le CPU dans un kernel qui
  scrute un drapeau (`doorbell_wait`), puis `copy_from_mapped` et `post`.
- Côté hôte : une scrutation par couche (`session.cpp:896`), les experts CPU, un `sfence` puis le drapeau.
- Une seule `cudaStreamSynchronize` par token (`session.cpp:927`). La tête et l'échantillonnage sont hors du graphe.
- Sur ce chemin, la PLE est lue avant les couches (`generate.cpp:3345-3363`). **Il n'y a pas de part PCIe** : tous les
  manqués vont au CPU.

### 1.4 Prefill (prompts longs)

**Découpage.** `--prefill auto` choisit des morceaux de 8 192 tokens (`prefill.cpp:752`), en empruntant jusqu'à 90 %
des emplacements du cache d'experts (`generate.cpp:2272-2295`).

**Pour chaque morceau, chaque couche, en deux moitiés :**
- **Lecture GR.** GEMM BF16 cuBLAS.
- **Moitié 0, GDN.**
  - Les projections natives sont déquantifiées en FP16 puis multipliées par une GEMM FP16 cuBLAS
    (`gemm.cu:96-112`).
  - `gdn_conv` : un thread par canal qui **parcourt les 8 192 tokens en série** (`prefill/kernels.cu:121-133`).
  - `gdn_recurrence` : 48 blocs (`kernels.cu:457`).
- **Moitié 0, QSA.** Projections ; `native_qsa_indexer_append` **lancé une fois par token** (`prefill.cpp:992-999`) ;
  attention avec le kernel de décodage, par lots de 32 requêtes (`prefill.cpp:1077-1082`).
- **Moitié 1, MoE.**
  - Routeur, expert partagé, puis **copie des ids vers l'hôte et `cudaStreamSynchronize`** (`prefill.cpp:1108-1109`),
    puis un tri par expert sur l'hôte.
  - Tous les experts sont calculés sur le GPU, avec les kernels MMQ int8 de llama.cpp.
  - Les experts non résidents sont diffusés en anneau (`stream_all`, `prefill.cpp:802-826`) : **tous** les experts
    non résidents de toutes les couches, dans un ordre fixe, sur un flux de copie séparé. Leur transfert recouvre
    l'attention de la couche précédente.
  - Pour chaque expert : un événement, une copie device→device vers un tampon de groupe, puis MMQ.

**Après le prompt.** Les emplacements prêtés sont re-remplis **un par un, de façon bloquante**
(`generate.cpp:3293-3307`).

---

## 2. Modèle de performance

### 2.1 Décodage, machine de référence (RTX 5070 12 Go, Ryzen 5 7600, DDR5-5200)

**Mesuré (article, tableau 5, Q2_0 à 4K) :**

| Tour de fenêtre | GPU | Experts CPU | Brouillon | Autre | Total | Tokens par tour |
|---|---:|---:|---:|---:|---:|---:|
| ms | 14,6 | 13,4 | 2,0 | 4,3 | 34,3 | 3,23 |

Le taux de hits VRAM est de 0,72 ; le CPU lit 41 Go/s d'experts.

**Octets lus par tour (T ≈ 4), estimé :**

| Poste | Octets | Bande passante | Plancher |
|---|---:|---|---:|
| Poids denses (lus une fois par fenêtre) | ~3,5 Go | VRAM 672 Go/s | 5,2 ms |
| Experts en VRAM (~1 500 distincts, 72 % en VRAM) | ~1,5 Go | VRAM | 2,2 ms |
| Experts sur CPU (28 %) | ~0,58 Go | DDR5, 41 Go/s mesurés en concurrence, 52 seul | 10,6–14 ms |
| États GDN (lecture et écriture, plus la réécriture du commit) | ~0,45 Go | VRAM | 0,7 ms |
| KV et indexeur à 4K | < 10 Mo | VRAM | ~0 |
| Lignes PLE | T × 16 lignes (cache RAM, puis SSD) | NVMe, ~80 µs par lecture 4K à QD1 | 0–2 ms (voir O1) |

**Ce que cela montre :**
- **GPU.** Le plancher est d'environ 7,4 ms, pour 14,6 ms mesurés : environ 50 % d'efficacité. L'écart vient des
  latences et des lancements : environ 1 000 nœuds de graphe par fenêtre, beaucoup de petits kernels et deux attentes
  de drapeau par couche.
- **CPU.** Il tourne à 80 % de la bande passante DDR5 disponible en concurrence ; il est presque au plafond.
- **Temps fixe.** « Brouillon + autre » font 6,3 ms (18 %) de temps hôte : synchronisations, lectures PLE, commit.

### 2.2 Décodage, projection pour votre RTX 5090 32 Go (ESTIMÉ)

**Données de la carte :** 1 792 Go/s, 170 SM, 96 Mo de L2, PCIe 5.0 x16.

**Place pour les experts.** VRAM libre ≈ 32 − 4,5 (dense) − 0,9 (MTP) − 1,7 (KV INT8 à 128K, ou moins avec
`--kv-resident`) − 0,7 (réserve) − ~1 (tampons et affichage) ≈ 23 Go. Cela fait environ 16 000 à 17 500 experts Q2_0
sur 24 576, soit 65 à 70 %. L'article compte environ 700 experts par Go.

**Hits.** La courbe de l'article (figure 3) avec le cache adaptatif donne un taux de hits estimé entre 0,90 et 0,95.

**Par tour (T ≈ 4) :**
- **GPU.** (3,5 + ~1,9) Go / 1 792 Go/s ≈ 3,0 ms de plancher. Il s'y ajoute les latences et lancements, qui ne
  diminuent pas avec la bande passante : environ 5 à 6 ms. Total : **8 à 9 ms**.
- **CPU.** ~100 experts manqués, 0,14 Go à 45 Go/s ≈ 3 ms, **en recouvrement** avec le GPU couche par couche. Ce n'est
  plus le chemin critique, sauf sur les couches où les manqués s'accumulent.
- **Temps fixe.** Brouillon et autre, soit ~6 ms si rien ne change.
- **Total :** ~15 à 16 ms pour ~3,2 tokens, soit **~200 tokens/s à 4K** (l'article estimait 128 à 140 pour une 3090).

**Conséquence : le temps fixe hôte représente environ 40 % du tour sur une 5090.** Les optimisations O1 à O3 pèsent
donc deux à trois fois plus que sur la 5070, et celles du CPU (O8) presque rien.

**Remplissage des 170 SM (HYPOTHÈSE).** Plusieurs grilles ont été calibrées pour les 48 SM de la 5070 et
sous-remplissent une 5090 :
- `gdn_ab_kernel` : 12 blocs (`fused_gdn.cu:144`) ;
- le pas GDN : 48 blocs (un par tête) ;
- `qsa_decode_attn` : 66 blocs ;
- prefill `gdn_rec_kernel` : 48 blocs (`prefill/kernels.cu:457`) ;
- `fetch_blobs` et `gather_rows` : codés en dur à `48 * 8` (`verify_kernels.cu:310, 341`).

### 2.3 Prefill, 32K tokens (mesuré : 1 308 tok/s en Q2_0 sur la 5070, soit ~6,35 s par morceau de 8 192)

**Calcul par morceau (ESTIMÉ) :**
- dense : 2 × 3,6 G paramètres × 8 192 ≈ 59 TFLOP, en FP16 avec accumulation FP32 (~60 TFLOPS sur la 5070) : ≥ 1 s ;
- experts : ~39 TFLOP en int8 MMQ : 0,3 à 0,5 s.

**PCIe :** ~31 à 33 Go d'experts non résidents par morceau. Sur 6,35 s, cela fait ~5 Go/s : **le PCIe n'est pas la
limite**. Passer l'anneau de 96 à 384 emplacements (1 153 → 1 294, mesuré) montre que c'est la latence aux frontières
de couche qui comptait.

**Le reste, environ 3 s par morceau (HYPOTHÈSE),** se répartit entre :
- les kernels séquentiels sur T (`gdn_conv`, `gdn_rec`) ;
- 48 synchronisations hôte par morceau et ~3 000 appels d'API CUDA par couche MoE ;
- 98 000 lancements de l'indexeur par morceau ;
- l'attention QSA exécutée par le kernel de décodage.

**Sur une 5090,** le calcul dense et le PCIe s'accélèrent d'environ 3× et 2×, mais pas les parties séquentielles.
Celles-ci pèseront donc davantage. **Première action : lancer `STRATA_PREFILL_TIMING=1`** pour obtenir la répartition
réelle (`prefill.cpp:665-669`).

---

## 3. Optimisations de vitesse, classées par gain de bout en bout attendu

Deux estimations sont données : sur la 5070 de référence (tour de 34 ms) et sur votre 5090 (tour projeté de ~16 ms).

### O1. Sortir les lectures n-gram (PLE) du chemin critique de chaque fenêtre

**1. Fichiers :** `src/core/verify.cpp:760-769`, `src/kernels/ngram.cpp:277-290`,
`src/platform/direct_file.cpp:219-230`, `src/ngram/ple_reader.cpp:206-228`.

**2. Problème :** chaque fenêtre lit ses T×16 lignes PLE et attend la fin de toutes les lectures **avant** de lancer
le GPU :
```cpp
if (!ss.ple.table->gather_batch(rows, (size_t) T, h_ple_, err)) return false;   // verify.cpp:768, bloquant
...
const cudaError_t le = cudaGraphLaunch(exec_[T], cs_);                         // verify.cpp:781
```
La PLE n'est pourtant utilisée qu'à la couche 1 (`verify.cpp:358`, `if (l == 1 && ple_on)`).

Sous Linux s'ajoute un second problème : les lectures sont **séquentielles**. `submit` appelle `pread` directement
(`direct_file.cpp:224`), `wake()` est vide et `--ple-inflight 64` n'a aucun effet. Le commentaire du fichier le dit :
« Phase L replaces this with io_uring ».

**3. Changement :**
- (a) **Lancer le graphe d'abord.** Copier `h_ple_` à la couche 1 derrière un drapeau mappé, avec le même mécanisme que
  la sonnette (`wait_flag_ge`) :
  ```cpp
  // verify.cpp, record_window : avant ple_block à l = 1
  if (l == 1 && ple_on) { wait_flag_ge(m_flagP_, 1u, cs); copy_from_mapped(ple_, m_ple_, T * N, cs); ... }
  // run() : lancer exec_[T] AVANT gather_batch, puis
  if (!ss.ple.table->gather_batch(rows, T, h_ple_, err)) return false;
  std::atomic_thread_fence(std::memory_order_seq_cst); *(volatile uint32_t*) h_flagP_ = 1;
  ```
- (b) **Sous Linux, paralléliser les lectures.** Utiliser io_uring, ou à défaut un petit pool de threads `pread` à
  profondeur de file 16, et sortir `pread` de la section tenue par `mu`.

**4. Gain (ESTIMÉ) :**
- La couche 0 dure ~0,3 à 0,6 ms de GPU. C'est ce que la lecture peut recouvrir.
- Sous Linux, 64 lignes par fenêtre, avec 20 à 50 % d'échecs de cache à ~80 µs chacun en série, coûtent **1 à 2,5 ms
  par tour** : **3 à 7 % sur la 5070, 6 à 15 % sur la 5090**.
- Sous Windows (IOCP déjà parallèle), le recouvrement seul gagne 0,1 à 0,3 ms (**1 à 2 %**).
- Le moteur affiche déjà la mesure de base : `PleTable::io_report` (`ngram.cpp:294-306`) imprime `blocked ... ms` et la
  latence p50/p99.

**5. Risque de précision :** aucun. Les mêmes octets arrivent avant leur utilisation. Pour valider : comparer le hash
d'état GDN (`STRATA_STATE_HASH`) et les tokens générés en glouton avant et après, et relire le `blocked` de la ligne
`ple io`.

**6. Effort :** M. **GPU :** toutes les architectures ; le gain est le plus grand sous Linux.

### O2. Supprimer les allers-retours hôte↔GPU du tour spéculatif

**1. Fichiers :** `src/core/mtp.cpp:573, 586` ; `src/core/verify.cpp:839-842, 921-927, 943-945` ;
`src/program/generate.cpp:3598-3606, 3739-3742, 3054`.

**2. Problème :** 18 % du tour sur la 5070, et ~40 % sur la 5090, est du temps hôte (tableau 5 : brouillon 2,0 ms et
autre 4,3 ms). Sources relevées :
- **Brouillon MTP :** un graphe et une synchronisation par brouillon, uniquement pour tester `pj >= min_p` sur l'hôte :
  ```cpp
  if (cudaGraphLaunch(step_exec_[j], cs_) != cudaSuccess || cudaStreamSynchronize(cs_) != cudaSuccess) {   // mtp.cpp:586
  ```
- **`commit` :** suivi d'une `cudaStreamSynchronize` (`verify.cpp:945`), alors que la fenêtre suivante est ordonnée
  sur le même flux.
- **Drapeau B, seulement avec `--pcie-mode dma` :** depuis le commit e6265c7, le mode par défaut est le kernel de
  copie. En mode DMA, le drapeau est levé par `cudaLaunchHostFunc` (`verify.cpp:927`), qui s'exécute sur un thread du
  pilote avec des dizaines de µs de latence, sur chaque couche qui a des copies PCIe.
- **Table de résidence :** `apply_pending` fait un `cudaMemcpy` synchrone de `d_res` (`generate.cpp:3604`). Or, pendant
  les fenêtres, la répartition lit `host_res` côté hôte (`expert_source.cpp:224`) : cet envoi est probablement inutile
  dans la boucle spéculative (HYPOTHÈSE).
- **Cache adaptatif :** un `std::thread` créé et détruit tous les 4 tours (`generate.cpp:3742`).

**3. Changement :**
- **Chaîne MTP en un seul graphe.** Capturer toute la chaîne avec un nœud conditionnel `WHILE`
  (`cudaGraphConditionalHandle`, CUDA 12.4+, disponible en 13.0). Un petit kernel positionne la condition à partir de
  `probs_[j] >= min_p && j < max_drafts`. Variante plus simple : toujours calculer les 3 pas et tronquer sur l'hôte ;
  un pas coûte moins qu'un aller-retour.
- **`commit` :** supprimer la synchronisation. Si le MTP doit attendre le commit, un `cudaEventRecord` sur le flux de
  vérification suivi d'un `cudaStreamWaitEvent` sur le flux MTP suffit.
- **Drapeau B :** remplacer `cudaLaunchHostFunc` par `cuStreamWriteValue32(copy_, (CUdeviceptr) m_flagB_, want, 0)`.
  Le mettre en file même quand `n <= 0`, pour que la valeur reste monotone.
- **Table de résidence :** passer `apply_pending` en `cudaMemcpyAsync` depuis un tampon épinglé, ou le supprimer
  pendant les fenêtres après vérification.
- **Cache adaptatif :** un thread persistant réveillé par une variable de condition.

**4. Gain (ESTIMÉ) :**
- Sous Linux, 20 à 60 µs par synchronisation et 20 à 50 µs par fonction hôte sur les couches qui ont du PCIe.
- Sous Windows (WDDM), 0,3 à 0,4 ms par lancement ; mesuré, `session.hpp:363`.
- Total : 1 à 2,5 ms par tour, soit **3 à 7 % sur la 5070 et 6 à 15 % sur la 5090** (davantage sous Windows).

**5. Risque de précision :** aucun ; l'ordre des calculs est inchangé. Pour valider : génération gloutonne identique
token par token (256 tokens, trois prompts), avec le test des brouillons forcés faux que cite l'article (§3.3).

**6. Effort :** M. **GPU :** toutes (les nœuds conditionnels exigent CUDA 12.4+).

### O3. Kernel groupé des experts en VRAM : conflits de banques et somme recalculée

**1. Fichiers :** `src/kernels/cuda/s2_expert_grouped.cu:520-534` (`chunk_dot`), `545-560` (placement en mémoire
partagée), `586` (lecture) ; même motif dans `down_grouped_kernel` (`607, 641`). Ce kernel est appelé par
`verify.cpp:574` et `mtp.cpp:398` : **c'est le calcul des experts en VRAM de chaque fenêtre** pour les packs Q2_0
canoniques.

**2. Problème :**
```cpp
__shared__ int xs_q[GMAX][H / 4];
xs_q[k][c * 8 + w] = v;                                           // écriture, :560
acc += chunk_dot(cb[q], &xs_q[k][c * 8], dw[q], xs_d[k][c]);     // lecture, c = lane + 32q, :586
...
s = __dp4a(cw, xw[j], s);  hx = __dp4a(ones, xw[j], hx);          // chunk_dot, :529-530
```
- **Conflits de banques.** Pour un j donné, la voie l lit le mot `8l + j` : les voies l, l+4, l+8… tombent sur la
  même banque. C'est un **conflit d'ordre 8** sur chaque lecture d'activation. Le PTX montre 48 `ld.shared.u32`
  scalaires.
- **Somme recalculée.** `hx` est la somme des 32 activations int8 d'un morceau. Elle ne dépend que du token, pas de la
  ligne de poids, mais elle est recalculée avec 8 `dp4a` supplémentaires pour **chaque ligne**. Cela double le travail
  entier.
- **Ordre de grandeur (ESTIMÉ).** Par ligne et par entrée : ~190 cycles de mémoire partagée, contre ~130 cycles de
  DRAM sur la 5070 (720 o par ligne, 14 Go/s par SM) et ~170 sur la 5090. Le kernel est limité par la mémoire
  partagée, pas par la DRAM.

**3. Changement (résultat identique au bit près : mêmes entiers, même expression flottante) :**
```cpp
__shared__ int xs_q[GMAX][8][H / 32 + 1];   // [entrée][mot du morceau][morceau], +1 de bourrage
__shared__ int xs_h[GMAX][H / 32];          // hx précalculé une fois par (entrée, morceau)
// placement :
xs_q[k][w][c] = v;
if (w == 7) xs_h[k][c] = hx_of_chunk;       // somme des 32 int8, faite une fois
// chunk_dot : s = __dp4a(cw, xs_q[k][j][c], s);  ...  return dw * dx * (float) (s - xs_h[k][c]);
```
- Autre option : charger les codes avec la table `__byte_perm` de `native_mmvq.cu:273-276`, qui donne directement
  `code - 1`. Le `hx` disparaît alors complètement.
- Le kernel du chemin « token », `gu_kernel`/`down_kernel` (`s2_expert_grouped.cu:65-96`), fait 44 lectures d'un
  octet par itération. Il peut lire ses codes en `uint2` et ses activations depuis un plan `qs` aligné.

**4. Gain (ESTIMÉ) :**
- Le kernel serait 1,5 à 2× plus rapide.
- Sur la 5070, la part « experts VRAM » du temps GPU vaut ~3 à 5 ms par tour ; le gain est de 1 à 2,5 ms, soit
  **3 à 7 %**.
- Sur la 5090, où ~93 % des experts sont en VRAM, **4 à 8 %**.

**5. Risque de précision :** nul ; le résultat est identique au bit près. Pour valider : `native_expert_parity` et
`expert_parity` doivent rester bit à bit ; `moe_hit_parity` est absent de l'arbre (voir section 0).

**6. Effort :** S. **GPU :** toutes.

**Ce n'est pas l'idée déjà rejetée.** `s2_gemv_fast.cu:1-18` a mesuré « pire » le fait de placer x en mémoire
partagée dans la GEMV S2. Ici, x est déjà en mémoire partagée : seule la disposition change, pour supprimer les
conflits.

**Packs natifs (IQ).** Ils passent par `native_expert_grouped` (`iq_kernels.cu:710`), qui re-décode les grilles pour
chaque entrée (`iq_kernels.cu:342-345, 369-372`). Appliquer la séparation chargement/application de
`native_mmvq.cu:784-990` pourrait aller jusqu'à 2× sur les fenêtres de 4 à 8 tokens (HYPOTHÈSE).

### O4. Réduire le nombre de nœuds du graphe de fenêtre (fusions et PDL)

**1. Fichiers :** `src/core/verify.cpp:443-444, 472-503, 512-516, 531, 588-594` ; `src/core/layer.cpp:398-402, 887` ;
`src/kernels/cuda/shared_expert.cu:258-306`.

**2. Problème :** le graphe de fenêtre contient de nombreuses boucles par token, là où un kernel par lot suffirait :
```cpp
for (int t = tb; t < te; ++t) { ... moe_route(...) }   // verify.cpp:512-516 : T GEMV BF16 de 2,6 Mo + T top-10
```
Idem pour les projections de l'indexeur (443-444, 482-486), la conversion BF16 (531), la porte (497-503) et la
combinaison (590-594). S'y ajoutent :
- l'expert partagé natif : 9 lancements (`shared_expert.cu:258-306`), dont un `native_scalar_sigmoid<<<1,1>>>` ;
- une conversion BF16 inconditionnelle dont la sortie ne sert pas en mode natif (`layer.cpp:398`) ;
- un `cudaMemcpy2DAsync` pour séparer q et la porte à chaque couche QSA (`layer.cpp:887`), soit 12 nœuds de copie par
  token.

Le coût du lancement est mesuré : le graphe par bloc tourne en 1,585 ms, contre 2,393 ms avec des lancements directs
(`session.hpp:108-112`).

**3. Changement :**
- une GEMV BF16 multi-colonnes pour le routeur, avec un top-10 par lot ;
- une combinaison, une porte et une conversion par lot ;
- fusionner swiglu + quantification et porte sigmoïde + échelle + ajout dans la combinaison ;
- faire lire la moitié q directement par le kernel norm+RoPE ;
- **Programmatic Dependent Launch** : `cudaLaunchAttributeProgrammaticStreamSerialization` +
  `cudaGridDependencySynchronize()`, disponible sur sm_90+ dont sm_120 et compatible avec la capture de graphe. Il
  recouvre le prologue de chaque petit kernel avec la fin du précédent.

**4. Gain (HYPOTHÈSE) :** environ 1 000 nœuds de moins par fenêtre à 1–2 µs pièce, plus le recouvrement PDL : **1 à
3 ms par tour, soit 3 à 8 % (5070) et 6 à 15 % (5090)**. À mesurer avec `nsys` avant d'y investir.

**5. Risque de précision :** faible. Seul l'ordre de réduction du routeur multi-colonnes change si l'on ne garde pas
exactement l'ordre des paires de ggml. Tests : `router_top10_parity`, `shared_expert_parity`, `bf16_gemv_parity`, et
la comparaison KL forcée par l'enseignant (méthode de `bench/results/2026-09-27-cache-parity`).

**6. Effort :** L. **GPU :** toutes ; PDL sur sm_90 et plus.

### O5. Échantillonnage non glouton : sélection top-k en O(k²·V) et fin FP64 redondante

**1. Fichier :** `src/kernels/cuda/sampler.cu:227-238` et `262-303`.

**2. Problème :**
```cpp
for (int j = 0; j < i; ++j) if (sel_ids[j] == v) { taken = true; break; }   // :234, pour CHAQUE v à chaque tour
```
- **Sélection top-k.** Pour k = 20, cela fait ~190 comparaisons en mémoire partagée par entrée du vocabulaire, sur
  248 320 entrées, soit 47 millions par ligne, **sur un seul SM**. Et 20 passes relisent 1 Mo de logits.
- **Fin FP64.** La fin (top_p, température) est calculée en `double` **par les 1 024 threads** (`:262-264` : « Every
  thread computes the same chain redundantly »). GeForce exécute le FP64 à 1/64 du FP32.

**3. Changement (sélection et ordre identiques) :**
```cpp
// éligible si strictement après la sélection précédente dans l'ordre (valeur desc., index asc.)
const bool after_prev = (i == 0) || (s < prev_v) || (s == prev_v && v > prev_i);
```
- Cette condition remplace la boucle `taken` ; `prev_v` et `prev_i` sont diffusés par la mémoire partagée.
- Pour la fin FP64 : faire calculer les `exp` par le warp 0 (une par voie) dans un tableau partagé, puis le préfixe
  ordonné par le thread 0.
- Plus loin : un top-k en deux étapes, sur 64 à 128 blocs par ligne puis une fusion.

**4. Gain (ESTIMÉ) :**
- 0,5 à 0,8 ms par fenêtre, soit **2 à 4 % (5070) et 4 à 6 % (5090) des requêtes échantillonnées**.
- Cela concerne les clients qui envoient `temperature > 0`, ce que font la plupart des applications de chat.
- Le glouton n'est pas concerné.

**5. Risque de précision :** nul ; mêmes candidats, même ordre, même arithmétique. Test : `sampler_parity` (11
fixtures, égalité exacte des tokens).

**6. Effort :** S. **GPU :** toutes.

### O6. Contexte long : l'indexeur, le top-k de blocs et l'attention

**1. Fichiers :** `include/strata/kernels/qsa.hpp:79-86` ; `src/kernels/cuda/qsa_select.cu:24-51, 53-149` ;
`src/kernels/cuda/qsa_decode_attn.cu:18, 111-165` ; `src/core/layer.cpp:845-851`.

**2. Problème :**
- **(a) Parcours de l'indexeur.** Les clés d'indexeur sont en FP32, et chaque bloc de 4 cellules est relu à chaque
  requête (`qsa_select.cu:35-36`). À 128K, cela fait 16,8 Mo par couche QSA et **201 Mo par token**, plus que
  l'attention elle-même (~50 Mo en FP16).
- **(b) Top-k de blocs.** `block_topk_kernel` tourne sur **un seul bloc**. Chaque thread parcourt un morceau contigu
  (lectures non coalescées) et fait des `atomicAdd` sur un histogramme partagé très disputé.
- **(c) Attention.** `qsa_decode_attn` (66 blocs) lit V élément par élément, en série, sans recouvrement.
- **(d) Pas et position.** Ils sont recopiés depuis la mémoire hôte deux fois par couche QSA (`layer.cpp:845-851`),
  soit 22 petits kernels par token au lieu d'un.

**3. Changement :**
- **(a)** Garder la copie FP32 comme référence, ajouter une copie FP16/BF16 écrite par le même kernel de pooling, et
  noter depuis cette copie :
  ```cpp
  const uint2 raw = *reinterpret_cast<const uint2*>(pooled16 + b * IDX_DIM + lane * 4);
  ```
- **(b)** Faire les passes d'histogramme avec des lectures strided coalescées et un histogramme par warp (8 × 256
  entiers) ; garder le parcours contigu seulement pour le comptage et l'émission.
- **(c)** Pour chaque morceau, `cp.async` des lignes V vers la mémoire partagée dès que `srow` est connu, recouvert par
  QK et le softmax. Tester CHUNK = 32 (~130 blocs, mieux adapté à 170 SM). Intégrer la porte sigmoïde dans
  `attn_merge_kernel`.
- **(d)** Un seul tampon de pas partagé par les 12 états QSA.

**4. Gain (ESTIMÉ) :**
- (a) 0,15 ms par token à 128K en FP16 (0,23 en INT8) ;
- (b) 0,1 à 0,5 ms par token (HYPOTHÈSE) ;
- (c) 0,1 à 0,2 ms par token ;
- (d) 30 à 60 µs par token.

Ensemble, **3 à 8 % à 128K**, rien à 4K.

**5. Risque de précision :**
- **(a) change la sélection des cellules.** `qsa.hpp:84-85` exige de le justifier par un taux de changement de
  sélection mesuré. Mesure proposée : sur des prompts de 32K et 128K, le pourcentage de cellules sélectionnées qui
  diffèrent, puis le KL forcé par l'enseignant et le test des aiguilles (5/5).
- (b), (c) et (d) sont identiques au bit près, ou à l'ordre de réduction près pour (c). Tests : `qsa_parity`
  (sélection exacte) et `kv_stream_parity`.
- `qsa_parity` **ne teste ni `qsa_decode_attn` ni `block_scores`/`block_topk`**. Il faut ajouter ces cas.

**6. Effort :** M. **GPU :** toutes.

### O7. Adapter les grilles aux GPU de plus de 48 SM (5090 : 170 SM)

**1. Fichiers :** `src/kernels/cuda/verify_kernels.cu:310, 341` (`48 * 8`) ; `src/kernels/cuda/fused_gdn.cu:144`
(12 blocs) ; pas GDN (48 blocs) ; `qsa_decode_attn.cu:213` (66 blocs) ; `src/prefill/kernels.cu:457` (48 blocs).

**2. Problème :** ces grilles ont été dimensionnées pour les 48 SM de la 5070. Sur une 5090, `gdn_ab_kernel` n'occupe
que 12 SM sur 170, et le pas GDN 48.

**3. Changement :**
- Remplacer les constantes par `cudaDevAttrMultiProcessorCount × k`.
- `gdn_ab` : 4 warps par ligne (48 blocs ou plus).
- Pas GDN : répartir les colonnes d'une tête sur un cluster de 2 à 4 blocs, avec une réduction RMS en mémoire
  partagée distribuée (les clusters existent sur sm_90 et plus ; à vérifier sur sm_120 GeForce).
- `gdn_rec` du prefill : voir P3.

**4. Gain (HYPOTHÈSE) :** 0,2 à 0,6 ms par tour sur la 5090 (**2 à 4 %**), rien sur la 5070. À mesurer avec
`ncu --set full` sur la 5090.

**5. Risque de précision :** nul pour les grilles et les copies. Pour la réduction RMS du pas GDN, seul l'ordre change ;
à valider avec `gdn_parity`, qui ne couvre pas le kernel fusionné (voir B9).

**6. Effort :** S à M. **GPU :** sm_120 grand format (5080 et 5090), et toute carte de plus de 48 SM.

### O8. Côté CPU : équilibrage du pool et pages larges transparentes (surtout RTX 5070 et 5060)

**1. Fichiers :** `src/kernels/cpu/pool.cpp:438` ; `src/core/pinned.cu:87` ; `src/kernels/cpu/expert.cpp:174-192`.

**2. Problème :**
- **Équilibrage.**
  ```cpp
  parts_a_ = (std::max)(1, (3 * threads + n - 1) / n);   // n = 10 manqués, 6 threads -> 20 tâches : 4/4/3/3/3/3
  ```
  La dernière vague ne fait tourner que 2 threads sur 6. `run_split_multi` découpe déjà exactement en 3 × threads
  tâches (`pool.cpp:467-468`).
- **Pages larges.** Sans pool `hugetlb` configuré, l'arène de 34 Go retombe sur des pages de 4 Ko sans
  `madvise(MADV_HUGEPAGE)` (`pinned.cu:87`). Chaque expert couvre alors ~338 pages, avec un défaut de TLB presque à
  chaque page, puisque les experts sont tirés au hasard dans 34 Go.

**3. Changement :**
- Reprendre le découpage plat de `run_split_multi` dans `run_split`. C'est identique au bit près : chaque ligne passe
  par le même `row_dot`.
- Pour les pages :
  ```cpp
  p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p != MAP_FAILED) madvise(p, bytes, MADV_HUGEPAGE);   // avant cudaHostRegister et avant le premier accès
  ```
  Puis indiquer dans `note` la part réellement obtenue (`AnonHugePages`).

**4. Gain (ESTIMÉ) :** équilibrage 5 à 8 % du temps CPU des experts ; pages larges 2 à 8 % sous Linux, sans hugetlb. De
bout en bout, **2 à 5 % sur la 5070** (où le CPU pèse 40 % du tour) et **< 1 % sur la 5090**.

**5. Risque de précision :** nul. Tests : `pool_test`, `pool_stress` et `expert_parity`. Ajouter à `pool_test` un cas
bit à bit pour `run_split`, qui n'est pas couvert aujourd'hui.

**6. Effort :** S. **Plateforme :** CPU x86 ; la partie pages larges concerne Linux.

**Déjà tentés, à ne pas refaire :**
- les pages larges de 2 Mo `hugetlb` sur V100 (PR #3 : 12,2 contre 35 à 37 tok/s) ;
- le passage aux pages larges `hugetlb` sous Linux.

La proposition ici est la version transparente (THP), à mesurer.

### O9. Côté serveur : tokenisation et détokenisation incrémentales

**1. Fichiers :** `serve/server.py:482-487` et `594-595` ; `tools/strata_tokenizer.py:140-156, 197-207`.

**2. Problème :**
- **Détokenisation.** Chaque token re-décode **toute** la réponse :
  ```python
  self.ids.append(t)
  text = self.tok.decode(self.ids)   # server.py:483-484
  ```
  **MESURÉ** (boucle synthétique reproduisant `decode`) : 0,7 ms par token à 4K tokens, 2,2 ms à 16K. Soit ~17 s de
  CPU pour une réponse de 16K, sur un cœur que le pool d'experts utilise aussi.
- **Tokenisation.** À chaque requête, la conversation entière est rendue par Jinja puis re-tokenisée en BPE Python
  pur, sans cache. Estimé à 1 à 2 s de délai avant le premier token à 100K tokens, même quand le moteur réutilise
  99 % du préfixe.

**3. Changement :**
- Détokenisation : un tampon d'octets plus `codecs.getincrementaldecoder("utf-8")`.
- Tokenisation :
  - `functools.lru_cache` sur `_bpe` ;
  - puis mémoriser le rendu et les ids précédents, et ne tokeniser que la fin après le dernier token spécial commun.
    Le résultat est exact, car les tokens spéciaux sont des frontières BPE.

**4. Gain (ESTIMÉ) :** 1 à 2 s de délai avant le premier token par tour à 100K. Un cœur libéré pendant les longues
réponses : quelques % de décodage sur les machines où le CPU est limitant (HYPOTHÈSE).

**5. Risque :** nul si les ids sont identiques. À valider en comparant les ids sur `serve/chat_golden.json` et sur de
longues conversations.

**6. Effort :** S. **Plateforme :** toutes.

### Prefill : optimisations spécifiques

Mesurer d'abord avec `STRATA_PREFILL_TIMING=1`. Les gains ci-dessous portent sur le débit de lecture du prompt, donc
sur le délai avant le premier token.

| # | Fichiers | Problème | Changement | Gain (étiquette) | Précision / test | Effort |
|---|---|---|---|---|---|---|
| **P1** | `prefill.cpp:1108-1109, 1207-1218, 838-847` | Une `cudaStreamSynchronize` et une copie des ids vers l'hôte **par couche MoE** (48 par morceau), puis ~6 appels d'API CUDA par expert (~3 000 par couche). En `stream_all`, l'ordre des copies ne dépend pas du routage. | Tri par expert sur le GPU (atomiques + scan), groupes à plages d'ids fixes, copies des experts consécutifs en un seul `cudaMemcpyAsync` (ils sont contigus dans l'arène), rassemblement par groupe en un lancement ; la couche devient capturable en graphe. | 5 à 10 % (HYPOTHÈSE), davantage sur 5090 | Identique au bit près si l'ordre des entrées par expert est conservé ; hash d'état GDN après 32K | M |
| **P2** | `prefill/kernels.cu:121-133` | `gdn_conv` : un filtre FIR à 4 coefficients exécuté comme une récurrence (un thread par canal, 8 192 pas en série, 80 blocs). | Un thread par (canal, t), qui lit `in(t-3..t)` ; un second petit kernel écrit l'historique ; fusionner avec `gdn_l2_kernel`. | 2 à 5 % (ESTIMÉ) | **Identique au bit près** (mêmes opérations, même ordre) | S |
| **P3** | `prefill/kernels.cu:149-195, 457` | `gdn_rec_kernel` : 48 blocs (un par tête), ~5 `__syncthreads` par token. | Répartir les 128 colonnes d'une tête sur 4 blocs ; appliquer la norme dans une seconde passe. Plus tard, forme par morceaux « WY » sur tenseurs. | 5 à 10 % (HYPOTHÈSE) | Seul l'ordre de réduction de la norme change ; `gdn_parity` et KL forcé par l'enseignant | M |
| **P4** | `gemm.cu:96-112` | Les projections denses natives sont déquantifiées en FP16 puis multipliées en FP16 avec accumulation FP32 : sur GeForce, demi-débit FP16 et ~¼ du débit INT8. ~59 TFLOP par morceau. | Réutiliser MMQ int8 (déjà compilé pour les experts) en mode un seul expert, `bounds = {0, T}`. | ~10 % (ESTIMÉ) | **Change l'arrondi** (activations q8_1) : KL forcé par l'enseignant, avec la méthode et l'étalon de `bench/results/2026-09-28-prefill-speed` | M |
| **P5** | `prefill.cpp:992-999` | `native_qsa_indexer_append` lancé **une fois par token** : 8 192 × 12 ≈ 98 000 lancements par morceau. | Un kernel d'ajout par lot : pooling parallèle par bloc complet, une passe série seulement pour le bloc de queue. | 2 à 6 % (HYPOTHÈSE) | `qsa_parity` (clés poolées ≤ 1e-6) | M |
| **P6** | `prefill.cpp:1077-1082` | Attention QSA du prefill faite par le kernel de décodage, par lots de 32 requêtes (256 lancements par couche, aucune réutilisation de K/V sur tenseurs). | Si `STRATA_PREFILL_TIMING` montre une part importante : un noyau flash clairsemé sur l'union des blocs choisis par tuile de 64 requêtes. | Inconnu : à mesurer | À valider par KL et aiguilles | L |
| **P7** | `generate.cpp:3293-3307`, `2886-2900` | Re-remplissage **bloquant, un emplacement à la fois**, des emplacements prêtés après chaque prompt : ~180 ms en service selon un commentaire (`2843-2844`), plusieurs Go en ligne de commande. Ce temps n'est pas compté dans les tok/s de prefill affichés (`3316-3319`). | Réadmettre de façon asynchrone avec le mécanisme `pending` + événement du cache adaptatif (`3595-3606`) ; le CPU calcule ces experts en attendant. | 0,2 à 0,5 s de délai avant le premier token par requête (ESTIMÉ) | Aucun (même mécanisme que les échanges adaptatifs) | S à M |
| **P8** | `prefill/kernels.cu:58-101` | Chaîne GR élément par élément : ~370 Ko par token par moitié, soit ~290 Go par morceau. | Ne stocker que l'échelle RMS par ligne, recalculer `xn` dans `gr_mix` ; fusionner `gr_write(h)` avec `gr_norm(h+1)`. | 2 à 3 % (ESTIMÉ) | Ordre d'arrondi : `gr_parity` | M |

**Déjà essayé et rejeté :** les GEMM groupées cuBLAS pour les experts du prefill (770 contre 790,
`bench/results/2026-09-28-prefill-speed/README.md:18`). P1 et P4 sont des propositions différentes.

---

## 4. Efficacité (mémoire, copies, allocations)

| # | Fichiers | Constat | Proposition | Gain |
|---|---|---|---|---|
| E1 | `session.cpp:844-847`, `verify.cpp:588-594` | `copy_from_mapped` relit **les k lignes** de sortie CPU sur PCIe (100 Ko par couche), y compris les lignes des experts calculés sur le GPU, qui valent zéro ; ensuite `moe_hit_add` et la combinaison font trois kernels. | Un seul kernel qui lit les lignes manquées dans le `y_miss` mappé, filtrées par le masque de hits, puis les lignes GPU, et calcule `Σ w·y + partagé`. Allouer `y_miss` en `cudaHostAllocWriteCombined` (HYPOTHÈSE). | 0,2 à 0,5 ms par token (ESTIMÉ) |
| E2 | `generate.cpp:648-661` | Points de reprise de conversation : `cudaDeviceSynchronize` puis ~118 Mo copiés vers l'hôte dans des `std::vector` **paginables et réalloués**. | Un pool de tampons épinglés et `cudaMemcpyAsync` sur un flux séparé. | 10 à 30 ms par point de reprise (ESTIMÉ) |
| E3 | `generate.cpp:3001-3004`, `suffix_drafter.cpp:27-31` | Le chercheur de suffixe est réinitialisé à chaque requête : 16 Mo à 128K, 32 Mo à 262K, puis tout le prompt est re-haché. | Ajouter seulement les nouveaux tokens quand la requête prolonge la session vivante ; sinon un effacement paresseux par numéro d'époque. | 5 à 15 ms de délai avant le premier token (ESTIMÉ) |
| E4 | `rope.cu`, `session.cpp:57`, `layer.cpp:632-641` | En mode `--native`, la table RoPE FP64→FP32 n'est plus utilisée (`native_rope` calcule les angles), mais elle occupe 64 Mio de VRAM à 262K. | Ne pas l'allouer en mode natif. | 64 Mio, soit ~46 experts |
| E5 | `expert_cache.cpp:248-295` | Au démarrage, `verify_slot` relit **chaque** emplacement du profil en entier (~5,7 Go, et bien plus sur une 5090). | Vérification par échantillonnage ou par hash. | Démarrage plus rapide de plusieurs secondes |
| E6 | `tools/make_profile.py:82-91` | Le profil livré classe déjà les 24 576 paires, donc les traces de l'utilisateur ne changent rien au profil de base (`n_trace = 0`). La docstring dit le contraire. | Classer par score mélangé : fréquence dans la trace, puis rang de base. | Meilleur taux de hits initial (HYPOTHÈSE) |

---

## 5. Précision, bugs et divergences

### B1. Historique de pénalités en fenêtre spéculative : lignes non initialisées ou hors tampon (GRAVE)

- **Fichiers :** `src/program/generate.cpp:2343-2349, 2940-2941, 3043-3055` ; `include/strata/core/verify.hpp:74-81` ;
  `src/core/verify.cpp:849` ; `src/kernels/cuda/sampler.cu:97, 185`.
- **Constat (vérifié) :**
  - Le moteur alloue **une seule ligne** de 4 096 entiers (`d_hist`) et n'en remplit que `hist_n`.
  - `verify.cpp:849` appelle `sample_tokens(head_logits_, T, ..., hist_d_, hist_len_, ...)` avec T lignes.
  - Le kernel lit la ligne t à `history + t * history_len` (`sampler.cu:97, 185`).
  - Les lignes 1 à T−1 lisent donc de la mémoire `cudaMalloc` jamais écrite. Dès que `(t+1) × hist_n > 4096`, elles
    lisent **hors du tampon** (par exemple `penalty_last_n = 1024` avec T = 5).
  - Un identifiant parasite ≥ `n_vocab` écrit ensuite **hors du bitmap en mémoire partagée**
    (`atomicOr(&penal_bits[hrow[i] >> 5], ...)`, qui ne vérifie que `hrow[i] >= 0`).
  - Même avec un tampon valide, la ligne t ne compte pas les brouillons 1..t qui la précèdent dans la fenêtre.
- **Conséquences :**
  - Toute requête avec `presence_penalty`, `frequency_penalty` ou `repetition_penalty`, gloutonne ou non, produit un
    texte qui diffère du décodage non spéculatif et dépend de ce que contient la mémoire.
  - Dans le pire cas, une écriture hors limites en mémoire partagée.
- **Correction :**
  - Construire T lignes par fenêtre : ligne t = la queue d'historique décalée de t, suivie de `window[1..t]`.
  - Allouer T_max × 4 096.
  - Borner `hrow[i] < n_vocab` dans les deux kernels.
- **Test :** ajouter à `sampler_parity` une fixture multi-lignes. Vérifier aussi, avec une pénalité active, qu'un
  décodage spéculatif produit exactement le même texte que `--spec 0`.
- **Effort :** S.

### B2. Barrière sautée dans `s_gemv_q8_split_kernel` (latent)

`src/kernels/cuda/s_gemv.cu:150` : `if (o >= n_out) return;` précède le `__syncthreads()` de `:157`. C'est un
comportement indéfini dès que `n_out % 8 != 0`. Les autres kernels du fichier évitent ce piège (`:96-97`). Les formes
actuelles sont des multiples de 8. **Correction :** placer la sortie après la barrière. **Effort :** S.

### B3. `top_k` hors de 1..64 dans le sampler

`sampler.cu:211-218` :
- pour `top_k <= 0` avec température > 0, le kernel **n'écrit pas** `out[t]` et renvoie un token périmé ;
- au-delà de 64, il **tronque en silence**.

La ligne de commande accepte `--top-k 0` (`generate.cpp:1043` ne refuse que les valeurs < 0). Le serveur, lui, ignore
les `top_k` hors plage (`server.py:237-238`). **Correction :** refuser explicitement, ou traiter `0` comme
« désactivé » via le chemin k = 64.

### B4. Précision du RoPE natif aux grandes positions (HYPOTHÈSE, chemin de production)

- `native_rope.cu:54-55` calcule `theta = pos * powf(theta_scale, pair)` puis `cosf` et `sinf` en FP32. Le fichier est
  compilé avec `--use_fast_math` (`CMakeLists.txt:204-217`), donc `__powf`, `__sinf` et `__cosf`. Même chose dans
  `native_qsa_indexer.cu:86-87`.
- À la position 131 072, l'erreur de phase estimée atteint ~1e-2 rad sur les paires hautes fréquences, et elle croît
  avec la position. Le chemin par table (`rope.cu:39-42`) est calculé en FP64 et est exact au dernier arrondi FP32 près.
- C'est sans doute voulu, pour reproduire llama.cpp, qui compile aussi en fast-math. Mais `rope_parity` ne teste que
  les positions < 32 (`rope_parity.cpp:49`) et jamais `native_rope`.
- **À faire :** ajouter des positions de 1K à 262K au test. Pour mesurer l'effet, comparer par KL et aiguilles à 128K
  l'angle calculé en révolutions FP64 (`sincospi`).

### B5. Deux quantificateurs Q8_1 compilés différemment (dernier bit)

- `iq_kernels.cu:416-417` fait une division IEEE ; il n'est pas dans la liste fast-math.
- `native_mmvq.cu:147-148` fait une division approchée ; il est dans la liste.

Or `native_expert_grouped` quantifie ses activations intermédiaires avec le premier. Un int8 bascule quand `x/d` tombe
à moins d'un ulp de .5. **Correction :** utiliser `__fdividef` explicitement, comme le fait déjà
`shared_expert.cu:68`.

### B6. Softplus différent entre deux chemins GDN

- Chemin fusionné : `v > 20 ? v : log1pf(__expf(v))` (`fused_gdn.cu:120`).
- Pré-traitement natif : `logf(1.0f + expf(value))` (`native_gdn_preprocess.cu:95`), qui renvoie exactement 0 pour
  v ≲ −17 (décroissance = 1) et perd de la précision relative pour v < −5.

Les deux chemins ne sont pas bit à bit égaux. **À faire :** confirmer que le chemin natif reproduit bien llama.cpp, et
sinon passer à `log1pf`.

### B7. Débordements FP16 possibles dans le prefill (HYPOTHÈSE)

`hf(f)` vaut `__float2half_rn(f)` sans écrêtage (`prefill/kernels.cu:36`). Il alimente les produits SwiGLU de l'expert
partagé (`:282`), `y_h`, `attn_h`, `mixed_h` et le K/V FP16. La présence de `STRATA_DBG_NAN` suggère que des valeurs
non finies ont déjà été vues. **Correction :** écrêter à ±65 504, ou passer en BF16 les tampons à risque.

### B8. Routeur du prefill et du décodage : `expf` précis contre approché

Le prefill utilise `expf` précis (`prefill/kernels.cu:221`) ; le routeur natif du décodage est compilé en fast-math
(`CMakeLists.txt:210`). Près d'une égalité, le top-10 peut différer entre prompt et génération. C'est un effet mineur
mais systématique. De plus, le prefill remplace les NaN par `-FLT_MAX` en silence (`:224`).

### B9. Tests de parité manquants sur les kernels de production

Plusieurs kernels qui tournent en production n'ont **aucun** test dans l'arbre publié :
- `qsa_decode_attn`, `block_scores` et `block_topk` ;
- `fused_gdn` et `native_gdn` ;
- `fused_gr` ;
- `native_rope` ;
- le routeur natif ;
- le MMVF natif.

`gdn_parity` ne teste qu'un pas, jamais la dérive sur plusieurs pas. `kv_q4_parity` affiche l'erreur de reconstruction
**sans l'affirmer** (`kv_q4_parity.cpp:325`). Les tests oracle contre llama.cpp (`bench/micro/*`) ne sont pas publiés.
**Avant d'optimiser ces kernels, il faut ajouter ces tests**, sans quoi aucune régression ne sera détectée.

### B10. Étalon de qualité du prefill suspect (à vérifier)

`bench/results/2026-09-28-prefill-speed` prend comme étalon « chunk 6144 contre 8192, même code » : top-1 à 89,8 %,
KL moyen 0,33. C'est très élevé pour un simple changement d'ordre de sommation. Cela suggère un effet lié aux
frontières de morceau : bloc de queue de l'indexeur, historique PLE, ou encore la sélection QSA.

**Test :** comparer un prompt de 2K lu en morceaux de 8 192, 4 096 et 1 024 au chemin token par token, et comparer le
hash d'état GDN entre tailles de morceau. Si c'est un bug de frontière, sa correction améliorera aussi la qualité.

### B11. Divers

- **Sans MTP, fenêtres pleines de token 0.** Avec `--spec` sans `--mtp` ni oracle, les fenêtres sont remplies de
  token 0 (`generate.cpp:3687, 3712-3713`) : autant de travail perdu. Refuser cette configuration ou forcer T = 1.
- **Échantillonnage ignoré hors serveur.** En dehors de `--serve`, `--spec` ignore `--seed`, `--top-k`, `--top-p` et
  `--temperature` après le premier token : `set_sampling` n'est appelé qu'en service (`generate.cpp:2939`).
- **Débordement latent.** `expert_source.cpp:281-300` utilise des tableaux fixes `kind[128]`, `distinct[128]` et
  `dma_src[64]`, valides seulement tant que MAXT × k ≤ 128. Ajouter un `static_assert`.
- **Page non résolue.** `qsa_decode_attn.cu:104-105` n'a pas de garde `page < 0` si la résolution du streaming KV
  déborde. Ajouter `if (page < 0) r = -1;`.
- **NaN converti en −0.** `bf16_from_f32` (`bf16_bits.hpp:66-71`) : `0x7FFFFFFF + 0x8000` donne −0.
- **Balayage CLOCK.** Dans `kv_stream.cu:114`, le balayage peut attribuer deux fois le même emplacement si
  `n_slots < 1024`. Ce cas est impossible aujourd'hui (minimum de 5 120 emplacements), mais doit être vérifié.
- **Non-déterminisme du glouton.** Le glouton n'est pas reproductible d'un lancement à l'autre avec le cache adaptatif
  (`generate.cpp:3598-3606`) : le GPU et le CPU arrondissent différemment. Le projet le documente déjà
  (`DETAILS.md:305`). **Pour toute mesure A/B, utiliser `--adapt-every 100000`.**

---

## 6. Les trois changements à faire en premier

**Préalable, S, obligatoire :** corriger **B1** et rendre les tests de parité compilables (le `tests/` absent,
section 0).

1. **O1, lectures PLE hors du chemin critique.** Effort M. Gain estimé de 6 à 15 % en décodage sur une 5090 sous
   Linux, 1 à 2 % sous Windows. Résultat identique au bit près. La mesure existe déjà dans le moteur (ligne `ple io`).
2. **O3, kernel groupé des experts VRAM.** Effort S. Gain estimé de 4 à 8 %, identique au bit près. C'est le calcul qui
   domine sur une carte de 32 Go.
3. **O2, suppression des allers-retours du tour spéculatif.** Effort M. Gain estimé de 6 à 15 % sur une 5090, plus
   sous Windows. Commencer par la chaîne MTP en un graphe et la synchronisation du `commit`.

**Protocole de mesure** pour chacun, avant et après :
- 3 exécutions × 3 prompts : `--adapt-every 100000`, glouton, 256 tokens ;
- tokens/s et `tokens_per_round` ;
- `nsys profile --cuda-graph-trace=node` sur 20 tours ;
- tokens identiques pour O1 et O3.

## 7. Pistes spéculatives, à mesurer avant d'investir

- **O4, fusions et PDL** (3 à 15 %) : d'abord `nsys` pour compter le temps entre nœuds par fenêtre.
- **Fenêtre plus longue (`--spec 5` ou `6`) sur la 5090.** Avec ~93 % d'experts en VRAM, un token de plus par fenêtre
  coûte peu de CPU. La politique de brouillon (`draft_policy.cpp`) décide déjà en tokens par ms ; balayer
  `--spec` de 4 à 6.
- **Re-régler `--pcie-frac` en PCIe 5.0.** Depuis e6265c7 (repli sur le kernel de copie à 0,55 pour les packs natifs,
  à cause du blocage #31), aucun balayage n'a été fait. La PR #44 (sonde de bande passante) n'est pas fusionnée.
- **Prédire les experts de la couche suivante** à partir de l'état caché, pour lancer leurs copies avant le routeur
  (piste de l'article §7). Aucune mesure de localité du routage n'existe dans le dépôt.
- **Brouillon MTP mieux quantifié.** On mesure, sur des blocs synthétiques, une erreur RMS de
  0,41 → 0,35 en élargissant la recherche d'échelle de `mtp_pack.py:58-82`, avec échelle négative. Un meilleur taux
  d'acceptation serait un gain direct, mais il faut d'abord la prise en charge de `d < 0` et `mtp_rt.py:72`, qui
  n'accepte que Q2_0.
- **État GDN en BF16** (moitié moins de trafic, ~0,17 ms par token) : risque de dérive sur les longues séquences, à
  valider par un test multi-pas qui n'existe pas encore.
- **Top-k d'échantillonnage en deux étapes** et argmax glouton multi-blocs fusionné dans la tête (O5, étape 2).
- **Prefill par couche entière** (tout le prompt traverse la couche l avant la couche l+1) : chaque expert n'est
  diffusé qu'une fois par prompt (33 Go au lieu de ~130 Go à 32K). C'est intéressant sur les emplacements PCIe x8 ou
  reliés au chipset ; ~1,3 Go de VRAM pour le résidu à 32K.

## 8. Ce qui n'a pas pu être vérifié

- **Toute mesure sur GPU.** Chaque gain est une estimation. Les coûts FP64 supposent le rapport 1/64 des GeForce, et
  les caractéristiques de la 5090 viennent des fiches techniques.
- **Les descriptions, discussions et reviews des PR, et les issues** (accès refusé). L'historique a été reconstitué à
  partir des commits et des diffs des PR.
- **L'exactitude des kernels natifs face à llama.cpp** : les tests oracle (`bench/micro/*`) ne sont pas publiés.
- **Le taux de réussite du cache de lignes PLE et la latence SSD sur votre machine**, qui déterminent le gain d'O1.
- **La répartition réelle du temps de prefill** (`STRATA_PREFILL_TIMING`), qui ordonne P1 à P6.
- **Les chiffres de référence cités dans les commentaires mais absents du dépôt :** les ~180 ms du re-remplissage,
  les ~9 ms de temps hôte, les 82 % de réussite du cache PLE, les coûts du contrôleur de brouillon.
- **Le code SASS** : les conclusions sur les conflits de banques et les lectures d'octets viennent du PTX
  (`cuobjdump` et `nvdisasm` ne sont pas dans les paquets pip de CUDA).
- **La prise en charge des clusters et de la mémoire partagée distribuée sur sm_120 GeForce** (O7).

## Annexe : ce que le projet a déjà essayé (ne pas reproposer)

| Idée | Résultat | Source |
|---|---|---|
| Couper la fenêtre en deux groupes de tokens pour recouvrir CPU et GPU | 7 % plus lent (85,8 contre 91,8 tok/s) | article, constat 5 ; `generate.cpp:253` |
| x en mémoire partagée dans la GEMV S2 ; table de codes en mémoire constante | pire ; neutre | `s2_gemv_fast.cu:1-18` |
| `#pragma unroll 4` dans `s_gemv` | 184,6 → 183,9 Go/s, retiré | `s_gemv.cu:202-207` |
| TPR = 64 pour les GEMV découpées en lignes | 11,75 contre 11,99 tok/s | `layer.cpp:101-106` |
| GEMM groupées cuBLAS pour les experts du prefill | 770 contre 790 tok/s | `prefill-speed/README.md:18` |
| Part PCIe par kernel de copie au-delà de 20 % | plus lent ; le DMA payait, mais il a été abandonné à cause du blocage #31 | article, constat 9 ; commit e6265c7 |
| Admission globale dans l'ordre d'arrivée ; remplissage par manqués obligatoires | 2,97 % et 0,4864 de hits, contre 0,6447 avec un profil | `expert_cache.hpp:92-103` ; `generate.cpp:182-194` |
| Experts en `mmap` | 71,97 contre 34,78 ms par token | `generate.cpp:403-405` |
| Pages larges `hugetlb` (V100) | 12,2 contre 35 à 37 tok/s | PR #3 |
| KV Q4_0 par défaut | perplexité +8 à 12 %, laissé en option | `bench/results/2026-09-27-kv-q4` |
| Rendre le dump et le scan des logits optionnels | mesuré plus lent | `generate.cpp:3466-3471` |
