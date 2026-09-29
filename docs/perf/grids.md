# grids : grilles de décodage dimensionnées sur le GPU, plus sur 48 SM (issue #26, O7)

Branche `perf/grids`, partie de `perf/base` (b858f9b, moteur 0.1.20). Aucun chiffre GPU n'a été mesuré : la
machine de développement n'a pas de GPU. Tout a été compilé (CUDA 13.0, sm_120 ; les deux fichiers de kernels aussi
pour sm_86, pour vérifier la garde des clusters) et les tests CPU ont été lancés ; `grids_parity` compile mais reste
à lancer sur la RTX 5090.

## Ce qui change

| # | Source (issue #26) | Changement | Par défaut | Retour à l'ancien chemin |
|---|---|---|---|---|
| 1 | `fetch_blobs`, `gather_rows` : `48 * 8` blocs codés en dur | `8 × multiProcessorCount` du GPU courant, lu une fois par GPU (`launch_grid.hpp`). Sur une carte à 48 SM, lancement identique. | **activé** | `STRATA_OLD_GRIDS=1` |
| 2 | `gdn_ab_kernel` (et `gdn_ab_multi_kernel` du verify) : 96 lignes sur **12 blocs** (un warp par ligne) | **Un bloc de 4 warps par ligne** (96 blocs) : les 128 threads chargent la ligne en mémoire partagée d'un coup, puis la même chaîne d'`fmaf` qu'avant. | **activé** | `STRATA_OLD_GDN_AB=1` |
| 3 | Pas GDN (`fused_gdn_step_norm`, `gdn_step_norm_multi`) : **48 blocs** (un par tête de valeur) | Les 128 colonnes d'une tête réparties sur un **cluster de 4 blocs** (192 blocs), somme RMS échangée en mémoire partagée distribuée. sm_90 et plus, vérifié à l'exécution. | **activé** (si le GPU accepte les clusters) | `STRATA_OLD_GDN_STEP=1` |
| 4 | `qsa_decode_attn` (66 blocs), `gr_down` (41 blocs), prefill `gdn_rec_kernel` (48 blocs), autres | Revus, **non modifiés** (voir « Ce qui reste »). | — | — |

Les trois changements sont **identiques au bit près par construction** (mêmes valeurs, mêmes opérations, même
ordre ; seuls le nombre de blocs et le warp ou le bloc qui porte une chaîne changent). Ils sont donc actifs par
défaut, chacun avec sa variable pour l'A/B ; les trois ensemble redonnent exactement les lancements de 0.1.20. Les
variables sont lues une fois par processus.

Dans la configuration écrite par `setup.py` (pack natif, `--spec 4`), le décodage passe par les fenêtres de
vérification : ce sont `gdn_ab_multi`, `gdn_step_norm_multi` (verify **et** commit) et `fetch_blobs`
(`--pcie-mode auto` = kernel de copie) qui tournent. `fused_gdn_ab` et `fused_gdn_step_norm` sont ceux du chemin
jeton (`--spec 0`, pack S2).

### 1. Copies à pas de grille (`src/kernels/cuda/verify_kernels.cu`, `include/strata/kernels/launch_grid.hpp`)

- `grid_device()` lit une fois **par GPU** `cudaDevAttrMultiProcessorCount`, la capacité de calcul et
  `cudaDevAttrClusterLaunch` (`PerDevice<T>` : un `std::call_once` par numéro de GPU ; la réponse « le kernel en
  cluster se lance-t-il ici » de chaque pas GDN est gardée de la même façon). Avec la répartition des couches
  sur plusieurs GPU (0.1.21, [../MULTI_GPU.md](../MULTI_GPU.md)), chaque carte a ses propres grilles, et une RTX 3090
  (sm_86, sans clusters) à côté d'une RTX 5080 garde l'ancien pas GDN au lieu de recevoir les kernels en cluster. Le premier lancement pouvant se trouver dans une capture de graphe (mode
  `ThreadLocal`), les requêtes passent par `cudaThreadExchangeStreamCaptureMode(Relaxed)` ; une requête qui échoue
  n'est pas remontée comme erreur du lancement suivant (48 SM par défaut).
- `fetch_blobs` et `gather_rows` : `grid_sms() * 8` blocs de 256. Ce sont des boucles à pas de grille : chaque
  élément est copié une fois, quel que soit le nombre de blocs.
- **Gain attendu : à peu près nul pour `fetch_blobs`.** Il lit la part PCIe des experts manqués en mémoire hôte
  mappée : 384 blocs × 256 threads × 16 o = 1,5 Mo en vol, bien au-delà du produit débit × latence du PCIe. Sur la
  5090 la grille passe à 1 360 blocs ; quand la part PCIe d'une couche est vide (`*n = 0`, fréquent avec un grand
  cache d'experts) le lancement de blocs vides peut coûter quelques dixièmes de µs de plus. `grids_parity --bench`
  mesure les deux cas (`fetch_blobs 0 x 1 MiB` et `8 x 1 MiB`) ; si la ligne `0 x` se dégrade, garder
  `STRATA_OLD_GRIDS=1`. `gather_rows` ne sert qu'au chargement (tête de brouillon MTP).

### 2. Projection alpha/beta, un bloc de 4 warps par ligne

`fused_gdn.cu` : `gdn_ab_row_kernel` ; `verify_kernels.cu` : `gdn_ab_multi_row_kernel`.

- Avant : 12 blocs de 8 warps, un warp par ligne. Chaque voie parcourt ses 10 blocs de 8 valeurs d'une ligne de
  2 560 ; le compilateur déroule par 4, soit environ 3 allers-retours DRAM successifs par warp, sur 12 SM.
- Après : 96 blocs de 128 threads, un par ligne. Tous les threads chargent la ligne (et `x` pour le jeton seul) en
  mémoire partagée avec des chargements prédiqués entièrement déroulés (4 blocs de 8 au plus par thread, tous en
  vol ensemble : **un** aller-retour DRAM), puis :
  - `fused_gdn_ab` : le warp 0 exécute la chaîne d'origine telle quelle ;
  - `gdn_ab_multi` : le warp w exécute les jetons w et w + 4 (`kVerifyMaxT` = 8), chaque chaîne (voie, jeton)
    étant celle d'origine ; `x` reste lu en mémoire globale, comme avant.
- **Pourquoi c'est identique au bit près** : même voie pour chaque bloc de 8, même ordre des blocs, même suite
  d'`fmaf` explicites, même papillon `__shfl_xor_sync`, même épilogue (softplus avec sa branche `> 20`, sigmoïde).
  Seul l'endroit d'où viennent les opérandes change (mémoire partagée au lieu de globale).
- Pourquoi pas une somme répartie sur les 4 warps : elle changerait l'ordre de réduction (donc opt-in). La chaîne
  d'un warp fait 80 `fmaf` dépendants (~0,15 µs), moins qu'un aller-retour DRAM ; la répartir ne gagnerait que
  < 0,1 µs par lancement. Non fait.
- Lignes de plus de 4 096 : ancien kernel (garde dans le lanceur). Mémoire partagée dynamique : 48 o par bloc de 8
  (15 Ko pour 2 560) pour le jeton seul, 16 o (5 Ko) pour le verify.

### 3. Pas GDN sur un cluster de 4 blocs par tête

`fused_gdn.cu` : `gdn_step_norm_cluster_kernel` ; `verify_kernels.cu` : `gdn_step_norm_multi_cluster_kernel`.

- Avant : un bloc de 512 threads (128 colonnes × 4 groupes de 32 lignes) par tête : les 3,1 Mo d'état lus (et
  réécrits au commit) par 48 SM, quel que soit le GPU.
- Après : 4 blocs de 128 threads (32 colonnes × 4 groupes de lignes) par tête, en cluster `(4, 1, 1)` ; 192 blocs.
  - L'arithmétique d'une colonne ne quitte pas son thread et reste ligne pour ligne celle d'avant (ses 32 lignes par
    groupe, les 4 partiels sommés dans le même ordre).
  - La norme RMS couvre la tête : le warp 0 de chaque bloc tient 32 colonnes consécutives avec la même voie par
    colonne que le warp `part` de l'ancien kernel, donc le même papillon ; il pousse sa somme dans **chaque** bloc du
    cluster (`__cluster_map_shared_rank`), puis une barrière de cluster (release/acquire) ; `ss` additionne les
    mêmes 4 sommes de warp dans le même ordre. Une première barrière (`arrive.relaxed` au début, `wait` avant la
    première écriture distante) garantit que tous les blocs tournent ; la dernière garantit qu'aucun bloc ne lit
    ni n'écrit chez un autre après.
  - Verify/commit : la norme de chaque jeton de sortie est **reportée après la boucle des jetons** (sommes par
    jeton dans `wsum[t][4]`, sorties du groupe 0 dans `ocs[t][32]`) : une seule barrière de cluster par lancement.
    Les partiels kv et o ont chacun leur tableau : 4 barrières de bloc par jeton au lieu de 6 (5 pour un jeton
    rejoué). Mêmes valeurs écrites aux mêmes adresses, `y` du commit compris.
  - `wsum`/`ocs` contiennent `kVerifyMaxT` jetons. L'hôte garantit `T` et `*n_keep` ≤ `kVerifyMaxT` ; au-delà
    (appelant défectueux), le kernel s'arrête par `__trap()` au lieu de tronquer en silence (l'ancien kernel, lui,
    lirait au-delà de `hbuf`).
- **Vérification statique** : pour le jeton seul, la suite des opérations flottantes du PTX des deux kernels est
  identique (164 opérations, mêmes arrondis et constantes ; script de comparaison sur le PTX de `nvcc -O3`).
- **Support sm_120 GeForce** (la question de l'audit) : le lanceur n'emploie le cluster que si
  `cudaDevAttrClusterLaunch` = 1, le kernel a été compilé pour sm_90+ (`cudaFuncGetAttributes().ptxVersion` ;
  compilé pour sm_80/86/89, son corps est vide) et `cudaOccupancyMaxActiveClusters` > 0. Sinon : ancien kernel. Un
  refus de `cudaLaunchKernelEx` arrête le programme avec un message qui cite `STRATA_OLD_GDN_STEP=1`.
  `grids_parity` affiche la ligne `cluster: fused_gdn_step_norm yes, gdn_step_norm_multi yes` quand le chemin
  cluster tourne, et **échoue** (code de sortie 1, donc ctest rouge) si le GPU accepte les clusters mais que le
  chemin cluster n'a pas été pris (requête de disponibilité en échec, binaire sans code sm_90+) : sinon les tests
  du pas compareraient l'ancien kernel à lui-même et resteraient verts. `--allow-no-cluster` accepte ce repli ;
  `--require-cluster` échoue sur tout GPU sans chemin cluster.
- **Premier lancement dans une capture de graphe** : ctest `grids_parity_capture` (`grids_parity --capture`, un
  processus neuf) fait le **premier** appel de chaque lanceur (`fetch_blobs`, `gather_rows`, `fused_gdn_ab`,
  `gdn_ab_multi`, `fused_gdn_step_norm`, `gdn_step_norm_multi` verify et commit) dans une capture
  `cudaStreamCaptureModeThreadLocal` : les requêtes uniques de `launch_grid.hpp` (`cudaGetDevice`,
  `cudaDeviceGetAttribute`, `cudaFuncGetAttributes`, `cudaOccupancyMaxActiveClusters`) et le
  `cudaLaunchKernelEx` en cluster sont donc capturés, comme dans le moteur. Le graphe est instancié et rejoué deux
  fois ; ses octets sont comparés à un lancement direct, aux anciens lancements (`STRATA_OLD_*`), à l'hôte (copies)
  et à la référence double (état du commit).

### Registres (ptxas, sm_120, `-O3`), aucun débordement

| Kernel | Registres | Mémoire partagée | Ancien |
|---|---|---|---|
| `gdn_ab_row_kernel` | 62 | dynamique, 15 Ko (2 560) | `gdn_ab_kernel` : 40 |
| `gdn_ab_multi_row_kernel` | 40 | dynamique, 5 Ko | `gdn_ab_multi_kernel` : 48 |
| `gdn_step_norm_cluster_kernel` | 128 | 1 552 o | `gdn_step_norm_kernel` : 128, 3 136 o |
| `gdn_step_norm_multi_cluster_kernel` | 126 | 3 200 o | `gdn_step_norm_multi_kernel` : 126, 3 136 o |

128 registres × 128 threads : 4 blocs par SM au plus ; les 192 blocs du pas tiennent en une vague sur la 5070
(48 SM) comme sur la 5090.

## Ce qui reste (non fait, et pourquoi)

- **`qsa_decode_attn` (66 blocs)** : la grille vient des morceaux de 64 cellules × 2 têtes KV ; plus de blocs =
  morceaux plus petits = autre ordre de fusion softmax : change la numérique (donc opt-in), et le fichier relève du
  lot attention/QSA. Non touché.
- **`gr_down_kernel` (41 blocs, `fused_gr.cu`)** : même motif que l'ancien `gdn_ab` (un warp par ligne de 10 240,
  40 blocs de 8 lignes, 6,5 Mo de poids). Un étage en mémoire partagée par 4 warps serait identique au bit près, mais
  chaque bloc refait la norme d'entrée (80 Ko lus) : à mesurer. Hors des fichiers de ce lot.
- **prefill `gdn_rec_kernel` (48 blocs)** : c'est le prefill (P3 de l'audit), pas le décodage.
- Revus sans changement : `mtp_select` (16 blocs, 40 Ko copiés), `copy_from_mapped` et `copy_indexed` (≤ 64 blocs,
  bornés par le PCIe ou minuscules), `gdn_conv_l2` (80 blocs de 128), l'attention courte native (un bloc par tête).

## Gains attendus

- **ESTIMÉ (audit)** : 0,2 à 0,6 ms par tour sur la 5090 (2 à 4 %) pour l'ensemble d'O7, `qsa_decode_attn` et
  `gr_down` compris ; rien sur la 5070.
- **HYPOTHÈSE (ce lot)** : 0,1 à 0,25 ms par tour avec `--spec 4`, soit environ 1 à 2 % :
  - `gdn_ab_multi` : ~3 allers-retours DRAM → 1 : 1 à 2 µs × 36 couches GDN par fenêtre ;
  - `gdn_step_norm_multi` : lecture de l'état (verify) et lecture + écriture (commit) sur ~170 SM au lieu de 48,
    et 2 barrières de bloc de moins par jeton : 0,5 à 2 µs × 36 × 2 lancements par tour ;
  - `fetch_blobs` : ~0 (borné par le PCIe), voir le point 1.
- La 5070 profite aussi du point 2 (un aller-retour DRAM au lieu de ~3, indépendant du nombre de SM) ; les points 1
  et 3 y sont neutres (point 1 identique ; point 3 : même nombre de SM occupés, 2 barrières de cluster en plus).
- `grids_parity --bench` donne le temps par lancement ancien/nouveau de chaque kernel, avec et sans L2 froid.

## Validation sur la RTX 5090

```bash
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON \
      -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_GGML_DIR=/opt/llama.cpp
cmake --build build -j

# 1. parité : tout doit être « ok » et la dernière ligne « PASS (0 failed) »
./build/grids_parity
./build/grids_parity --capture            # premiers lancements dans une capture de graphe
#    vérifier l'en-tête : « 170 SMs, clusters yes », « copy grids: 1360 blocks »,
#    « cluster: fused_gdn_step_norm yes, gdn_step_norm_multi yes » (sinon le chemin cluster n'a pas tourné)
(cd build && ctest --output-on-failure -R "grids_parity|gdn_parity|elementwise_parity")   # grids_parity et grids_parity_capture

# 2. micro-banc : ancien contre nouveau, par kernel
./build/grids_parity --bench
```

A/B de bout en bout : `ARGS` = les arguments de votre configuration (ceux écrits par `setup.py` : `--pack …
--native … --ple-gguf … --expert-profile … --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp …
--max-context …`). Trois prompts tokenisés `p1.txt p2.txt p3.txt` (`--tokens-file`).

```bash
OLD="STRATA_OLD_GRIDS=1 STRATA_OLD_GDN_AB=1 STRATA_OLD_GDN_STEP=1"
for p in p1 p2 p3; do
  for run in 1 2 3; do
    ./build/strata $ARGS --tokens-file $p.txt --max-new 256 --greedy --adapt-every 100000 > new_${p}_$run.log
    env $OLD ./build/strata $ARGS --tokens-file $p.txt --max-new 256 --greedy --adapt-every 100000 > old_${p}_$run.log
  done
  # 1. tokens identiques (les trois changements sont exacts)
  diff <(grep '^output' new_${p}_1.log) <(grep '^output' old_${p}_1.log) && echo "$p: tokens identiques"
done
# 2. débit
grep -H '^decode' new_*.log old_*.log
```

À vérifier :
1. `output  :` identique entre `new_*` et `old_*` pour chaque prompt. Une différence est un défaut (tous les
   changements sont exacts) : relancer avec une seule variable `STRATA_OLD_*=1` pour trouver lequel, et le
   signaler avec la sortie de `./build/grids_parity`.
2. `decode … tok/s` : new ≥ old (moyenne des 3 passages par prompt).
3. Chaque point seul : `new` avec une seule variable parmi `STRATA_OLD_GRIDS=1`, `STRATA_OLD_GDN_AB=1`,
   `STRATA_OLD_GDN_STEP=1`, pour isoler son effet (le point 1 devrait être neutre).
4. Chemin jeton (kernels `fused_gdn_*`) : la même boucle avec `--spec 0` à la place de `--spec 4` si votre pack le
   permet (un pack natif exige `--spec` ≥ 2 ; sinon, seul `grids_parity` couvre ces deux kernels).
5. Aucun message `(cluster; STRATA_OLD_GDN_STEP=1 avoids it)` sur stderr.

Profil (facultatif) : `nsys profile --cuda-graph-trace=node ./build/strata $ARGS --tokens-file p1.txt --max-new 64
--greedy --adapt-every 100000`, puis comparer la durée de `gdn_ab_multi_row_kernel` / `gdn_ab_multi_kernel` et de
`gdn_step_norm_multi_cluster_kernel` / `gdn_step_norm_multi_kernel` ; `ncu --set full -k regex:gdn_` sur quelques
lancements pour l'occupation et le débit DRAM.

## Fichiers

- `include/strata/kernels/launch_grid.hpp` (nouveau) : `grid_device()`, `grid_sms()`, les trois interrupteurs
  `STRATA_OLD_*`, `cluster_launchable()`.
- `src/kernels/cuda/verify_kernels.cu` : grilles de `fetch_blobs` / `gather_rows`, `gdn_ab_multi_row_kernel`,
  `gdn_step_norm_multi_cluster_kernel` et leurs lanceurs.
- `src/kernels/cuda/fused_gdn.cu`, `include/strata/kernels/fused_gdn.hpp` : `gdn_ab_row_kernel`,
  `gdn_step_norm_cluster_kernel` et leurs lanceurs.
- `src/kernels/grids_parity.cpp`, `CMakeLists.txt` : le test (ctest `grids_parity` et `grids_parity_capture`).

Tests lancés ici (sans GPU) : compilation complète (sm_120) sans avertissement ; `verify_kernels.cu` et
`fused_gdn.cu` compilés aussi pour sm_86 ; `ple_reader_selftest`, `platform_memory_test`, `pool_stress`,
`expert_multi_test`, `suffix_drafter_test`, `draft_policy_test`, `controller_test`, `conv_cache_test` : OK ;
`pool_test` : échec attendu (pas de pack de modèle) ; `grids_parity` et `grids_parity_capture` : échec attendu
(pas de pilote CUDA) ;
`python3 serve/test_server.py` : 26 tests OK.

## Intégration (`perf/all`)

Avec `research` (#53, `--gdn-state-bf16`), les kernels en cluster du pas GDN (`gdn_step_norm_cluster_kernel`,
`gdn_step_norm_multi_cluster_kernel`) sont des templates sur le type de stockage de l'état, comme les kernels d'un
bloc par tête : l'état BF16 garde le lancement en cluster. L'instanciation FP32 est le kernel décrit ici, inchangé ;
`STRATA_OLD_GDN_STEP=1` rend l'ancien kernel dans les deux cas. Voir [../PERF-CHANGES.md](../PERF-CHANGES.md).
