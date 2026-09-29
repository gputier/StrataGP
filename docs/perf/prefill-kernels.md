# prefill-kernels : les kernels de la lecture du prompt (issues #37, #38, #43, #6, #7)

Branche `perf/prefill-kernels`, partie de `perf/base` (b858f9b, moteur 0.1.20). **Aucun chiffre GPU n'a été
mesuré** : la machine de développement n'a pas de GPU. Tout a été compilé (CUDA 13.0, sm_120), les tests CPU ont été
lancés, et le nouveau test GPU `prefill_kernels_parity` compile mais reste à lancer sur la RTX 5090.

Ce qui a été vérifié sans GPU :
- **le SASS** (`cuobjdump -sass`) de chaque nouveau kernel contre l'ancien : mêmes instructions flottantes dans le même
  ordre (voir le détail par point) ;
- **une émulation CPU** des anciens et des nouveaux kernels (le source CUDA exécuté avec un thread système par thread
  CUDA, `__syncthreads` et les shuffles émulés, des pauses aléatoires après chaque barrière) : sorties identiques au
  bit sur des morceaux enchaînés de 1 à 33 tokens (conv), 1 à 5 (récurrence) et 1 à 3 (GR). Le même banc détecte une
  variante volontairement fausse (ordre de somme inversé, double tampon au lieu de triple). Cet outil n'est pas
  versionné.

## Ce qui change

| # | Issue | Changement | Par défaut | Ancien chemin / activation | Résultat |
|---|---|---|---|---|---|
| P2 | #37 | `gdn_conv` : un thread par (canal, token) au lieu d'un thread par canal qui parcourt le morceau ; la norme L2 des têtes q/k dans le même kernel ; un petit kernel écrit l'historique | **activé** | `STRATA_OLD_PREFILL_GDN_CONV=1` | identique au bit |
| P3 | #38 | `gdn_recurrence` : 4 blocs par tête de valeur (192 au lieu de 48), sommes partielles par shuffles, 1 barrière par token au lieu de 6, chaîne kv(t+1) en parallèle de o(t), entrées chargées 2 tokens à l'avance + préchargement L2 à 8 tokens ; la norme de sortie dans un second kernel, **dans l'ordre exact de l'ancien** | **activé** | `STRATA_OLD_PREFILL_GDN_REC=1` | identique au bit |
| P8 | #43 | chaîne GR : échelle RMS par ligne au lieu de `xn` en FP32, `xn` recalculé dans `gr_mix`, `gr_write(h)` fusionné avec `gr_norm(h+1)` ; le tampon `xn` passe de T×40 Ko à T×16 o | **activé** | `STRATA_OLD_PREFILL_GR=1` (rétablit aussi le tampon FP32) | identique au bit |
| B7 | #6 | les images FP16 du prompt saturent à ±65 504 au lieu de passer à ±inf | désactivé | `STRATA_PREFILL_F16_SAT=1` | change seulement les anciens ±inf |
| B8 | #7 | le routeur du prompt prend celui du décodage (`native_router` en fast-math avec `--native-router`, sinon `router_top10`) | désactivé | `STRATA_PREFILL_ROUTE_DECODE=1` | peut changer la sélection près d'une égalité |
| B8 | #7 | `STRATA_DBG_NAN=1` compte aussi les lignes du routeur dont un logit est non fini | (mode debug) | — | aucun effet hors debug |

Les trois réécritures (P2, P3, P8) gardent chaque opération flottante et son ordre : elles sont donc actives par
défaut, l'ancien chemin restant sélectionnable pour l'A/B. B7 et B8 modifient des résultats : désactivés par défaut.
Toutes les variables se lisent une fois au démarrage ; `=1` active, absente ou `=0` n'active pas.

### P2 — `gdn_conv` (#37), `src/prefill/kernels.cu`

- `gdn_conv_tile_kernel` : bloc = (une tête de 128 canaux, 16 tokens), grille 80 × ⌈T/16⌉ (40 960 blocs à T = 8 192
  au lieu de 80). Chaque thread charge une fois les 19 entrées `t0-3 … t0+15` (l'historique pour t < 0), calcule les
  16 sorties, puis, pour les 32 têtes q/k, la norme L2 (somme de warp, 4 sommes partielles, `rsqrt`).
- `gdn_conv_hist_kernel` : les trois dernières entrées du morceau (les anciennes entrées de l'historique si T < 3),
  lancé après, sur le même flux (tous les blocs ont lu l'ancien historique).
- Bit à bit : même expression `v0*w0 + v1*w1 + v2*w2 + x*w3` (SASS : `FMUL w1·v1`, puis `FFMA` w0, w2, w3, comme
  l'ancien), même SiLU (`__expf`, division IEEE), même carré non fusionné avant la somme de warp, même ordre
  `part0+part1+part2+part3+eps`.
- Trafic : h n'est plus relu ni réécrit par le kernel L2 séparé (40 % de h), et les entrées sont lues 19/16 fois.
- Registres : 48, 0 spill.

### P3 — `gdn_recurrence` (#38), `src/prefill/kernels.cu`

- `gdn_rec_split_kernel`, grille (48 têtes, 4 quarts de 32 colonnes), 128 threads. Un warp tient 8 colonnes × les
  mêmes 4 groupes de 32 lignes que l'ancien (lane = 4 × colonne + groupe) : `kv` et `o` sont les quatre mêmes sommes
  partielles (chaînes FMA dans l'ordre des lignes), additionnées dans l'ordre `g0+g1+g2+g3` par `__shfl_sync` au
  lieu de la mémoire partagée.
- q|k d'un token passent par la mémoire partagée, trois tampons (le token t+1 est écrit avant la barrière du token
  t) : **une barrière par token** (l'ancien en avait six), et la chaîne `kv(t+1)` (état mis à jour contre `k(t+1)`)
  tourne en même temps que la chaîne `o(t)` — les mêmes opérations dans le même ordre, faites un token plus tôt.
  Pas de conflit de banc (groupes au pas de 36 flottants, lectures `LDS.128`).
- Les entrées (q, k, v, gate, beta) sont chargées **deux tokens à l'avance**, et leurs lignes de cache demandées
  **huit tokens à l'avance** (`prefetch.global.L2`) : à 8 192 tokens, h fait 335 Mo, bien plus que le L2 ; l'ancien
  kernel faisait jusqu'à trois lectures globales en série par token, chacune après une barrière.
- `o × rsqrt(128)` va dans `y` ; `gdn_out_norm_kernel` (un bloc par (token, tête), 128 threads) applique la norme :
  même papillon par warp de 32 colonnes, mêmes 4 sommes de warp dans le même ordre, puis
  `o * rsqrt(ss/128 + eps) * gamma * sigmoid(z)`, `y` et `y16`. **L'ordre de réduction de la norme est donc
  conservé** : la variante opt-in `STRATA_PREFILL_GDN_SPLIT` prévue pour un ordre différent n'a pas lieu d'être.
- SASS vérifié : `v - g*kv` fusionné en `FFMA -kv, g, v` comme l'ancien (fusion faite par ptxas), `rsqrt(128)` plié
  en la même constante `0.0883883461…`, carré non fusionné (`__fmul_rn`, l'ancien passait par un `select`),
  `ss*(1/128)+eps` en `FFMA` comme l'ancien.
- Registres : 72, 0 spill ; 3,4 Ko de mémoire partagée. Norme : 22 registres.

### P8 — la chaîne GR (#43), `src/prefill/kernels.cu`, `src/prefill/prefill.cpp`

- `gr_norm_scale` : l'arithmétique de `gr_norm` (même boucle, même `block_sum` à 256 threads) ; écrit l'image BF16
  et l'échelle `rs` de chaque ligne (token, flux), sans l'image FP32.
- `gr_mix_scale` : recalcule `xn = (R * rs) * w` — le produit que `gr_norm` stockait, bit pour bit — puis le même
  `fmaf(xn, sigmoid(g), s)` sur les 4 flux.
- `gr_write_norm` : la mise à jour de `gr_write` (`fmaf(bo, 2·sigmoid(inj/4), R)`), gardée en registres, écrite,
  puis normalisée avec le poids de la moitié suivante, avec la même correspondance thread → élément que `gr_norm`
  (donc la même somme de carrés). Utilisée quand rien d'autre n'écrit R entre les deux : toujours dans une couche ;
  entre deux couches, sauf si le vecteur de contrôle couvre la couche ou si la couche suivante est la couche 1 avec
  le bloc PLE (dernière couche : `gr_write` seul).
- `m.xn` ne garde que les échelles : T×4 flottants au lieu de T×10 240 (−335 Mo à T = 8 192), dans `carve` et dans
  `bytes_needed` (le prompt emprunte donc moins d'emplacements du cache d'experts).
- Économie : l'écriture FP32 de `xn` (40 Ko par token et par moitié) et une lecture de R (40 Ko) sur ~290 Ko.

### B7 — FP16 saturé, opt-in (#6)

- `hf()` (toutes les images FP16 de `kernels.cu` : SwiGLU des experts et de l'expert partagé, `y_h`, `attn_h`,
  `mixed_h`, K/V FP16 et échelles INT8, `to_f16`) : avec `STRATA_PREFILL_F16_SAT=1`, une valeur finie au-delà de la
  plage FP16 devient ±65 504 ; NaN et ±inf passent tels quels (un débordement FP32 en amont reste visible). Toute
  valeur qui était finie garde ses bits.
- Mécanisme : un `__constant__` lu par `hf` (une comparaison prédiquée), écrit une fois avant le premier kernel
  concerné, par une copie synchrone suivie d'une synchronisation du GPU (visible quel que soit le flux). Désactivé : `hf` est exactement `__float2half_rn`.
  Le `__constant__` existe une fois par GPU : avec la répartition des couches (0.1.21), il est écrit une fois sur
  chaque carte qui lit une partie du prompt (un `std::once_flag` par GPU).

### B8 — routeur aligné sur le décodage, opt-in ; NaN comptés (#7)

- `STRATA_PREFILL_ROUTE_DECODE=1` : `route` appelle `route_decode`, qui fait le choix du décodage
  (`core/layer.cpp`) : `route_native` avec `--native-router` (activé par `--native`) et 512 experts, sinon
  `router_top10` sur le morceau.
- `route_native` (`src/prefill/route_native.cu`, compilé en `--use_fast_math` comme `native_router.cu`) : le kernel
  du routeur natif recopié, un warp par token. Le SASS contient le même multiensemble de 616 instructions
  flottantes (`FADD.FTZ`, `MUFU.EX2`, `MUFU.RCP`…) que `native_router_top10` ; le test le compare au bit près.
- `STRATA_DBG_NAN=1` : la ligne `strata dbg: layer …` compte aussi `router rows N`, les tokens dont un logit du
  routeur est non fini (le routeur remplace une probabilité NaN par −FLT_MAX ; une ligne entièrement NaN choisit les
  experts 0 à 9 sans rien dire).

## Gains attendus (aucun n'est mesuré)

| Point | Estimation | Raisonnement |
|---|---|---|
| P2 | **1 à 3 %** du prefill sur 5090 (HYPOTHÈSE ; l'audit disait 2 à 5 %) | ancien : 10 240 threads liés à la latence, ~1 à 3 ms par couche GDN à T = 8 192 ; nouveau : limité par la bande passante, ~0,5 ms ; × 36 couches GDN = 20 à 90 ms par morceau |
| P3 | **5 à 15 %** (HYPOTHÈSE ; l'audit disait 5 à 10 %) | ancien : ~1 à 2 µs par token (6 barrières à 512 threads, trois attentes mémoire en série), soit 8 à 16 ms par couche ; nouveau : ~0,15 à 0,25 µs par token (une chaîne de 32 FMA sur le chemin critique, une barrière) + ~0,3 ms de norme |
| P8 | **~1 %** (ESTIMÉ ; l'audit disait 2 à 3 %) | 96 moitiés × 8 192 tokens × 80 Ko ≈ 64 Go de moins par morceau, ~40 ms à 1,5 To/s ; plus 335 Mo de VRAM rendus au cache d'experts pendant le prompt |
| B7, B8 | aucun | corrections de robustesse / cohérence |

Ensemble P2 + P3 + P8 : **de l'ordre de 10 % du temps de lecture d'un long prompt** sur la 5090 (HYPOTHÈSE), puisque
ces parties séquentielles ne profitent pas de la bande passante de la carte. `prefill_kernels_parity --bench` donne
le gain par kernel sans le modèle ; `STRATA_PREFILL_TIMING=1` donne la part de chaque phase dans le moteur.

## Validation sur la RTX 5090

### 1. Construction et tests

```bash
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON \
      -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_GGML_DIR=/opt/llama.cpp
cmake --build build -j

./build/prefill_kernels_parity            # doit finir par « prefill_kernels_parity: OK »
./build/prefill_kernels_parity --bench    # ms par appel, ancien -> nouveau, à T = 8192 (--bench 4096 pour un autre T)
(cd build && ctest --output-on-failure -R "prefill_kernels_parity|gdn_parity|gr_parity|router_top10_parity")
```

`prefill_kernels_parity` compare, **au bit près** : `gdn_conv` (h et historique, T = 1 à 257, deux morceaux
enchaînés), `gdn_recurrence` (y, y16, état ; T = 1 à 130, deux morceaux), la chaîne GR (xn16, mixed FP32/BF16/FP16,
R écrit, puis la moitié suivante), la saturation FP16 (valeurs limites, SwiGLU saturé), `route_native` contre
`native_router_top10` (égalités, ligne plate, probabilités sous-normales) et l'aiguillage de `route_decode`. Il
compare aussi `gdn_conv` et `gdn_recurrence` à une référence hôte en double (tolérance large) et affiche combien de
lignes le routeur par défaut choisit autrement que le routeur natif (information, pas un échec).

### 2. A/B de bout en bout (chemins par défaut contre anciens chemins)

`ARGS` = les arguments de votre configuration (ceux écrits par `setup.py`, avec `--prefill auto`). Trois prompts
tokenisés longs `p1.txt p2.txt p3.txt` (`--tokens-file`), par exemple ~6K, ~16K et ~32K tokens (plusieurs morceaux de
8 192), comme le prompt de 32K de `bench/results/2026-09-28-prefill-speed`.

```bash
OLD="STRATA_OLD_PREFILL_GDN_CONV=1 STRATA_OLD_PREFILL_GDN_REC=1 STRATA_OLD_PREFILL_GR=1"
DBG="STRATA_PREFILL_TIMING=1 STRATA_STATE_HASH_GDN=1"
for p in p1 p2 p3; do
  for run in 1 2 3; do
    env $DBG STRATA_PREFILL_DUMP_R=new_$p.r ./build/strata $ARGS --tokens-file $p.txt \
        --max-new 256 --greedy --adapt-every 100000 > new_${p}_$run.log 2> new_${p}_$run.err
    env $OLD $DBG STRATA_PREFILL_DUMP_R=old_$p.r ./build/strata $ARGS --tokens-file $p.txt \
        --max-new 256 --greedy --adapt-every 100000 > old_${p}_$run.log 2> old_${p}_$run.err
  done
  cmp new_$p.r old_$p.r && echo "$p: résidus finaux identiques au bit"
  diff <(grep GDN_HASH new_${p}_1.err) <(grep GDN_HASH old_${p}_1.err) && echo "$p: états GDN identiques"
  diff <(grep '^output' new_${p}_1.log) <(grep '^output' old_${p}_1.log) && echo "$p: tokens identiques"
done
grep -H '^prefill' new_*.log old_*.log                  # tok/s du prompt : new > old attendu
grep -H 'strata generate: prefill' new_*.err old_*.err  # tok/s, experts streamés / résidents
grep -H 'strata prefill timing' new_*.err old_*.err     # ms par phase : « gdn » et « hc read » doivent baisser
```

PowerShell (Windows), un passage :

```powershell
$A = @('--pack', '...')   # les arguments de votre configuration (setup.py), un élément par argument
$env:STRATA_PREFILL_TIMING = 1; $env:STRATA_STATE_HASH_GDN = 1
.\build\strata.exe @A --tokens-file p1.txt --max-new 256 --greedy --adapt-every 100000 > new_p1.log 2> new_p1.err
$env:STRATA_OLD_PREFILL_GDN_CONV = 1; $env:STRATA_OLD_PREFILL_GDN_REC = 1; $env:STRATA_OLD_PREFILL_GR = 1
.\build\strata.exe @A --tokens-file p1.txt --max-new 256 --greedy --adapt-every 100000 > old_p1.log 2> old_p1.err
Remove-Item Env:STRATA_OLD_PREFILL_*
```

À vérifier :
1. `prefill_kernels_parity: OK`.
2. Pour chaque prompt : fichiers `STRATA_PREFILL_DUMP_R` identiques (`cmp`), lignes `GDN_HASH` identiques, `output :`
   identique. Une différence veut dire que le compilateur de la machine a contracté autrement l'ancien ou le nouveau
   code (le test de parité le montre alors aussi) : garder l'ancien chemin concerné et le signaler.
3. `prefill … tok/s` : new > old ; `decode … tok/s` inchangé (le décodage n'est pas touché).
4. `strata prefill timing` : la phase `gdn` baisse nettement (P2 + P3) ; `hc read` baisse (P8). La norme fusionnée de
   la moitié suivante est désormais comptée dans la phase qui précède l'écriture (`combine`, `gdn`, `qsa attn`) :
   comparer le total.
5. Chaque point seul : relancer avec une seule variable `STRATA_OLD_PREFILL_*=1`.
6. VRAM : avec le nouveau chemin, `bytes_needed` est plus petit de T×40 Ko ; la ligne `strata generate: prefill`
   peut montrer davantage d'experts résidents (moins d'emplacements empruntés).

### 3. Les deux options (B7, B8)

```bash
# B7 : combien de valeurs non finies sur des prompts longs, avec et sans saturation
env STRATA_DBG_NAN=1 ./build/strata $ARGS --tokens-file p3.txt --max-new 256 --greedy --adapt-every 100000 2> sat0.err
env STRATA_DBG_NAN=1 STRATA_PREFILL_F16_SAT=1 ./build/strata $ARGS --tokens-file p3.txt --max-new 256 --greedy \
    --adapt-every 100000 2> sat1.err
grep 'strata dbg' sat0.err sat1.err     # lignes « non-finite … router rows … » et « prompt end »
```
Sans ligne `strata dbg: layer` dans `sat0.err` et avec `output :` identique entre les deux passages, rien n'a
débordé sur ces prompts et B7 n'y change rien. S'il y a des lignes, `sat1.err` doit en montrer moins (ou aucune) et la
sortie doit rester sensée ; la ligne `prompt end` donne l'état laissé au décodage.

```bash
# B8 : le routeur du prompt = celui du décodage (avec --native, donc --native-router)
env STRATA_PREFILL_ROUTE_DECODE=1 ./build/strata $ARGS --tokens-file p3.txt --max-new 256 --greedy \
    --adapt-every 100000 > route1.log 2> route1.err
```
La sortie peut différer (sélection près d'une égalité) : c'est l'effet voulu. Pour juger la qualité, la mesure
forcée par l'enseignant de `bench/results/2026-09-28-prefill-speed` (`--prefill-until N-256`, `--dump-logits`, même
top-1 / KL moyen / perplexité contre la référence) avec et sans la variable ; attendu : un écart bien plus petit que
le « yardstick » d'un changement de taille de morceau (KL 0,33). `prefill_kernels_parity --bench` donne le coût du
routeur natif par lot (négligeable devant les GEMM).

## Ce qui reste / risques

- **Rien n'a tourné sur GPU.** L'identité au bit repose sur le SASS vérifié ici (CUDA 13.0) : un autre `nvcc` pourrait
  contracter autrement une expression de l'ancien ou du nouveau code. Les nouveaux kernels écrivent les mêmes
  expressions sous la même forme que les anciens (et `__fmul_rn` là où l'ancien code ne fusionnait que par
  structure) ; `prefill_kernels_parity` et l'A/B ci-dessus le confirment ou non sur la machine.
- P3 : la forme par morceaux « WY » (style FLA, sur tenseurs) de l'issue reste à faire ; elle changerait l'ordre des
  calculs (opt-in, validation forcée par l'enseignant). La récurrence actuelle reste séquentielle en T : son coût par
  token est maintenant d'environ une chaîne de 32 FMA.
- P8 : la fusion entre couches est désactivée devant le bloc PLE de la couche 1 et après une couche couverte par le
  vecteur de contrôle ; tout nouveau code qui écrirait R entre deux moitiés devra faire de même (condition au point
  d'écriture dans `prefill.cpp`).
- B7 reste désactivé par défaut : il ne change que des valeurs qui étaient ±inf, mais il change des sorties.
- B8 : seul le chemin du prompt est aligné ; le routeur `route_kernel` par défaut est inchangé.
- Le test CPU `ple_reader_selftest` a échoué une fois sur trois ici : il utilise un fichier fixe
  `/tmp/ple_reader_selftest.bin`, partagé par les copies de travail qui compilaient en même temps (sans lien avec
  cette branche ; il passe seul).
