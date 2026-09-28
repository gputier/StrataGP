# Experts VRAM au format Q2_0 : kernels groupés et par hit (O3, #17)

Branche `perf/grouped-experts`, partie de `perf/base` (moteur 0.1.20). Fichiers modifiés :
`src/kernels/cuda/s2_expert_grouped.cu`, `include/strata/kernels/s2_expert_grouped.hpp`, et un test,
`src/kernels/s2_expert_grouped_parity.cpp` (enregistré dans `CMakeLists.txt`).

**Portée.** Ces kernels calculent les experts résidents en VRAM des packs **Q2_0 canoniques**. Les packs natifs (IQ)
passent par `native_expert_grouped` (`iq_kernels.cu`), qui n'est pas modifié ici. Le prefill (MMQ) ne l'est pas non plus.

## Ce qui change

### 1. Kernels groupés : `moe_grouped_s2` (fenêtre de vérification `verify.cpp`, couche MTP `mtp.cpp`)

Les anciens `gu_grouped_kernel` et `down_grouped_kernel` sont remplacés par `gu_grouped_t_kernel` et
`down_grouped_t_kernel`.

- **Plus de conflits de banques.** Les activations d'un groupe sont placées en mémoire partagée en disposition
  « mot d'abord » : `xs_w[j][k * NC + c]` (mot `j` du morceau `c` de l'entrée `k`). Avant, `xs_q[k][c * 8 + j]` faisait
  lire à la voie `L` le mot `8L + j` : un conflit d'ordre 8 sur chacune des 8 lectures d'un morceau. Désormais, les
  voies lisent des mots consécutifs. L'écriture au placement l'est aussi, car l'indice plat `i` vaut exactement
  `k * NC + c`. Aucun bourrage n'est donc nécessaire. C'est une variante de la proposition de l'issue
  (`[GMAX][8][H/32+1]`) qui supprime aussi les conflits à l'écriture.
- **`hx` calculé une seule fois.** La somme des 32 int8 d'un morceau est calculée une fois par (entrée, morceau) au
  placement. Elle est stockée avec `dx` dans `xs_dh[k * NC + c]` (une lecture 64 bits par morceau). Avant, elle était
  recalculée avec 8 `dp4a` pour chaque ligne.
- **Codes développés avec moins d'opérations.** Au placement, les activations sont regroupées en
  `X[4h+f] = {x[16h+f], x[16h+4+f], x[16h+8+f], x[16h+12+f]}` (8 `__byte_perm`). Chaque mot de codes se développe alors
  par `(w >> 2f) & 0x03030303`, soit 7 opérations pour 16 éléments au lieu de 7 par octet. Le développement est fait
  une seule fois par ligne, hors de la boucle sur les entrées.
- **Deux lignes par passe de warp.** Chaque warp prend deux lignes adjacentes à la fois : porte `r` et montée `r` pour
  gate/up, lignes `r` et `r+1` pour down. Chaque lecture en mémoire partagée sert ainsi aux deux lignes.
- **Placement en mots alignés.** Le placement lit 9 mots alignés (décalage en entonnoir) au lieu de 32 octets
  (voir § 2).

### 2. Kernels par hit : `moe_hit_grouped_s2`, `_dev` (graphe « token », `session.cpp`), `_multi`

Les anciens `gu_kernel` et `down_kernel` sont remplacés par `gu_pair_kernel` et `down_pair_kernel`.

- **Lectures larges.** Le PTX de l'ancien kernel montrait 44 `ld.global.nc.u8` par itération. Désormais :
  - les codes se lisent en `uint2` ;
  - les échelles en 16 bits ;
  - les 32 int8 d'un `block_q8_0` (34 octets, donc jamais alignés sur 4 octets pour un morceau pair) se lisent en
    **9 mots alignés** puis un `__funnelshift_r`.

  Le 9ᵉ mot n'est lu que si le décalage vaut 2 octets. Il se trouve alors dans le bloc suivant de la même ligne, car
  les lignes font 80 ou 20 morceaux et un morceau pair n'est donc jamais le dernier.
- **Un warp par paire de lignes.** Une paire (porte `r`, montée `r`) pour gate/up, deux lignes adjacentes pour down.
  Un morceau d'activation est chargé et regroupé une fois pour les deux lignes. `hx` est calculé une fois par morceau.

### Pourquoi le résultat est identique au bit près

- `s = Σ code·x` et `hx = Σ x` sont des sommes entières exactes, avec |s| ≤ 32·3·128. Le regroupement des produits en
  mots `dp4a` ne change donc pas ces entiers.
- L'expression flottante est inchangée : `acc += dw * dx * (float) (s - hx)`, avec le même ordre des morceaux par voie
  (`c = lane, lane+32, lane+64`) et le même arbre de `__shfl_down_sync` (16, 8, 4, 2, 1).
- Le PTX le confirme. Ancien et nouveau kernel émettent, par morceau et par ligne, `mul.f32` (dw·dx), `cvt.rn.f32.s32`
  (s − hx) et `fma.rn.f32` sur l'accumulateur, puis la même réduction `add.f32`. Aucun `mul.f32` n'alimente un
  `add.f32` que ptxas pourrait contracter autrement.

## Activé par défaut, et comment revenir à l'ancien chemin

| Élément | Par défaut | Retour à l'ancien |
|---|---|---|
| Kernels groupés (`moe_grouped_s2`) | **activé** | `STRATA_OLD_GROUPED=1` |
| Kernels par hit (`moe_hit_grouped_s2`, `_dev`, `_multi`) | **activé** | `STRATA_OLD_GROUPED=1` |

- Le changement est identique au bit près par construction ; il n'y a donc pas d'option d'activation.
- La variable d'environnement est lue une seule fois, au premier appel ; la changer ensuite dans le même processus
  n'a aucun effet. Le choix du kernel, lui, est refait à chaque lancement : un graphe capturé garde les kernels
  choisis à sa capture. Pour basculer en cours d'exécution, utiliser `moe_grouped_select_old`.
- `STRATA_GROUPED_PAIR_MIN_HITS=N` (lue une fois, défaut 0) garde les anciens kernels **par hit** en dessous de N hits
  de capacité. Les nouveaux kernels par hit lancent deux fois moins de warps (deux lignes par warp) : pour 1 hit, le
  gate/up tient en 80 blocs, moins que les 170 SM d'une RTX 5090. Si `--bench` montre une perte à 1-3 hits, régler
  N (par exemple 4) ; le résultat reste identique au bit près. Ignorée quand `moe_grouped_select_old` force un choix.
- En code, `moe_grouped_select_old(1 | 0 | -1)` force l'ancien kernel, force le nouveau, ou revient à l'environnement.
  Le test de parité s'en sert pour exécuter les deux versions dans un même processus.
- **Repli automatique** sur l'ancien kernel si les lectures larges ne sont pas possibles : ligne d'activation ou
  `scratch` non aligné sur 4 octets, ou arène de blobs / taille de blob non alignée sur 8 octets (chemin par hit).
  Le moteur est toujours aligné. Le test couvre ce repli avec une ligne décalée de 2 octets, un pas de slot de
  `BLOB + 4` octets et une arène décalée de 4 octets. Un `scratch` non aligné n'est pas testé : l'ancien kernel y
  écrit des floats, ce n'est donc pas une entrée valide.
- `moe_grouped_last_path()` indique quels kernels le dernier appel a lancés (1 nouveau, 0 ancien) et se remet à -1.
  Le test de parité vérifie à chaque exécution que le chemin forcé a bien été pris : un nouveau chemin qui se
  désisterait en silence ne peut plus passer pour « ancien contre ancien ».

## Ressources (ptxas, sm_120, `-O3`)

| Kernel | Registres | Mémoire partagée | Spills |
|---|---|---|---|
| `gu_kernel` (ancien) | 39 | 0 | 0 |
| `gu_pair_kernel` | 40 | 0 | 0 |
| `down_kernel` (ancien) | 40 | 0 | 0 |
| `down_pair_kernel` | 40 | 0 | 0 |
| `gu_grouped_kernel` (ancien) | 63 | 23 040 o | 0 |
| `gu_grouped_t_kernel` | 80 | 25 600 o | 0 |
| `down_grouped_kernel` (ancien) | 40 | 5 760 o | 0 |
| `down_grouped_t_kernel` | 40 | 6 400 o | 0 |

`gu_grouped_t_kernel` passe de 4 à 3 blocs de 256 threads par SM (24 warps, chacun avec 2 lignes en vol). Forcer 4 blocs
(`__launch_bounds__(256, 4)`) plafonne à 64 registres mais provoque 32 octets de spill ; ce n'est donc pas retenu.

## Gains attendus

Aucun chiffre n'a été mesuré sur GPU (pas de GPU dans l'environnement de développement). Les estimations ci-dessous
viennent du comptage des instructions dans le PTX.

**Kernels groupés.**
- Mémoire partagée, par (ligne, entrée) : l'ancien kernel fait environ 2,5 itérations × (8 lectures en conflit d'ordre
  8 + 1) ≈ **160 « wavefronts »**. Le nouveau fait environ 2,5 × (8 + 2) / 2 lignes ≈ **13**, soit environ 12× moins
  (ESTIMÉ).
- Travail entier par (ligne, entrée, morceau) : 16 `dp4a` et le développement des codes avant, 8 `dp4a` après.
- Le kernel devrait devenir limité par la DRAM, ou par le PCIe pour les blobs lus en mémoire hôte mappée (HYPOTHÈSE).
- L'audit estime le kernel **1,5 à 2× plus rapide** et le gain de bout en bout à **3 à 7 % (5070) et 4 à 8 % (5090)**
  en décodage spéculatif (ESTIMÉ, audit § O3).

**Kernels par hit.**
- Par warp et par itération, l'ancien kernel faisait 44 lectures d'un octet. Les 32 lectures d'activations touchent
  chacune environ 9 lignes de 128 o (32 blocs espacés de 34 o), soit environ 300 wavefronts L1.
- Le nouveau en fait environ 9 × 9 pour les activations, partagées par 2 lignes, plus environ 3 pour les codes et les
  échelles. Cela fait environ **45 par ligne au lieu de ~300** (ESTIMÉ).
- Si le kernel est limité par L1, comme ce modèle l'indique, le gain du kernel serait de 2 à 4× (HYPOTHÈSE, à confirmer
  avec `--bench` et `nsys`).
- Ce chemin sert au décodage token par token avec le graphe « token ». L'audit ne chiffre pas son effet de bout en bout.

## Validation déjà faite (sans GPU)

1. **Compilation.** Build complet vert en Release, CUDA 13.0, sm_120. Les registres et spills sont ci-dessus.
2. **Tests CPU.** Via ctest : `ple_reader_selftest`, `platform_memory_test`, `pool_stress`, `expert_multi_test`,
   `suffix_drafter_test`, `draft_policy_test`, `controller_test` et `conv_cache_test` passent. `pool_test` échoue
   comme attendu (pack de modèle absent).
3. **Émulation hôte.** Hors dépôt : les sources réelles `s2_expert_grouped.cu` et `quantize_act.cu`, avec la syntaxe
   `<<<>>>` réécrite, et le test `s2_expert_grouped_parity.cpp` lui-même ont été compilés pour le CPU avec une cale
   CUDA.
   - Chaque thread CUDA est une coroutine. `__syncthreads` et les shuffles sont des barrières en pas synchrone ; un
     shuffle divergent serait signalé comme interblocage.
   - Chaque lecture `uint2`, 16 bits et `__ldg` vérifie son alignement.
   - Résultat : **les 9 configurations du test sont identiques au bit près**, avec et sans contraction FMA (en
     contraction complète, comme NVVM). Les lignes au-delà du compte du chemin `_dev` restent intactes. La référence hôte en double est respectée partout (écart ≤ 1,3e-7 de Σ|terme|).
   - Six mutations volontaires sont toutes détectées : sélecteur `__byte_perm`, décalage oublié, mauvaise ligne écrite,
     indice de placement, échelle de la mauvaise ligne, `hx` oublié.

## À exécuter sur la RTX 5090

Les commandes supposent un build dans `build/` avec les tests (`-DSTRATA_BUILD_TESTS=ON`). Sous Windows avec un
générateur Visual Studio, les binaires sont dans `build\Release\`.

**1. Parité bit à bit (obligatoire) et outils de contrôle CUDA.**
```sh
ctest --test-dir build -R s2_expert_grouped_parity --output-on-failure
compute-sanitizer --tool memcheck  build/s2_expert_grouped_parity
compute-sanitizer --tool racecheck build/s2_expert_grouped_parity
compute-sanitizer --tool synccheck build/s2_expert_grouped_parity
```
Attendu : `bitwise identical` sur chaque ligne, `0 failures` / `PASS`, et aucune erreur des sanitizers.

**2. Temps des kernels, ancien contre nouveau (même binaire).**
```sh
build/s2_expert_grouped_parity --bench
```
La commande affiche µs par appel et Go/s effectifs pour :
- `moe_hit_grouped_s2` à 9, 5, 3, 2 et 1 hits (les petits nombres de hits correspondent aux configurations à faible
  résidence ; si le nouveau y est plus lent, voir `STRATA_GROUPED_PAIR_MIN_HITS` ci-dessus) ;
- `moe_grouped_s2` sur des fenêtres de 4 et 8 tokens avec 0 %, 30 % et 100 % d'experts partagés.

Les blobs tournent sur plus de 3× la L2.

**3. Parité existante du chemin expert** (inchangée, à relancer par précaution) :
```sh
ctest --test-dir build -R "expert_parity|native_expert_parity" --output-on-failure
```
`native_expert_parity` et `expert_parity` ont besoin du pack et du GGUF. Ils ne passent pas par ces kernels, mais l'audit
les demande pour O3.

**4. Bout en bout, A/B, glouton, 256 tokens, 3 prompts × 3 exécutions.** Reprendre les arguments du fichier de lancement
(`strata-<model>.json` : `--pack`, `--ple-gguf`, `--expert-cache`, `--spec`, `--mtp`…), sur un pack **Q2_0**, et
ajouter :
```sh
COMMON="<arguments du fichier de lancement> --adapt-every 100000 --greedy --max-new 256 --stats"
for p in prompt1.ids prompt2.ids prompt3.ids; do
  for r in 1 2 3; do
    build/strata $COMMON --tokens-file $p                      > new_${p}_$r.txt 2>&1
    STRATA_OLD_GROUPED=1 build/strata $COMMON --tokens-file $p > old_${p}_$r.txt 2>&1
  done
done
```
Sous PowerShell, remplacer le préfixe par `$env:STRATA_OLD_GROUPED=1` avant la commande, puis
`Remove-Item Env:STRATA_OLD_GROUPED`.

À comparer :
- les **tokens générés** (ligne `output  :`), qui doivent être identiques entre `old_*` et `new_*`, par exemple avec
  `grep -h "^output" old_prompt1.ids_1.txt new_prompt1.ids_1.txt | uniq | wc -l`, qui doit afficher 1 ;
- les tokens/s de la ligne `decode` ;
- les `tokens per round` de la ligne `speculation`, qui doivent aussi être identiques.

Pour voir le kernel seul :
```sh
nsys profile --cuda-graph-trace=node -o o3 build/strata $COMMON --tokens-file prompt1.ids
```
Puis comparer `gu_grouped_t_kernel`/`down_grouped_t_kernel` contre `gu_grouped_kernel`/`down_grouped_kernel` (avec
`STRATA_OLD_GROUPED=1`), et `gu_pair_kernel`/`down_pair_kernel` contre `gu_kernel`/`down_kernel`.

## Ce qui reste

- **Mesures GPU.** Aucune n'a été faite ; les gains ci-dessus sont des estimations.
- **Variante à 4 lignes par warp** (chemin groupé, pour des groupes de 6 à 8 entrées). Elle diviserait encore par deux
  le trafic en mémoire partagée, au prix d'environ 50 registres de plus. Non faite : il faut d'abord mesurer si le
  chemin actuel reste limité par la mémoire partagée pour les grands groupes.
- **Packs natifs (IQ).** `native_expert_grouped` redécode les grilles pour chaque entrée (audit § O3, « Packs natifs »).
  Ce n'est pas traité ici.
