# Paquet `cpu` : les experts calculés sur le CPU

*29/09/2026 : ce document a été écrit sur une machine sans GPU ; la première exécution sur une RTX 5090 est dans
[PERF-CHANGES.md, section 6](../PERF-CHANGES.md#6-première-exécution-sur-gpu-rtx-5090-29092026).*

*Branche `perf/cpu`, partie de `perf/base` (`b858f9b`, moteur 0.1.20). Issues #27 (O8), #28 (O8b), #29 (O8c),
#30 (O8d) et #11 (B13). Contexte : [`docs/AUDIT-PERF.md`](../AUDIT-PERF.md), section 3 (O8) et section 5 (B11).*

Étiquettes des gains, comme dans l'audit : **MESURÉ** = chiffre obtenu ici, **ESTIMÉ** = calcul, **HYPOTHÈSE** = à
confirmer sur la machine cible. Aucun chiffre n'a été mesuré sur GPU : ce conteneur n'en a pas.

## En bref

| Issue | Changement | Par défaut | Interrupteur | Résultat |
|---|---|---|---|---|
| #27 O8 | `run_split` : 3 × threads plages égales de lignes, à cheval sur les experts | **activé** | `STRATA_OLD_RUN_SPLIT=1` rétablit l'ancien découpage | identique au bit près |
| #28 O8b | `madvise(MADV_HUGEPAGE)` sur l'arène en pages de 4 Ko (Linux), part `AnonHugePages` dans la note | **activé** | `STRATA_NO_THP=1` | identique au bit près |
| #29 O8c (a) | prélecture logicielle des flux d'échelles et de codes, 2 Ko en avance | **activé** | `STRATA_OLD_CPU_PREFETCH=1` (ou `STRATA_CPU_PREFETCH=0`, `STRATA_CPU_PREFETCH_CODES=0`) | identique au bit près |
| #29 O8c (b) | lignes down : correction Q2_0 exacte en entier (`vpdpbusd` amorcé avec `-sum`) | désactivé | `STRATA_CPU_INT_CORR=1` | **change les derniers bits** |
| #30 O8d | noyaux i-quant AVX-512 : grilles par `gather`, signes par `vpermi2b`, activations Q8_K recopiées alignées sur 64 o et hors aliasing 4K | **activé** | `STRATA_OLD_IQ512=1` (tout l'ancien), `STRATA_IQ512_NOPACK=1` (gather sans la copie) | identique au bit près |
| #30 O8d | experts IQ à un seul token sur le noyau AVX-512 au lieu du `vec_dot` de ggml-cpu | désactivé | `STRATA_IQ512_ONE=1` | **change les derniers bits** |
| #11 B13 | `static_assert(MAXT × 10 ≤ 128)`, refus à l'exécution d'un `k` trop grand, `static_assert(GMAX ≥ kVerifyMaxT)` | — | — | aucun changement de calcul |

Tous les changements activés par défaut sont identiques au bit près par construction (mêmes octets, mêmes opérations
dans le même ordre ; seuls le découpage en tâches, la mémoire et les instructions de chargement changent). Les tests
ci-dessous le vérifient sur données synthétiques. Les deux options qui changent l'arrondi restent désactivées.

## #27 (O8) : `run_split` en plages de lignes plates

**Avant.** `run_split` découpait chaque expert en `ceil(3 × threads / n)` parts. Avec 10 experts sur 6 threads
(5 workers + l'hôte), cela donne 20 tâches : la dernière vague n'occupe que 2 threads sur 6.

**Après.** Même découpage que `run_split_multi` : chaque phase (gate/up, puis down) est coupée en exactement
3 × threads plages égales de lignes, prises sur l'ensemble de ses experts ; une plage peut couvrir la fin d'un expert
et le début du suivant (`pool.cpp`, modes 1 et 2). Chaque ligne passe par le même `row_dot`, donc le résultat est
identique au bit près. `STRATA_OLD_RUN_SPLIT=1` rétablit l'ancien découpage (et `ExpertPool::set_flat_split`).

**Portée.** `run_split` sert au décodage **sans `--spec`**, avec un pack Strata (Q2_0). Avec `--spec` (le réglage
écrit par `setup`), la fenêtre de vérification passe par `run_split_multi`, déjà plat : ce chemin ne change pas.

**Gain (ESTIMÉ).** Temps d'une phase, 6 threads, tâches égales réclamées dynamiquement :

| experts CPU par couche | 1-3, 6, 8, 9, 12 | 4, 5, 10 | 7 | 11 | 13 |
|---|---|---|---|---|---|
| ancien découpage / idéal | 100 % | 83 % | 88 % | 92 % | 87 % |

Soit jusqu'à −17 % du temps de phase quand le CPU calcule 4, 5 ou 10 experts par couche. Comme le noyau est limité par
la RAM, les 2 threads de la dernière vague vont plus vite que la moyenne : l'audit estime 5 à 8 % du temps CPU des
experts. Sur la 5090, avec la plupart des experts en VRAM, le CPU en calcule peu par couche : gain proche de zéro.

**Test.** `pool_test --synthetic` (nouveau ctest `pool_test_synthetic`) : `run_split` dans les deux découpages, pour
n = 1 à 16, 3 fois, comparé au bit près au calcul série ; `run_split_multi` aussi. Un test de mutation (une ligne
sautée par plage) le fait bien échouer.

## #28 (O8b) : pages larges transparentes pour l'arène d'experts

**Avant.** Sans pool `hugetlb`, `mmap` anonyme en pages de 4 Ko : ~338 pages par expert de 1,38 Mo, donc un défaut de
TLB presque à chaque page, les experts étant tirés au hasard dans 34 à 50 Go.

**Après.** Dans `reserve()` (`pinned.cu`), juste après le `mmap` de repli et avant `cudaHostRegister` (qui fait fauter
toute l'arène) : `madvise(p, bytes, MADV_HUGEPAGE)`. Le noyau accorde des pages de 2 Mo si
`/sys/kernel/mm/transparent_hugepage/enabled` vaut `madvise` ou `always`. Une fois l'arène enregistrée (ou verrouillée),
la note de démarrage indique la part obtenue, lue dans `/proc/self/smaps` (`AnonHugePages` contre `Rss` sur toutes
les zones de l'arène) :

```
strata generate: expert arena: cudaHostRegister PORTABLE ok; MAP_HUGETLB unavailable (...); using 4 KB pages
+ MADV_HUGEPAGE; pinned/faulted in <t> s; AnonHugePages 34816 of 34820 MiB resident (100%) (smaps read in <n> ms)
```

`pinned/faulted in` est le temps de l'enregistrement (ou du verrouillage), c'est-à-dire
de la mise en mémoire de toute l'arène : c'est la mesure à comparer avec `STRATA_NO_THP=1`. La lecture de `smaps`
parcourt les tables de pages de l'arène (~10 M entrées à 4 Ko pour 40 Go) ; elle est chronométrée, et
`STRATA_THP_REPORT=0` la saute. `/proc/self/smaps_rollup` n'aurait rien fait gagner : il parcourt les mêmes tables
(toutes les zones du processus) et ne donne qu'un total, pas la part de l'arène.

`STRATA_NO_THP=1` saute le `madvise` (la note le dit). Windows et le chemin `hugetlb` ne changent pas.

**Vérifié ici** (sans GPU : `cudaHostRegister` échoue, l'arène est verrouillée par `mlock`) : 512 Mio, 100 %
en pages de 2 Mo avec le `madvise`, 0 % avec `STRATA_NO_THP=1`.
Temps (mêmes 512 Mio, 3 passes par bras, VM bruitée) : fault-in 0,11 à 2,75 s avec le `madvise` (les passes à
2,6-2,75 s incluent vraisemblablement l'initialisation du runtime CUDA par le premier `cudaHostRegister`, ou une
compaction ; non départagé ici), 0,17 à 0,23 s avec `STRATA_NO_THP=1` ; lecture de `smaps` 0 à 1 ms en pages de 2 Mo,
7 à 13 ms en pages de 4 Ko, soit par extrapolation linéaire ~0,5 à 1 s pour 40 Go en 4 Ko (HYPOTHÈSE), ~0 avec THP.

**Risque.** Avec `defrag` à `madvise` (valeur courante), un fault dans une zone `MADV_HUGEPAGE` peut compacter la
mémoire : si la RAM est fragmentée, l'enregistrement de 34 à 50 Go au démarrage peut prendre plus de temps. À surveiller
sur la ligne `loaded ... GiB at ... GiB/s` et le temps de démarrage. L'audit rappelle que `hugetlb` a été mesuré plus
lent sur V100 (PR #3) : c'est un cas différent, mais c'est pour cela que l'interrupteur existe.

**Gain (ESTIMÉ par l'audit).** 2 à 8 % du temps CPU des experts sous Linux.

## #29 (O8c) : prélecture et correction entière

### (a) Prélecture logicielle (activée par défaut)

Dans toutes les boucles de lignes Q2_0 (`s2_expert_vnni_q`, `s2_expert_gu_rows`/`down_rows`, les versions
multi-tokens et les lignes Q2_0 au format GGUF des packs natifs), `_mm_prefetch` des lignes situées 2 Ko plus loin,
dans le flux des échelles **et** dans celui des codes. Une prélecture par ligne de cache : d'une ligne de matrice à la
suivante, les adresses avancent de moins de 64 o, donc aucune ligne du flux n'est oubliée. Rien ne change dans le
calcul.

Réglages : `STRATA_CPU_PREFETCH=<octets>` (échelles), `STRATA_CPU_PREFETCH_CODES=<octets>` (codes), 0 = coupé ;
`STRATA_OLD_CPU_PREFETCH=1` coupe les deux (bras A/B).

**Mesuré ici** (voir « Mesures ») : c'est la prélecture des **codes** qui compte (+50 à +90 % sur un thread) ; celle
des échelles seules, proposée par l'audit, ne gagne presque rien. Ce conteneur est une VM : ses préchargeurs matériels
peuvent être moins efficaces que ceux d'un Ryzen nu. Le gain sur la machine cible est une **HYPOTHÈSE**, d'où le bras
A/B.

### (b) Correction entière des lignes down (opt-in, `STRATA_CPU_INT_CORR=1`)

La formule Q2_0 est `d_w × (d_x × Σ c·x̂ − d_x × Σ x̂)`. Le noyau 512 bits accumulait la seconde somme en flottant
(un accumulateur `corr`, une FMA par 8 blocs, une seconde réduction zmm complète). Avec l'option, `vpdpbusd` part d'une
graine `−Σ x̂` du morceau de 32 (voies 0 et 8 de chaque bloc), écrite par `act_quant_q8_1` dans `ActQ::seed` : chaque
morceau donne `Σ (c − 1)·x̂` exactement, et l'accumulateur de correction disparaît. Appliqué aux lignes down des packs
Strata et aux lignes down Q2_0 au format GGUF des packs natifs.

- L'arrondi change (écart relatif mesuré 1,1e-7 contre le chemin par défaut, et même écart à l'oracle scalaire :
  9,94e-7 contre 9,95e-7). D'où l'opt-in.
- Les noyaux mono-token et multi-tokens restent identiques entre eux au bit près dans les deux modes (testé) :
  le décodage spéculatif reproduit toujours le décodage simple.
- Le kernel GPU `--expert-cache-cpu-order` suit le noyau 256 bits (`STRATA_CPU_YMM=1`) et non le noyau 512 bits : il
  ne reproduisait déjà pas le chemin 512 bits par défaut, et ne reproduit pas non plus ce mode.
- **Mesuré ici :** entre −16 % et +7 % selon les passes, soit dans le bruit de cette machine.

**Statut de #29 : partiel.** (a) est fait et activé ; (b) n'est fait qu'à moitié : l'amorçage de `vpdpbusd` existe
mais reste opt-in, donc par défaut le travail en plus par ligne down mesuré par l'issue est inchangé. Restent : le
noyau down à 4 lignes par itération, et le passage de la correction entière par défaut après revalidation du chemin
GPU `--expert-cache-cpu-order` (à suivre dans une issue dédiée).

**Non fait :** « traiter 4 lignes par itération » (proposition de l'audit). Réduire 4 accumulateurs ensemble change
l'ordre de la réduction finale ; pour garder mono-token et multi-tokens identiques, il faudrait un arbre de réduction
explicite commun aux deux, et le gain de la correction seule n'est pas mesurable ici. Laissé de côté.

## #30 (O8d) : noyaux i-quant AVX-512

Constat (**MESURÉ** dans le dépôt) : ~5 Go/s par cœur, limité par le décodage des poids, pas par la RAM.

**Après** (`iq_avx512.cpp`, `decode_g` par format) :

- **Grilles.** Les 8 (IQ2) ou 16 (IQ3) lectures scalaires insérées voie par voie deviennent un seul
  `_mm512_i32gather_epi64` / `_mm512_i32gather_epi32`. Les index sont construits en SIMD (`pshufb`/`vpmovzx`,
  `srlv` pour les bits hauts d'IQ2_S, un `OR` masqué pour le 9e bit d'IQ3_S).
- **Signes.** Les 8 lectures de `ksigns_iq2xs` (128 octets) deviennent un seul `vpermi2b` sur la table tenue dans deux
  registres (IQ2_XXS, IQ2_XS, IQ3_XXS ; IQ2_S et IQ3_S lisent déjà un masque de 64 bits).
- **Activations.** Les `qs` d'un `block_q8_K` sont à l'offset 4 d'un bloc de 292 o : chaque chargement de 64 valeurs
  chevauchait deux lignes de cache. Et les activations du pool sont espacées d'exactement 4 096 o
  (`kNativeActBytes`) : le même chargement de chaque token tombait dans le même ensemble du L1. Chaque appel recopie
  maintenant les valeurs des tokens (10 × 256 o chacun) dans un tampon aligné sur 64 o, au pas de 3 648 o (57 lignes).
  Coût : 40 lectures/écritures de 64 o par token et par appel, contre ~80 lectures par ligne de poids.
- **Pas des activations down natives** (`pool.hpp`, `hq`) : 1 024 + 64 o au lieu de 1 024 (t et t+4 étaient à 4 Ko).
- **Pas des activations gate/up natives** (`expert_source.cpp`, `nact_multi`) : `kNativeActBytes` + 64 = 4 160 o au
  lieu de 4 096, pour les chemins qui les lisent sur place (`STRATA_IQ512_NOPACK=1`, lignes plus larges que la copie
  alignée, `vec_dot` de ggml-cpu). Seules les adresses changent : identique au bit près, sans interrupteur.
- Mêmes octets, mêmes opérations après le décodage : identique au bit près à l'ancien noyau, que
  `STRATA_OLD_IQ512=1` conserve (`STRATA_IQ512_NOPACK=1` : les `gather` sans la copie alignée).

**Opt-in `STRATA_IQ512_ONE=1`.** `native_gu_rows` envoyait les experts à un seul token vers le `vec_dot` de ggml-cpu
(« aussi rapide ou plus »). Avec le nouveau décodage, le noyau AVX-512 le dépasse aussi à un token (tableau ci-dessous),
et la plupart des experts d'une fenêtre n'ont qu'un token (~1,5 en moyenne). Mais l'ordre des additions flottantes
diffère de celui de ggml : c'est donc une option. Avec elle, chaque token passe par le même noyau quel que soit le
nombre de tokens routés vers son expert.

**Non fait :** les tables de correspondance « à la T-MAC » (piste de l'article, §7) : c'est un autre noyau, avec un
autre ordre de calcul ; hors de portée d'un changement identique au bit près.

**Test.** `iq512_test` (nouveau ctest) : pour les 5 formats, 1 à 8 tokens, lignes gate/up et lignes simples (départ
impair), les trois versions comparées au bit près ; l'ancienne comparée au `vec_dot` de ggml (rel ≤ 1e-5). Un test de
mutation (décalages d'IQ2_S faux) le fait bien échouer.

## #11 (B13) : gardes

- `expert_source.cpp` : `kMaxWindowEntries = 128`, `static_assert(MAXT * 10 <= kMaxWindowEntries)`, et
  `expert_pool_dispatch_multi` refuse (échec propre, message) `k < 1` ou `n_tok × k > 128` avant d'écrire dans
  `kind`/`distinct`/`first_of`. `dma_src[64]` était déjà gardé par `fetches < 64`.
- `s2_expert_grouped.cu` : `static_assert(GMAX >= kVerifyMaxT)`. Seule modification GPU du paquet : compile-time,
  aucun kernel ne change (pas de test de parité à ajouter).

## Mesures (ce conteneur)

Machine : VM KVM, Xeon Emerald Rapids (famille 6, modèle 207) à 2,1 GHz, 4 vCPU (pool : 3 workers + l'hôte), L3 de
260 Mo, AVX-512 complet, pas de GPU. **Très bruitée** : d'autres paquets compilaient et testaient en même temps (charge
moyenne entre 6 et 140). Tous les bras sont entrelacés et on garde le meilleur de N ; les chiffres absolus ne valent
rien, seules les tendances qui se répètent d'une passe à l'autre comptent. Ce n'est **pas** un Ryzen : les `gather`
et les préchargeurs matériels n'ont pas les mêmes coûts sur Zen 4 et Zen 5.

**O8d, `iq512_test --bench 16`** (gate/up de 16 experts synthétiques, 1 thread, poids en L2/L3, meilleur de 7 ; Go/s
de poids ; deux passes, la seconde entre parenthèses) :

| format | nt | ancien | gather + copie | gain | `vec_dot` ggml à 1 token |
|---|---|---|---|---|---|
| IQ2_XXS | 1 | 2,58 (1,84) | 2,93 (2,64) | +14 % (+44 %) | 3,15 (2,85) |
| IQ2_XXS | 3 | 1,46 (1,83) | 2,38 (2,21) | +63 % (+21 %) | |
| IQ2_XS | 1 | 2,91 (2,27) | 5,14 (4,00) | +76 % (+76 %) | 3,16 (3,13) |
| IQ2_XS | 3 | 2,56 (2,22) | 2,08 (3,29) | −19 % (+48 %) | |
| IQ3_XXS | 1 | 2,51 (2,90) | 4,86 (4,79) | +94 % (+65 %) | 3,33 (3,11) |
| IQ3_XXS | 3 | 1,80 (2,16) | 3,49 (4,15) | +94 % (+93 %) | |
| IQ3_S | 1 | 2,58 (2,48) | 5,95 (5,87) | +131 % (+137 %) | 3,05 (2,23) |
| IQ3_S | 3 | 1,72 (1,77) | 3,97 (4,13) | +132 % (+133 %) | |
| IQ2_S | 1 | 2,82 (3,50) | 4,40 (6,10) | +56 % (+74 %) | 3,63 (3,61) |
| IQ2_S | 3 | 2,26 (2,21) | 3,06 (3,23) | +35 % (+46 %) | |

Sur 50 lignes (5 formats × 5 nombres de tokens × 2 passes), une seule régression (IQ2_XS nt=3, première passe,
non reproduite). Presque tout vient des `gather` et du `vpermi2b` : la copie alignée seule reste dans le bruit
(−10 % à +10 % par-dessus les `gather` selon le format et la passe, plutôt négative sur IQ2_XXS). Le banc place les
activations à 4 096 o d'écart comme le pool, l'aliasing 4K y est donc inclus.

**O8c, `expert_multi_test --bench 1024`** (1 024 experts synthétiques = 1,4 Go depuis la RAM, 1 thread, meilleur de 5 ;
µs par expert ; deux passes) :

| bras | n = 1 | n = 2 | n = 4 |
|---|---|---|---|
| défaut avant ce paquet (sans prélecture) | 311 / 323 | 369 / 458 | 602 / 519 |
| prélecture échelles 2 Ko seule | +5 % / +2 % | +10 % / −1 % | +1 % / +12 % |
| prélecture codes 512 o seule | +53 % / +55 % | +64 % / +75 % | +51 % / +9 % |
| échelles 2 Ko + codes 2 Ko (**nouveau défaut**) | +77 % / +68 % | +61 % / +88 % | +83 % / +25 % |
| correction entière (opt-in) | +3 % / +7 % | +2 % / −3 % | −7 % / −16 % |

**O8 + O8b + O8c ensemble, `pool_test --bench 64`** (`run_split`, 48 couches par token tirées d'une arène de
640 experts = 885 Mo, meilleur de 15 tokens ; gain contre l'ancien découpage sans prélecture ni THP ; chaque case :
passe 1, passe 2 avec 2 workers + l'hôte / passe 1, passe 2 avec 3 workers + l'hôte) :

| experts CPU par couche | plat | plat + prélecture | plat + THP | plat + THP + prélecture |
|---|---|---|---|---|
| 10 | +13 %, −8 % / −8 %, +3 % | +74 %, +46 % / +33 %, +49 % | +65 %, −49 % / −17 %, −17 % | +85 %, +98 % / +33 %, +107 % |
| 7 | +41 %, +3 % / +5 %, +4 % | +48 %, +82 % / +75 %, +71 % | +10 %, +21 % / +8 %, +9 % | +135 %, +109 % / +101 %, +88 % |
| 4 | +9 %, +9 % / +23 %, +10 % | +90 %, +78 % / +64 %, +57 % | +11 %, +46 % / +17 %, +16 % | +52 %, +186 % / +88 %, +114 % |

Lecture : la prélecture gagne à chaque passe et à chaque taille ; la combinaison des quatre changements est toujours
la meilleure. Le découpage plat et les THP seuls restent dans le bruit de cette VM (le découpage plat ne peut gagner
que 0 à 17 % d'une phase, voir le tableau d'O8 ; à 4 threads il n'y a d'ailleurs rien à gagner pour 10 et 4 experts).

## Validation sur la RTX 5090

### 1. Construire

```bash
git fetch origin && git checkout perf/cpu
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON \
      -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j
```

(Sous Windows, la même chose depuis l'invite « x64 Native Tools », ou la construction habituelle de `SETUP.bat`.)

### 2. Tests

```bash
# CPU, données synthétiques : doivent tous passer
ctest --test-dir build -R "pool_test_synthetic|pool_stress|expert_multi_test|iq512_test" --output-on-failure
# avec le pack Strata dans pack/full (lancés depuis la racine du dépôt)
ctest --test-dir build -R "^pool_test$|^expert_parity$" --output-on-failure
# modèle IQ : noyaux AVX-512 contre ggml sur les vraies lignes (rel <= 1e-5 attendu)
build/native_expert_parity <shard 1 du GGUF IQ2_XS ou IQ3_XXS>
```

### 3. Microbenchmarks CPU (sans GPU, 2 minutes)

```bash
build/pool_test --bench              # découpage, prélecture et (Linux) THP, sur tous les cœurs sauf un
build/expert_multi_test --bench 1024 # bras O8c, un thread
build/iq512_test --bench             # ancien / gather / gather + copie, et vec_dot de ggml à un token
```

Ce sont eux qui disent si la prélecture et les `gather` gagnent aussi sur Zen 5 comme sur ce Xeon.

### 4. A/B de bout en bout

Même protocole que l'audit : 3 prompts × 3 exécutions par bras, `--adapt-every 100000`, glouton, 256 tokens, bras
entrelacés. `ARGS` = les arguments moteur écrits par `setup` dans `strata-<modèle>.json` (clé `"args"`), lancés depuis
la racine du dépôt.

```bash
mapfile -t ARGS < <(python3 -c 'import json; print("\n".join(json.load(open("strata-<modèle>.json"))["args"]))')
ENGINE=build/strata                  # Windows : build\strata.exe
OLD="STRATA_OLD_RUN_SPLIT=1 STRATA_NO_THP=1 STRATA_OLD_CPU_PREFETCH=1 STRATA_OLD_IQ512=1"
for p in p1.ids p2.ids p3.ids; do    # par exemple trois prompts de bench/prompts
  for run in 1 2 3; do
    for arm in old new int_corr iq_one; do
      case $arm in
        old)      E="$OLD" ;;
        new)      E="" ;;
        int_corr) E="STRATA_CPU_INT_CORR=1" ;;
        iq_one)   E="STRATA_IQ512_ONE=1" ;;   # n'a d'effet qu'avec un modèle IQ (pack natif)
      esac
      env $E $ENGINE "${ARGS[@]}" --adapt-every 100000 --greedy --max-new 256 --tokens-file $p \
        > ab_${arm}_$(basename $p .ids)_$run.txt 2>&1
    done
  done
done
grep -H "^decode" ab_*.txt                          # tok/s de décodage
grep -H "^pool multi\|pool phases" ab_*.txt         # temps CPU des experts (avec / sans --spec)
grep -H "expert arena" ab_new_*.txt | head -1       # part AnonHugePages obtenue (Linux)
# identique au bit près attendu : mêmes tokens entre old et new
for f in ab_new_*.txt; do cmp <(grep "^output" $f) <(grep "^output" ${f/new/old}) || echo "DIFF $f"; done
```

Sous PowerShell, un bras se lance avec `$env:STRATA_OLD_RUN_SPLIT=1; ...` puis `Remove-Item Env:STRATA_OLD_RUN_SPLIT`
(`STRATA_NO_THP` n'y a pas d'effet : THP est propre à Linux).

À vérifier :

- **Tokens identiques** (lignes `output`) entre `old` et `new` : tout ce qui est activé par défaut est identique au bit
  près. `int_corr` et `iq_one` peuvent diverger après quelques dizaines de tokens (arrondi différent) : comparer
  alors les tok/s, pas les tokens.
- **`run_split` (#27)** ne sert que sans `--spec` : pour le mesurer, refaire `old` / `new` en retirant `--spec` et
  `--mtp` des `ARGS`, et lire `pool phases ... drain`.
- **Démarrage (#28)** : comparer la ligne `loaded ... GiB at ... GiB/s` et le temps jusqu'au premier token entre
  `new` et `STRATA_NO_THP=1`.

**Gain attendu de bout en bout (ESTIMÉ / HYPOTHÈSE).** Sur la 5090, avec ~70 % des experts en VRAM, le CPU pèse peu
dans un tour : < 1 % pour O8 et O8b (audit). La prélecture (#29) et les noyaux IQ (#30) sont ceux qui peuvent se voir :
si les microbenchmarks de l'étape 3 donnent sur la machine des gains du même ordre qu'ici, le temps CPU des experts
baisse fortement, et le gain de bout en bout dépend de la part de ce temps dans le tour (`pool multi` contre
`wait for rings` dans la ligne `verify window`) : quelques % sur la 5090, davantage sur la 5070 et avec les modèles IQ.

## Fichiers touchés

`src/kernels/cpu/pool.cpp`, `include/strata/kernels/cpu/pool.hpp`, `src/kernels/cpu/expert.cpp`,
`include/strata/kernels/cpu/expert.hpp` (`ActQ::seed`, 640 o de plus par activation), `src/kernels/cpu/iq_avx512.cpp`,
`include/strata/kernels/cpu/iq_avx512.hpp`, `src/kernels/cpu/native_expert.cpp`, `src/core/pinned.cu`,
`src/core/expert_source.cpp` (gardes), `src/kernels/cuda/s2_expert_grouped.cu` (`static_assert`), tests
`src/kernels/cpu/pool_test.cpp`, `expert_multi_test.cpp`, `iq512_test.cpp` (nouveau), `CMakeLists.txt` (ctests
`pool_test_synthetic` et `iq512_test`).
