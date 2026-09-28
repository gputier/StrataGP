# Paquet `iq-kernels` : décodage IQ une seule fois (#18) et division des quantificateurs Q8_1 (#4)

Branche `perf/iq-kernels`, partie de `perf/base` (b858f9b, moteur 0.1.20). Fichiers touchés :
`src/kernels/cuda/iq_kernels.cu`, `include/strata/kernels/iq_kernels.hpp`, deux lignes de
`src/kernels/cuda/native_mmvq.cu`, les tests `src/kernels/iq_multi_parity.cpp` (nouveau), `src/kernels/iq_parity.cpp`,
`src/kernels/native_expert_parity.cpp`, et l'enregistrement du nouveau test dans `CMakeLists.txt`.

**Rien n'a été mesuré sur GPU** : la machine de développement n'en a pas. Tout ce qui suit sur la vitesse est une
estimation ; ce qui est dit de l'exactitude repose sur la construction du code, le PTX et une émulation hôte décrite
plus bas. Les tests GPU compilent mais n'ont pas été exécutés.

| Changement | Par défaut | Interrupteur | Résultat |
| --- | --- | --- | --- |
| #18, O3b : décodage une fois par partie de poids | **activé** | `STRATA_OLD_IQ_MMVQ=1` rend les anciens kernels | identique au bit près |
| #4, B5 : division `__fdividef` du quantificateur Q8_1 et du SwiGLU groupé | **désactivé** | `STRATA_IQ_FASTDIV=1` l'active | derniers bits modifiés |
| #4, B5 : `native_mmvq.cu` écrit `__fdividef` explicitement | activé | aucun | PTX identique octet pour octet |

**#4 n'est que partiellement traité** : l'interrupteur existe, mais il est désactivé par défaut. Dans la
configuration par défaut, `quantize_q8_1_rows` et le SwiGLU groupé divisent toujours en IEEE (`div.rn.f32`) et
`native_quantize_q8_1` en `div.approx.ftz.f32` : l'écart décrit par l'issue est inchangé tant que
`STRATA_IQ_FASTDIV=1` n'est pas passé par défaut (après `iq_multi_parity` et l'A/B de bout en bout sur GPU). L'issue
reste ouverte. La réécriture de `native_mmvq.cu` en `__fdividef` ne change rien au binaire (PTX identique) ; elle ne
fait que rendre le contrat explicite.

Les deux variables d'environnement sont lues une fois au démarrage (valeur non vide et différente de `0`). Les tests
les basculent en cours d'exécution avec `iq_set_old_kernels(bool)` et `iq_set_fast_div(bool)` (déclarés dans
`iq_kernels.hpp`) ; comme pour `native_mmvq_set_multi_exact`, un graphe CUDA déjà capturé garde les kernels qu'il a
capturés.

## #18 (O3b) : décoder chaque partie de poids une seule fois

### Le problème

`mmvq_kernel` (utilisé par `iq_mmvq`, donc par `native_mmvq` pour IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S et IQ1_M),
`native_gu_kernel` et `native_down_kernel` (les experts natifs en VRAM de `native_expert_grouped`, appelé par la
fenêtre de vérification, `verify.cpp:571`) appelaient `Fmt<T>::dot` une fois par colonne ou par entrée du groupe.
Chaque appel relit les octets du poids, refait les recherches dans les grilles (`iq2xs_grid`, `iq3s_grid`, …), le
dépaquetage des signes et des échelles, puis fait le produit avec l'activation. Pour m entrées, le décodage est fait
m fois.

### Ce qui change

Chaque produit scalaire est coupé en deux, comme les traits multi-colonnes de `native_mmvq.cu` :

- `Split<T>::load(poids, kbx, iqs)` : tout ce qui ne dépend que du poids — les 8 mots de grille signés (ou les
  paires `int2` de la table IQ4, les codes Q2_0), les échelles entières, l'échelle fp16 convertie en float, et pour
  IQ1_M les quatre `delta` ;
- `Split<T>::apply(w, activation, iqs)` : les lectures d'activation, la même chaîne de `dp4a` dans le même ordre, la
  même étape entière sur les échelles (`(sumi0*ls0 + sumi1*ls1 + (sumi0+sumi1)/2)/4`, `(ls*sumi + sumi/2)/2`, …) et
  la même expression flottante finale (`d * sumi` avec `d = dw * d8`).

`row_dot_multi<T, NC>` garde exactement la structure de `row_dot` : mêmes appels k, même pas entre voies
(`k = lane; k += 32`), même ordre d'accumulation par colonne, même `warp_sum`. Seul le décodage sort de la boucle
par colonne. Nouveaux kernels :

- `mmvq_multi_kernel<T, NC>` : NC = 1, 2, 4 ou 8 colonnes par passe selon `ncols` (1 → 1, 2 → 2, 3-4 → 4, 5-8 → 8 ;
  au-delà de 8, par tranches de 8). Mêmes grille et blocs que `mmvq_kernel` (une chaîne par ligne, 4 lignes par
  bloc).
- `native_gu_multi_kernel<T>` et `native_down_multi_kernel<T>` : les entrées d'un groupe par paquets de
  `GRP_NC = 4`. Un groupe a au plus une entrée par token de la fenêtre ; setup écrit `--spec 4`, donc une seule
  passe dans la configuration par défaut (deux passes pour les groupes de 5 à 8 entrées avec `--spec` 5 à 8).
  Avec 8, les kernels montaient à 64-80 registres ; avec 4, ~48.

Après `warp_sum`, toutes les voies ont la même somme (l'addition est commutative), donc la voie c écrit la colonne c
au lieu de la voie 0 : même valeur, écritures réparties.

### Pourquoi c'est identique au bit près

- Les opérations entières (`dp4a`, décalages, divisions entières) sont les mêmes sur les mêmes valeurs ; l'ordre
  des additions entières ne change rien de toute façon.
- Les opérations flottantes sont les mêmes, dans le même ordre, avec les mêmes contractions : le PTX de
  `mmvq_kernel<T>` et de `mmvq_multi_kernel<T, 1>` a le même inventaire (`mul.f32` pour `d`, `cvt.rn.f32.s32`,
  `fma.rn.f32` pour l'accumulation, 5 `add.f32` pour `warp_sum`), et NC colonnes donnent NC fois la partie
  « apply ». L'accumulation est un `fma.rn.f32` explicite, que ptxas ne peut pas re-fusionner. Pour IQ1_M, les
  valeurs internes (`delta` ∈ {-0,875 ; -1,125}, `sumf`, `(sumi + sumf) * sc`) sont exactes en float, donc la
  contraction éventuelle n'y change rien.
- **Émulation hôte (faite ici)** : le code device de `iq_kernels.cu` a été extrait tel quel et exécuté sur CPU, les
  32 voies de chaque warp, `warp_sum` rejoué dans l'ordre exact du papillon `__shfl_xor_sync`, les intrinsèques
  (`__vcmpne4`, `__vsub4`, `__byte_perm`, `__dp4a`) réimplémentées. Résultats : `Split::apply(load)` égal bit à bit à
  `Fmt::dot` sur chaque appel des 9 formats ; `mmvq_multi_kernel` égal bit à bit à `mmvq_kernel` pour les 9 formats,
  ncols 1 à 8 et 11, 67 × 2560 et 5 × 1024 ; les kernels groupés égaux bit à bit aux anciens pour les 8 formats
  gate/up × 3 formats down (512 × 256), et pour les 8 formats gate/up à la forme du modèle (2560 × 640), groupes de
  0 à 11 entrées. Erreur relative contre une référence double (poids déquantifiés, activations q8_1 déquantifiées) :
  ≤ 4,6e-5. Cette émulation ne teste pas les choix de contraction de nvcc (couverts par le PTX ci-dessus) : c'est
  `iq_multi_parity` sur la 5090 qui fait foi.

### Registres (ptxas -v, sm_120, aucun débordement sauf mention)

| Kernel | Ancien | Nouveau |
| --- | --- | --- |
| `mmvq`, ncols = 1 | 38-40 | 38-40 (56 pour IQ4_NL, que `native_mmvq` n'envoie pas ici) |
| `mmvq`, NC = 2 / 4 / 8 | 38-40 | 37-40 / 40-48 / 60-72 |
| `native_gu` | 40 | 46-55 ; IQ2_XS et IQ2_S : 48 avec 8 octets de débordement (choix de ptxas) |
| `native_down` | 39-40 | 40-48 |

Conséquence : à 48 registres, 5 blocs de 256 fils par SM au lieu de 6 pour les kernels groupés. Avec une seule
entrée par groupe, le nouveau kernel fait le même travail que l'ancien avec un peu moins d'occupation : c'est le
cas à surveiller dans l'A/B.

### Gain attendu

- **HYPOTHÈSE**, kernel groupé : si le décodage coûte ~1,5 à 2 fois l'application, m entrées par groupe passent de
  m(D + A) à D + mA : ~1,3× pour m = 1,5, ~1,5× pour m = 2, jusqu'à ~2× pour m = 4 ; rien pour m = 1. Le gain réel
  dépend de la part servie par L1/L2 lors des relectures de l'ancien kernel.
- **HYPOTHÈSE**, tour complet : sur la 5090 (~93 % des experts en VRAM), si le kernel groupé pèse 20 à 30 % du tour,
  2 à 7 % de débit en décodage spéculatif. Sur la 5070, moins (moins d'experts en VRAM).
- `iq_mmvq` dense : gain seulement si un poids dense du pack est en IQ (les packs llama.cpp gardent souvent
  l'attention en K-quants) ; à ncols = 1 (décodage token par token), même travail qu'avant.

## #4 (B5) : les deux quantificateurs Q8_1

### Le constat

- `iq_kernels.cu` n'est pas dans la liste `--use_fast_math` de `CMakeLists.txt` : `quantize_q8_1_kernel` fait
  `amax / 127.0f` et `xi / d` en `div.rn.f32` (IEEE).
- `native_mmvq.cu` est dans la liste : le même quantificateur llama.cpp y devient `div.approx.ftz.f32`.
- `native_expert_grouped` quantifie ses activations (via `quantize_q8_1_rows`, `verify.cpp:541`) et la sortie de son
  SwiGLU avec le premier. Un int8 bascule quand `x/d` tombe à un ou deux ulps d'un .5, et `d` peut différer d'un ulp
  avant l'arrondi fp16. Le SwiGLU groupé fait aussi une division IEEE, là où `shared_expert.cu:68` (chemin natif)
  utilise `__fdividef`.

### Le choix

`__fdividef` explicite, comme `shared_expert.cu:68`, plutôt qu'ajouter `iq_kernels.cu` à la liste fast-math : le
drapeau changerait aussi les déquantificateurs (`-ftz`), la division du SwiGLU et toutes les autres opérations du
fichier, sans interrupteur. Vérifié sur le PTX :

- dans un fichier fast-math, `a / b` et `__fdividef(a, b)` donnent tous deux `div.approx.ftz.f32` ;
- dans `iq_kernels.cu`, `__fdividef` donne `div.approx.f32` : la même approximation, sans la mise à zéro des
  sous-normaux. La différence ne touche que les blocs dont `amax < 127 × 2^-126`, dont l'échelle fp16 vaut 0 dans
  les deux cas (les int8 y sont multipliés par 0).

Ce qui est livré :

- `quantize_q8_1_kernel<FAST>` et `swiglu_entries_kernel<FAST>` ; `FAST = false` (défaut) produit **le même PTX
  qu'avant** (comparé), `FAST = true` utilise `__fdividef`. Activation : `STRATA_IQ_FASTDIV=1`. Cela change les
  derniers bits des experts natifs en VRAM, d'où l'opt-in.
- `native_mmvq.cu` : `amax / 127.0f` et `xi / d` écrits `__fdividef(...)`. Le PTX du fichier entier est identique
  octet pour octet (vérifié) ; le contrat ne dépend plus de la présence du fichier dans la liste fast-math.

Gain : aucun en vitesse. C'est une question de cohérence : avec `STRATA_IQ_FASTDIV=1`, les deux quantificateurs
produisent les mêmes octets (vérifié par `iq_multi_parity` sur GPU), et le SwiGLU groupé fait la même opération
que celui de l'expert partagé natif. Faut-il l'activer par défaut ? Seulement si l'A/B ci-dessous montre des
tokens identiques ou une KL négligeable contre llama.cpp ; ce n'est pas mesuré.

## Tests

| Test | Type | Ce qu'il vérifie |
| --- | --- | --- |
| `iq_multi_parity` (nouveau, ctest) | GPU, synthétique, sans modèle | (1) `iq_mmvq`, 9 formats, ncols 1..8 et 11, 67 × 2560 et 5 × 1024 : nouveau = ancien au bit près, et référence double à 1e-2 près (garde-fou) ; (2) `native_expert_grouped`, 8 formats gate/up × IQ4_NL/Q2_0 (2560 × 640) et IQ4_XS (1024 × 512), 9 groupes de 0 à 11 entrées, `cap_groups` > groupes, destinations mélangées, lignes non écrites comprises : nouveau = ancien au bit près, en division IEEE et en division rapide ; (3) `quantize_q8_1_rows` en division rapide = `native_quantize_q8_1` octet pour octet, y compris des valeurs placées à quelques ulps d'un .5 (le nombre d'int8 que la division IEEE fait bouger est affiché, sans échec) ; `--bench` ajoute les temps ancien/nouveau |
| `iq_parity` (étendu) | GPU, fixture réelle (non reproductible ici, voir plus bas) | ncols 1..8 au lieu de 2 ; chaque colonne d'un appel multi-colonnes égale au bit près à un appel une colonne (mise en page exacte, le défaut) |
| `native_expert_parity` (étendu) | GPU, GGUF réel | le chemin GPU exécuté avec les nouveaux et les anciens kernels : mêmes bits exigés |

`iq_parity` lit des fixtures `logs/iq_fixture/<FORMAT>.bin` / `.f32` (en-tête int32 type, lignes, colonnes, puis les
octets bruts des lignes ; en regard, les valeurs déquantifiées par gguf-py). Le script qui les produisait,
`tools/iq_fixture.py`, cité dans l'en-tête du test depuis l'origine du dépôt, **n'est pas dans l'arbre**, et gguf-py
ne sait pas déquantifier Q2_0. Cette validation n'est donc pas reproductible en l'état ; les vérifications qui
couvrent #18 et qu'on peut réellement lancer sont `iq_multi_parity` (synthétique, aucune fixture) et
`native_expert_parity` (sur le GGUF du modèle). Si une ancienne copie de `logs/iq_fixture` existe encore,
`build/iq_parity logs/iq_fixture` reste utile (ncols 1..8).

Exécuté ici (sans GPU) : compilation complète (`cmake --build build-wp`, vert) ; tests CPU `platform_memory_test`,
`pool_stress`, `expert_multi_test`, `suffix_drafter_test`, `draft_policy_test`, `controller_test`,
`conv_cache_test`, `ple_reader_selftest` (avec un `TMPDIR` privé : sans lui, il se heurte aux autres copies de
travail qui écrivent le même `/tmp/ple_reader_selftest.bin`) passent ; `pool_test` échoue comme prévu (pas de pack) ;
`python3 serve/test_server.py` : 26 tests OK. `iq_multi_parity` lancé ici s'arrête sur « CUDA driver version is
insufficient » (pas de GPU).

## À exécuter sur la RTX 5090

```bash
# 1. Construire (même configuration que d'habitude)
cmake --build build -j

# 2. Exactitude
ctest --test-dir build -R iq_multi_parity --output-on-failure        # doit finir par "iq_multi_parity: 0 failures"
build/native_expert_parity <shard1.gguf> 0 1 2 3 20 47               # "gpu decode-once vs per-entry kernels: bitwise equal" à chaque couche
ctest --test-dir build --output-on-failure                            # le reste de la suite

# 3. Temps des kernels seuls, ancien / nouveau (µs par appel)
build/iq_multi_parity --bench
#    lignes "iq_mmvq 8192 x 2560" pour ncols 1..8 et "grouped, 16 groups of m entries" pour m = 1, 2, 3, 4, 8.
#    À regarder : m = 1 (le nouveau ne doit pas être plus lent) et m = 2..4 (le gain attendu).
```

A/B de bout en bout (3 prompts × 3 exécutions, glouton, 256 tokens, résidence statique) avec les arguments que setup
a écrits dans `strata-<modèle>.json` :

```bash
ARGS=$(python3 -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['args']))" strata-<modèle>.json)
# prompts/p1.txt p2.txt p3.txt : des prompts pré-tokenisés (ids séparés par des virgules ou des espaces)
for p in p1 p2 p3; do for r in 1 2 3; do
  STRATA_OLD_IQ_MMVQ=1 build/strata $ARGS --adapt-every 100000 --greedy --max-new 256 --tokens-file prompts/$p.txt > logs/old_${p}_$r.txt 2>&1
                       build/strata $ARGS --adapt-every 100000 --greedy --max-new 256 --tokens-file prompts/$p.txt > logs/new_${p}_$r.txt 2>&1
  STRATA_IQ_FASTDIV=1  build/strata $ARGS --adapt-every 100000 --greedy --max-new 256 --tokens-file prompts/$p.txt > logs/fast_${p}_$r.txt 2>&1
done; done
grep -H "^decode" logs/*_p?_?.txt                                     # tok/s
for p in p1 p2 p3; do diff <(grep "^output" logs/old_${p}_1.txt) <(grep "^output" logs/new_${p}_1.txt) && echo "$p: tokens identiques"; done
for p in p1 p2 p3; do diff <(grep "^output" logs/new_${p}_1.txt) <(grep "^output" logs/fast_${p}_1.txt) && echo "$p: fastdiv identique"; done
```

Attendus :

- `old` et `new` : **tokens identiques** (sinon c'est un bug : le signaler avec la sortie de `iq_multi_parity`), débit
  égal ou meilleur ; le gain se voit surtout avec des fenêtres de 3-4 tokens et beaucoup d'experts en VRAM.
- `fast` : tokens identiques ou divergence tardive (derniers bits) ; aucun effet sur le débit attendu. Ne le passer
  par défaut qu'après une comparaison de qualité (KL contre llama.cpp).
- Pour un profil plus fin : `nsys profile --cuda-graph-trace=node` sur 20 tours, noms de kernels
  `native_gu_multi_kernel` / `native_down_multi_kernel` contre `native_gu_kernel` / `native_down_kernel`.

## Ce qui reste

- Aucune mesure GPU : ni le gain, ni l'absence de régression à une entrée par groupe (occupation 5 blocs/SM au lieu
  de 6) ; voir le point « Occupation à m = 1 » ci-dessous.
- `GRP_NC = 4` est un choix sur ptxas -v, pas sur une mesure ; avec `--spec` 5 à 8, un groupe de 5 à 8 entrées fait
  deux passes.
- B5 reste opt-in tant qu'aucune comparaison de qualité n'a été faite. #4 n'est donc pas résolu dans la configuration par défaut.
- Occupation à m = 1 : les kernels groupés (`__launch_bounds__(256)`) prennent 46 à 55 registres, IQ2_XS / IQ2_S
  débordent de 8 octets, d'où 5 blocs/SM au lieu de 6. L'hôte ne connaît pas la taille des groupes (elle est sur le
  GPU, `cap_groups == cap_entries == n × K` dans `verify.cpp`), donc un aiguillage « une entrée → ancien kernel »
  ne peut pas se faire au lancement ; dans le kernel, il ne changerait pas l'occupation. Rien n'a été modifié à ce
  sujet sans mesure : lancer `iq_multi_parity --bench` (ligne m = 1) avant la fusion ; en cas de perte,
  `STRATA_OLD_IQ_MMVQ=1` rend les anciens kernels, et `__launch_bounds__(256, 6)` est l'essai suivant.
- `tools/iq_fixture.py` manque (voir « Tests ») : à réécrire si l'on veut rejouer `iq_parity` ; il faudrait une
  référence Q2_0 hors gguf-py.
