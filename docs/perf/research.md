# research : localité du routage (#51) et état GDN en BF16 (#53)

Branche `perf/research`, partie de `perf/base` (b858f9b, moteur 0.1.20). Aucune mesure GPU n'a été faite : la
machine de développement n'a pas de GPU. Tout a été compilé (CUDA 13.0, sm_120, aucun spill) ; les tests CPU ont été
lancés, le nouveau test GPU `gdn_state_bf16_parity` compile et reste à lancer sur la RTX 5090. Étiquettes comme dans
l'audit : **MESURÉ** (chiffre obtenu ici, sur CPU), **ESTIMÉ** (calcul), **HYPOTHÈSE** (à confirmer au profileur).

Ces deux issues sont des pistes « à mesurer avant d'investir » (audit, section 7). Ce paquet livre donc surtout de
quoi mesurer : un état BF16 **optionnel** et ses tests de dérive (#53), et un outil qui chiffre la prévisibilité du
routage (#51). S6 (prefill par couche entière) n'est pas traité ici.

## Ce qui change

| # | Issue | Quoi | Par défaut | Commande / option |
|---|---|---|---|---|
| 1 | #53 | État récurrent GDN stocké en BF16, arithmétique FP32 | **désactivé** (change les valeurs) | `--gdn-state-bf16` |
| 2 | #53 | Checkpoints de conversation compacts quand l'état est en BF16 (61 Mo au lieu de 118 par emplacement) | seulement avec l'option 1 | — |
| 3 | #53 | Test `gdn_state_bf16_parity` : exactitude par construction + dérive sur 4 096 tokens | ctest | `--selftest`, `--cpu-drift`, `--tokens N` |
| 4 | #53 | `tools/logits_kl.py` : KL et accord top-1, position par position, entre deux `--dump-logits` | outil | — |
| 5 | #51 | `tools/routing_locality.py` : réutilisation entre tokens et prédiction de la couche suivante | outil | — |
| 6 | #51 | `--dump-routing` écrit aussi les tokens validés des fenêtres de vérification (avant : chemin « token » seulement) | actif seulement avec `--dump-routing` | — |

Sans `--gdn-state-bf16`, le moteur calcule exactement comme avant : les deux kernels touchés sont maintenant des
templates sur le type de stockage de l'état, et leur instanciation FP32 produit **le même PTX** que les kernels
d'origine (comparé ligne à ligne, noms normalisés, pour `gdn_step_norm_kernel` et `gdn_step_norm_multi_kernel` ; les
autres kernels des deux fichiers sont inchangés). Il n'y a donc pas d'ancien chemin séparé à garder derrière un
`STRATA_OLD_*` : c'est le même code. Le seul ajout sur le chemin par défaut est la lecture d'un booléen à l'hôte au
lancement (ou à la capture) de ces deux kernels.

### #53 : l'état GDN en BF16

**Principe.** Le pas GDN lit et écrit tout l'état (128 × 48 × 128 valeurs par couche, 3 Mio en FP32) à chaque token.
Avec l'option, seul le **stockage** passe en BF16 : le chargement élargit (exact), toute l'arithmétique est celle du
kernel FP32, et l'écriture arrondit au plus proche pair (un NaN reste un NaN, contrairement à `bf16_from_f32`, B11).

**Disposition.** La tranche d'état de chaque couche garde sa taille FP32 et sa disposition `[ligne][tête][colonne]` :
l'état BF16 en occupe la première moitié, l'historique de convolution (FP32) ne bouge pas. L'arène de session, le
budget du planificateur et la remise à zéro (`cudaMemsetAsync` à 0) sont donc inchangés ; la VRAM n'est pas
récupérée (~54 Mo en tout, sans intérêt ici).

**Où l'état est lu ou écrit, et ce que fait chaque chemin avec l'option :**

| Chemin | Fichier | Avec `--gdn-state-bf16` |
|---|---|---|
| Pas GDN fusionné (chemin « token » : premier token, sans `--spec`) | `fused_gdn.cu` | kernel BF16 ; arrondi à **chaque token** |
| Fenêtre de vérification (lecture seule) et `commit` (réécriture) | `verify_kernels.cu` | kernel BF16 ; la fenêtre part d'un seul chargement, le `commit` arrondit **une fois par fenêtre** (pour ses n_keep tokens) |
| Prefill (`gdn_recurrence`) | `prefill.cpp` | élargi dans la tranche FP32 inutilisée de la session (`SessionState::gdn_wide`, laissée libre par `gdn_point_at` : aucune VRAM en plus), récurrence FP32 inchangée, réarrondi **une fois par morceau** |
| Pas GDN non fusionnés (`native_gdn_step`, `gdn_step`) | `layer.cpp` | refusés (message d'erreur) : ils liraient les bits BF16 comme du FP32 |
| Checkpoints de conversation (`--serve`) | `generate.cpp` | seules les parties vivantes sont copiées (`cudaMemcpy2D`) : 61,0 Mo au lieu de 117,7 Mo par emplacement (ESTIMÉ, calcul exact des tailles) |
| Hash d'état (`STRATA_STATE_HASH`, `STRATA_STATE_HASH_GDN`) | — | inchangé (octets de la tranche) |

`strata` refuse l'option sans le pas GDN fusionné natif (`--native` ou `--native-gdn`, sans `--no-fused-gdn`).
L'interrupteur est global au processus et posé au démarrage, avant toute capture de graphe (comme
`native_gdn_set_enabled`) : `fused_gdn_step_norm` et `gdn_step_norm_multi` lisent alors leur `float* state` comme la
tranche BF16, si bien qu'aucun site d'appel de `verify.cpp` ou `layer.cpp` n'a changé.

**Registres** (`ptxas -v`, sm_120, aucun spill) : pas fusionné 128 (FP32) / 128 (BF16) ; fenêtre 126 (FP32, comme
avant) / 128 (BF16) ; élargissement 30, arrondi 26.

**Dérive, MESURÉ sur émulation CPU** (`gdn_state_bf16_parity --cpu-drift`, données synthétiques, 8 têtes dont les
décroissances vont de 0,37 à 0,9999 ; erreur L2 relative par rapport au FP32 ; « sortie » = pire token jusque-là) :

| Tokens | sortie, arrondi à chaque token | état | sortie, arrondi à chaque `commit` (fenêtres de 4, acceptation aléatoire) | état |
|---:|---:|---:|---:|---:|
| 16 | 0,68 % | 0,47 % | 0,39 % | 0,29 % |
| 256 | 1,28 % | 1,54 % | 0,85 % | 0,92 % |
| 1 024 | 1,65 % | 1,92 % | 1,18 % | 1,22 % |
| 4 096 | 1,74 % | 1,93 % | 1,18 % | 1,24 % |
| 16 384 | 1,83 % | 1,93 % | 1,18 % | 1,26 % |

L'erreur **se stabilise** vers 1 000 tokens (la règle delta et la décroissance oublient les anciens arrondis) ; la
tête la plus lente porte ~2,1 %, la plus rapide ~0,2 %. Un arrondi vers zéro (troncature) donne 25 % de sortie et
35 % d'état sur la même émulation : le test borne à 5 % et attrape ce genre de faute. Ces chiffres ne disent **pas**
ce que devient la qualité du modèle : c'est le rôle de la mesure KL ci-dessous, sur le vrai modèle.

**Gain attendu.**

- Trafic de l'état (ESTIMÉ) : chemin « token » 226 → 113 Mo par token ; fenêtre de vérification (lecture) +
  `commit` (lecture + écriture) 340 → 170 Mo par tour.
- Sur la 5090 (~1,5 To/s atteignables) : ~0,11 ms de moins par tour si ces kernels sont limités par la bande
  passante (ESTIMÉ), soit **~0,5 %** d'un tour de 15 à 20 ms. HYPOTHÈSE : le pas GDN n'occupe que 48 SM sur 170 et
  chaque thread a 32 chargements indépendants en vol ; il peut être limité par la latence plus que par le débit, et le
  gain réel être plus petit. Le `commit` est mesuré à part dans la ligne `verify window … commit X ms/round`.
- Prefill : un élargissement et un réarrondi par couche et par morceau (~340 Mo par morceau de ≤ 8 192 tokens, ~0,2
  ms, négligeable).
- `--serve` : 57 Mo de RAM en moins par checkpoint de conversation (6 emplacements par défaut : ~340 Mo), et des
  copies deux fois plus courtes.

Conclusion provisoire : un gain de débit **faible** (≤ 1 %) contre un risque de qualité réel. L'option n'a d'intérêt
que si la mesure KL montre une dérive négligeable **et** que l'A/B montre un gain au-delà du bruit.

**Le test GPU `gdn_state_bf16_parity --selftest`** (compilé, pas exécuté) :

1. exact par construction, comparé bit à bit : à partir d'un état représentable en BF16, le pas BF16 donne la sortie
   FP32 et écrit l'état FP32 arrondi (8 tokens) ; la fenêtre en vérification donne les 4 sorties FP32 et n'écrit
   aucun des deux états ; le `commit` (n_keep = 0 à 4) écrit l'état FP32 arrondi ; l'interrupteur fait bien passer
   `fused_gdn_step_norm(float*)` et `gdn_step_norm_multi(float*)` par la forme BF16 (tranche de taille FP32) ; une
   fenêtre d'un token égale le pas (FP32 et BF16) ; l'élargissement couvre les 65 536 motifs BF16, l'arrondi 1 M de
   valeurs (égalités, dénormaux, infinis, NaN) ;
2. la dérive ci-dessus sur GPU, 48 têtes, 4 096 tokens (`--tokens 16384` pour plus long), bornée à 5 %.

### #51 : localité du routage

**La trace.** `--dump-routing FICHIER` écrivait un enregistrement par couche et par position du chemin « token »
seulement : une exécution avec `--spec` ne traçait que son premier token, et un pack natif (IQ), qui ne décode qu'en
fenêtres, ne traçait **rien**. Maintenant, les identifiants de chaque fenêtre sont gardés par couche à mesure que le
pool les reçoit, et une fois l'acceptation connue, **seuls les tokens validés** sont écrits, dans l'ordre des
positions : exactement les enregistrements du chemin « token », avec des poids à 0 (la fenêtre ne les publie pas au
pool ; l'outil compte alors chaque expert pareil). Le format ne change pas (`tools/make_profile.py` le lit toujours).
Le prompt n'est tracé que s'il passe par le chemin « token » (pas de `--prefill`).

**L'outil** `tools/routing_locality.py TRACE…` (numpy) mesure, sur la partie de chaque trace qui suit celle qui
apprend les tables (`--train-frac`, 0,5 par défaut) :

- **réutilisation entre tokens** : part des experts du token t+1 à la couche l déjà routés à la couche l par les W
  derniers tokens (W = 1, 2, 4, 8), pondérée ou non, et le nombre d'experts distincts de cette fenêtre (ce que coûterait
  de les garder) ;
- **prédiction de la couche suivante** : part des experts du token t+1 à la couche l+1 contenus dans un ensemble
  prédit de B experts, pour quatre prédicteurs :
  - `prev` : les experts du token t à la couche l+1 (connus un token à l'avance) ;
  - `xlayer` : P(expert à l+1 | experts du token t+1 à l), table apprise et lissée (`--smooth`), connue une couche à
    l'avance, dès que le routeur de la couche l a tourné ;
  - `both` : les deux combinés (OU bruité) ;
  - `freq` : les B plus fréquents à l+1, un profil statique (ce que fait déjà le cache VRAM) ; le hasard vaut B/512 ;
- **`--resident N`** : même mesure restreinte aux experts **hors** d'un cache statique des N paires (couche, expert)
  les plus utilisées ; B devient le nombre de copies lancées par couche et par token, et l'outil donne la part des
  manqués qu'elles couvrent et leur précision (copies utiles / copies) : c'est le chiffre qui décide de #51.

Sortie texte, `--per-layer` pour la table par couche, `--json` pour tout garder. Tests unitaires sur traces
synthétiques de localité connue (mêmes experts partout, fenêtre glissante, permutation fixe entre couches, routage
aléatoire au niveau du hasard, cache résident, poids absents, lecture d'une trace coupée), enregistrés dans ctest.

**Lire le résultat (HYPOTHÈSE, à confirmer avec les chiffres).** Une copie anticipée d'expert Q2_0 (1,38 Mo) coûte
~55 µs de PCIe 4.0 (~25 Go/s) ; 2 copies par couche et par token, c'est ~130 Mo par token. Elle ne vaut que si sa
précision est haute : sur la 5090, avec ~93 % des experts en VRAM, il reste ~0,7 manqué par couche et par token.
Un prédicteur utile doit couvrir une part notable des manqués (≥ 30 %) avec une précision ≥ 0,3 à B = 1 ou 2 ; sinon
#51 ne vaut pas son coût. Si `xlayer` ne fait pas mieux que `freq`, le routage seul ne suffit pas et il faudrait
prédire depuis l'état caché (un petit prédicteur appris), hors de ce paquet. La prélecture en L3 côté CPU est moins
chère (pas de PCIe, mais de la bande passante DDR) et tolère une précision plus faible. Limite : la mesure est par
token, alors qu'en `--spec` le travail se fait par fenêtre de 3 à 4 tokens (union de leurs experts).

## Tests lancés ici

- Construction complète `-DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON` (sm_120) : **OK**, aucun nouvel
  avertissement.
- `ple_reader_selftest`, `platform_memory_test`, `pool_stress`, `expert_multi_test`, `suffix_drafter_test`,
  `draft_policy_test`, `controller_test`, `conv_cache_test` : **OK** ; `pool_test` échoue comme attendu (il lui faut
  un pack : « cannot read expert 0 of layer 0 from pack/full/experts.bin »).
- Nouveaux, CPU : `gdn_state_bf16_drift_cpu` (ctest, `gdn_state_bf16_parity --cpu-drift`), `routing_locality_test`
  (9 tests), `logits_kl_test` (5 tests) : **OK**. `python3 serve/test_server.py` (26) et
  `python -m unittest tools.test_calibrate` (10) : **OK**.
- PTX : instanciations FP32 identiques aux kernels d'origine (voir plus haut).
- Nouveau test GPU **compilé, pas exécuté** : `gdn_state_bf16_parity` (ctest).

## À valider sur la RTX 5090

```bash
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON \
      -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j

# 1) les tests : le nouveau, puis les voisins (le chemin FP32 par défaut ne doit pas avoir bougé)
ctest --test-dir build -R "gdn_state_bf16|gdn_parity|gdn_fused_parity|routing_locality|logits_kl" --output-on-failure
./build/gdn_state_bf16_parity --selftest --tokens 16384      # la dérive, plus longue
```

**2) Dérive sur le vrai modèle, token par token, prompt imposé** (le pire cas : un arrondi à chaque token). Il faut un
pack **Q2_0** (non natif) : un pack natif ne passe pas par le chemin « token » et n'écrit pas de logits de prompt. Un
prompt long (4K à 8K tokens), sans `--spec` ni `--prefill` :

```bash
KL="--pack <PACK_Q2_0> --ple-gguf <PLE> --native <SHARD1> --max-new 1 --logits-stride 16 --adapt-every 100000 --max-context 16384"
mkdir -p ab
./build/strata $KL --tokens-file prompts/long.ids --dump-logits ab/l_fp32.bin
./build/strata $KL --tokens-file prompts/long.ids --dump-logits ab/l_fp32_bis.bin      # témoin
./build/strata $KL --gdn-state-bf16 --tokens-file prompts/long.ids --dump-logits ab/l_bf16.bin
python3 tools/logits_kl.py ab/l_fp32.bin ab/l_fp32_bis.bin    # attendu KL 0, top-1 1.0000 ; sinon, c'est le bruit de fond
python3 tools/logits_kl.py ab/l_fp32.bin ab/l_bf16.bin        # KL par quart de prompt : la dérive croît-elle ?
```

Repère pour juger (HYPOTHÈSE) : le KV Q4_0, laissé en option, coûtait +8 à 12 % de perplexité
(`bench/results/2026-09-27-kv-q4`) ; un KL moyen de l'ordre de 1e-3 nat et un accord top-1 > 99 % sur 8K seraient
acceptables, un KL qui monte d'un quart à l'autre ne l'est pas.

**3) A/B de débit** (production, `--spec`) : 3 prompts × 3 exécutions, glouton, 256 tokens, résidence statique.
Remplacer `<OPTS>` par la ligne habituelle (`--pack`, `--ple-gguf`, `--native`, `--expert-profile`, `--expert-cache`,
`--spec`, `--mtp`, …).

```bash
OPTS="<OPTS> --greedy --max-new 256 --adapt-every 100000"
mkdir -p ab
for p in p1 p2 p3; do for r in 1 2 3; do
  ./build/strata $OPTS --tokens-file prompts/$p.ids > ab/fp32_${p}_$r.txt 2>&1
  ./build/strata $OPTS --gdn-state-bf16 --tokens-file prompts/$p.ids > ab/bf16_${p}_$r.txt 2>&1
done; done
# débit, tokens par tour, et le commit (il lit et écrit l'état) :
grep -H "^decode\|^speculation\|^verify window" ab/*.txt
# les tokens ne sont PAS identiques par construction ; où la première différence arrive-t-elle ?
for p in p1 p2 p3; do
  python3 -c "import sys; a,b=[[l.split(':',1)[1].split() for l in open(f) if l.startswith('output')][0] for f in sys.argv[1:]]; n=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), min(len(a),len(b))); print(sys.argv[1], 'mêmes tokens jusqu à', n, 'sur', min(len(a),len(b)))" ab/fp32_${p}_1.txt ab/bf16_${p}_1.txt
done
```

Critères : `fp32` doit donner les mêmes tokens d'une exécution à l'autre (sinon la comparaison n'a pas de sens) ;
garder l'option seulement si `decode … tok/s` monte au-delà du bruit entre les 3 exécutions **et** si la mesure 2 est
propre. Profil du `commit` : `nsys profile --cuda-graph-trace=node -o gdn_bf16 ./build/strata $OPTS --gdn-state-bf16
--tokens-file prompts/p1.ids`, puis comparer la durée de `gdn_step_norm_multi_kernel` entre les deux bras
(`nsys stats --report cuda_gpu_kern_sum`).

**4) Localité du routage (#51)** : les mêmes 3 prompts, la configuration de production (avec `--spec`, les tokens
validés sont tracés). Plus de tokens donnent une table `xlayer` plus fiable (`--max-new 1024`).

```bash
for p in p1 p2 p3; do
  ./build/strata $OPTS --tokens-file prompts/$p.ids --dump-routing ab/route_$p.bin > ab/route_$p.txt 2>&1
done
python3 tools/routing_locality.py ab/route_p1.bin ab/route_p2.bin ab/route_p3.bin --per-layer --json ab/locality.json
# hors du cache : N = le nombre d'emplacements affiché au démarrage (« expert cache N slots »)
python3 tools/routing_locality.py ab/route_p*.bin --resident N --budget 1,2,4,8 --json ab/locality_resident.json
```

Sous PowerShell, remplacer les boucles par `foreach ($p in "p1","p2","p3") { … }` et `ab/route_p*.bin` par la liste
explicite des fichiers.

## Ce qui reste (et pourquoi)

- **#51, le mécanisme lui-même** (copies PCIe ou prélecture L3 lancées avant le routeur) : non fait, volontairement.
  L'issue le conditionne à la mesure de localité, que ce paquet rend possible ; le choix du prédicteur et du budget B
  dépend des chiffres de l'étape 4.
- **#53, mesure de qualité sur pack natif** : le chemin « token » n'existe pas pour un pack natif, donc pas de logits
  de prompt imposé ; seule la comparaison des tokens gloutons (étape 3) est possible. La dérive d'un pack natif en
  fenêtres est de toute façon plus faible (un arrondi par `commit`, voir le tableau).
- **#53, pas GDN non fusionnés** en BF16 : refusés plutôt que convertis ; ce ne sont pas des chemins de production.
- **Historique de convolution en BF16** : non fait, 120 Ko par couche, sans enjeu de trafic.
- **VRAM de l'état** : la moitié libérée de chaque tranche n'est pas rendue (~54 Mo) ; le faire changerait la
  disposition de l'arène et le budget du planificateur pour un gain négligeable.
- **S6** (prefill par couche entière) : hors de ce paquet.
