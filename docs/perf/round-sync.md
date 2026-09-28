# round-sync : moins d'allers-retours hôte↔GPU dans le tour spéculatif (issue #16, O2)

Branche `perf/round-sync`, partie de `perf/base` (b858f9b, moteur 0.1.20). Aucun chiffre GPU n'a été mesuré :
la machine de développement n'a pas de GPU. Tout a été compilé (CUDA 13.0, sm_120) et les tests CPU ont été
lancés ; les tests GPU compilent mais restent à lancer sur la RTX 5090. **Aucun de ces chemins n'a encore tourné
sur un GPU : la validation ci-dessous (tokens identiques, `STRATA_MTP_CHAIN_CHECK`, `--spec-corrupt 3`,
`--pcie-mode dma`) est à faire avant de fusionner.**

## Ce qui change, point par point

| # | Source (issue #16) | Changement | Par défaut | Retour à l'ancien chemin |
|---|---|---|---|---|
| 1 | Brouillon MTP : un graphe pour la manche, puis un graphe + une `cudaStreamSynchronize` par pas | La manche **et** toute la chaîne sont **un seul graphe** : le travail de la manche, `mtp_chain_init`, puis un nœud conditionnel `WHILE` (CUDA 12.4+). Le test `p ≥ min_p` se fait sur le GPU. Un tour MTP = **1 lancement + 1 synchronisation**, y compris quand le premier brouillon arrête déjà la chaîne. | **activé** | `STRATA_OLD_MTP_CHAIN=1` (aussi utilisé automatiquement si le pilote refuse le nœud conditionnel, avec un message sur stderr). Contrôle : `STRATA_MTP_CHAIN_CHECK=N` (voir plus bas). |
| 2 | `commit` suivi d'une synchronisation | Dans les deux boucles de décodage (`--serve` et `--spec`), le `commit` n'est plus attendu par l'hôte. La fenêtre suivante suit sur le même flux ; le tour MTP attend un **événement** enregistré après le `commit` (`cudaStreamWaitEvent`). Une seule attente en fin de boucle (`sync_commit`). | **activé** | `STRATA_OLD_COMMIT_SYNC=1` |
| 3 | `apply_pending` : `cudaMemcpy` synchrone de `d_res` | Copie **asynchrone** depuis un tampon épinglé, sur le flux des échanges (derrière les blobs copiés). Aucune fenêtre ne lit `d_res` (la répartition lit `host_res`) ; la fin de boucle attend la copie. `apply_pending(true)` (début de requête serveur) reste synchrone. | **activé** | `STRATA_OLD_RES_UPLOAD=1` |
| 4 | Cache adaptatif : un `std::thread` créé et détruit tous les 4 tours | Un **thread persistant** (`AdaptWorker`, `src/program/round_sync.hpp`) réveillé par une variable de condition. | **activé** | `STRATA_OLD_ADAPT_THREAD=1` |
| 5 | `--pcie-mode dma` : drapeau B par `cudaLaunchHostFunc` | Drapeau B par **`cuStreamWriteValue32`** en file derrière les copies (point d'entrée du pilote résolu à l'exécution, pas d'édition de liens avec `libcuda`), et une règle d'ordre des levées qui corrige une course de l'ancien code (voir 5). | **activé** (seulement en `--pcie-mode dma`) | `STRATA_OLD_DMA_FLAG=1` : fonctions hôtes au lieu de l'écriture sur le flux (aussi le repli si le pilote refuse l'écriture), **mais la nouvelle règle d'ordre est gardée** (voir 5) |

Aucun de ces changements ne modifie l'arithmétique : mêmes kernels, mêmes valeurs, même ordre des travaux sur le GPU.
Ils sont donc actifs par défaut, avec l'ancien chemin sélectionnable pour l'A/B.

### 1. La manche et la chaîne MTP en un graphe (`src/core/mtp.cpp`, `src/kernels/cuda/mtp_chain.cu`)

- `capture_chain(T)` construit, pour chaque taille de fenêtre T utilisée, avec `build_while_graph()` (capture dans
  le graphe parent puis dans le corps du nœud via `cudaStreamBeginCaptureToGraph`) :
  `[manche T ; init] → WHILE(cond) { stage ; couche MTP (1 ligne) ; mtp_select ; next }`.
  La manche est enregistrée par `record_round(T)`, exactement le code de l'ancien graphe `capture_round(T)`
  (qui l'appelle aussi) : mêmes copies, même couche, même `mtp_select` qui laisse le premier brouillon et sa
  probabilité dans `probs_[row]`.
- `mtp_chain_init` : `j = 1`, `n = 1`, `cond = (1 < limit && probs[row] ≥ min_p)`.
- `mtp_chain_stage` : copie la fiche de pas et les positions de la ligne `max_t + j − 1` (préparées par l'hôte,
  comme avant) dans une ligne fixe (`max_t`) que le corps lit toujours.
- `mtp_chain_next` : `out[j]`, `out_p[j]`, `j += 1`, `n = j`, `cond = (j < limit && p ≥ min_p)`.
- C'est la boucle hôte `for (j = 1; j < limit && p >= min_p; ++j)` pas pour pas : même comparaison flottante (y
  compris NaN), même compte. `limit` et `min_p` sont lus en mémoire mappée à chaque lancement (pas figés à la capture).
- Les pas écrivent le cache K/V de la couche MTP exactement comme avant (seuls les pas exécutés écrivent).
- Tour dont le premier brouillon échoue `min_p` (fréquent avec `--spec-min-p 0.5`) : un lancement et une
  synchronisation comme avant ; le surcoût GPU est le kernel `init` (1 thread) et l'évaluation du nœud
  conditionnel, quelques µs. (La première version de la branche lançait deux graphes par tour, manche puis
  chaîne : ce cas pouvait devenir un peu plus lent sous WDDM ; ce n'est plus le cas.)
- Registres (ptxas, sm_120) : `init` 24, `stage` 16, `next` 24, 0 spill (kernels inchangés par ce correctif).

**Contrôle sur GPU, `STRATA_MTP_CHAIN_CHECK=N`** (débogage, désactivé par défaut) : pour les N premiers tours, le
moteur lance le graphe unique, puis l'ancien chemin (graphe de manche + un graphe et une synchronisation par pas)
sur les mêmes données préparées, et compare nombre de brouillons, brouillons et probabilités **bit à bit**.
Chaque écart est écrit sur stderr (`strata mtp: chain check, round K DIFFERS (chain/step: n/m drafts): …`), et au
N-ième tour : `strata mtp: chain check: N rounds compared, D differ`. Le résultat gardé est celui de l'ancien
chemin : le contrôle ne change pas la sortie. Rejouer la manche est sans effet de bord : mêmes entrées, mêmes
cellules K/V réécrites avec les mêmes valeurs ; la manche ne lit pas les cellules des brouillons (attention
causale jusqu'à `p + a`) ; si les deux chemins différaient sur le nombre de pas, les cellules en trop sont au-delà
de la position acceptée et réécrites par la manche suivante avant toute lecture, comme les brouillons rejetés.
Le coût du brouillon double pendant ces N tours : ne pas mesurer de débit avec ce contrôle.

### 2. `commit` sans synchronisation (`src/core/verify.cpp`, `src/program/generate.cpp`)

- `Verifier::commit(n, err, wait)` : `wait = true` par défaut (inchangé pour la lecture du prompt par fenêtres,
  qui fait un `mtp.prefill` juste après). Les boucles de décodage passent `false`.
- `commit_event()` est donné au drafter (`mtp.order_after`) : l'ordre GPU commit → tour MTP est celui d'avant.
- `run()` synchronise déjà son flux, donc le `commit` précédent est terminé à la fin de chaque fenêtre ;
  `sync_commit()` en fin de boucle rend l'état complet avant les points de contrôle, le hachage d'état, etc.
- Statistiques : quand les `commit` ne sont pas attendus, la ligne `verify window` affiche
  **`commit launch X ms/round`** (le lancement seul) au lieu de `commit X ms/round` (lancement + attente, ancien
  chemin ou `STRATA_OLD_COMMIT_SYNC=1`). Les deux valeurs ne se comparent pas ; comparer `decode … tok/s`.
- Gain réel : quand le MTP est utilisé, son tour attend l'événement puis synchronise son propre flux ; le gain est
  donc une synchronisation de moins et le recouvrement du travail hôte (affichage, suffixes, préparation MTP) avec
  le `commit`. Sans MTP (suffixes seuls), l'hôte enchaîne directement sur la fenêtre suivante.

### 3. `d_res` asynchrone (`ResidencyUpload`, `src/program/round_sync.hpp`)

Lecteurs de `d_res` : le graphe « token » (`session.cpp`, `moe_hit_select`) et le chemin de prefill emprunteur.
Aucun ne tourne dans les boucles de décodage (le `Verifier` vérifie seulement que `d_res` n'est pas nul). Les
écritures synchrones suivantes (`refill`, `lend`) arrivent après la fin de boucle, qui attend la copie.

### 4. Thread persistant (`AdaptWorker`)

`start()` / `wait()` remplacent `std::thread(...)` / `join()` aux mêmes endroits. Le mutex donne à l'appelant tout
ce que la tâche a écrit (`host_res`, `pending`, `adapt_ok`). Testé par `round_sync_test` (et sous ThreadSanitizer).

### 5. Drapeau B en `--pcie-mode dma`

Le drapeau B ne doit jamais redescendre ni passer devant des copies : une écriture directe de l'hôte (couche sans
copie) pourrait passer avant une levée en file plus petite. Règle : tant qu'aucune levée n'est en file dans la
fenêtre, l'hôte écrit directement (maximum atomique, comme avant) ; dès qu'une levée est en file sur le flux de
copie, toutes les suivantes de la fenêtre y vont aussi (dans l'ordre). Le flux de copie est vidé en fin de fenêtre
comme avant.

**Cette règle corrige une course latente de l'ancien code.** Dans une fenêtre coupée en deux groupes, l'ancien
code écrivait directement le drapeau pour (l + 1, A) (pas de copie) pendant que la fonction hôte de (l, B) attendait
encore ses copies : le maximum atomique empêchait le drapeau de redescendre, mais la valeur plus grande libérait
déjà l'attente GPU de (l, B), qui pouvait lire sa zone de transit avant l'arrivée des copies.

**Conséquence pour l'A/B** : `STRATA_OLD_DMA_FLAG=1` remplace seulement l'écriture sur le flux par des fonctions
hôtes ; il **garde la règle**. Dès la première levée en file d'une fenêtre, toutes les couches suivantes (même sans
copie) passent donc par un `cudaLaunchHostFunc` en file derrière les copies, là où l'ancien code écrivait tout de
suite. Ce mode est donc un peu plus lent que le vrai code d'avant #16 en `--pcie-mode dma` :
- nouveau contre `STRATA_OLD_DMA_FLAG=1` mesure l'effet de `cuStreamWriteValue32` seul ;
- la référence exacte d'avant #16 (avec sa course) est un build de `perf/base` (b858f9b). Aucun interrupteur ne
  la rétablit ici, volontairement : il réintroduirait la course.

## Gains attendus

- **ESTIMÉ (audit)** : 1 à 2,5 ms par tour pour l'ensemble O2, soit 6 à 15 % sur une 5090, davantage sous Windows
  (WDDM : 0,3 à 0,4 ms par lancement/synchronisation).
- **HYPOTHÈSE** par point : (1) avec `--spec 4` (3 brouillons), jusqu'à 3 lancements et 2 synchronisations de moins
  par tour quand la chaîne va au bout, 0 de moins (et pas de plus) quand elle s'arrête au premier brouillon —
  c'est l'essentiel ; (2) une synchronisation de moins par tour ; (3) ~10 à 50 µs tous les 4 tours ; (4) ~20 à
  50 µs tous les 4 tours ; (5) seulement en `--pcie-mode dma`, des dizaines de µs par couche avec copies.
- Le banc `mtp_chain_parity --bench` prépare les données une fois, puis ne chronomètre que lancements et
  synchronisations : 6 pas en un lancement contre 6 lancements + 6 synchronisations. **Chiffres indicatifs
  seulement** : corps synthétique, et le côté « un lancement par pas » relance ce même graphe avec `limit = 2`
  (tête et nœud conditionnel compris) au lieu d'un graphe simple par pas.

## Validation sur la RTX 5090

```bash
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON \
      -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_GGML_DIR=/opt/llama.cpp
cmake --build build -j

# tests : la chaîne WHILE (après un travail de tête, comme la manche) contre la boucle hôte (GPU) ;
# le thread persistant (CPU)
./build/mtp_chain_parity            # doit afficher PASS
./build/mtp_chain_parity --bench    # indicatif : un lancement unique contre un lancement + une synchro par pas
./build/round_sync_test             # PASS
(cd build && ctest --output-on-failure -R "mtp_chain_parity|round_sync_test")
```

A/B : `ARGS` = les arguments de votre configuration (ceux écrits par `setup.py` : `--pack … --native …
--ple-gguf … --expert-profile … --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp … --max-context …`).
Trois prompts tokenisés `p1.txt p2.txt p3.txt` (`--tokens-file`).

```bash
OLD="STRATA_OLD_MTP_CHAIN=1 STRATA_OLD_COMMIT_SYNC=1 STRATA_OLD_RES_UPLOAD=1 STRATA_OLD_ADAPT_THREAD=1 STRATA_OLD_DMA_FLAG=1"
RUN="./build/strata $ARGS --max-new 256 --greedy --adapt-every 100000"

# 0. le graphe unique contre l'ancien chemin, sur le vrai modèle, 50 premiers tours (doit finir par "0 differ")
for p in p1 p2 p3; do
  STRATA_MTP_CHAIN_CHECK=50 $RUN --tokens-file $p.txt > check_$p.log 2> check_$p.err
  grep "chain check" check_$p.err      # attendu : "chain check: 50 rounds compared, 0 differ", aucune ligne DIFFERS
done

# 1. A/B nouveau / ancien, 3 prompts x 3 passages
for p in p1 p2 p3; do
  for run in 1 2 3; do
    $RUN --tokens-file $p.txt > new_${p}_$run.log 2> new_${p}_$run.err
    env $OLD $RUN --tokens-file $p.txt > old_${p}_$run.log 2> old_${p}_$run.err
  done
  diff <(grep '^output' new_${p}_1.log) <(grep '^output' old_${p}_1.log) && echo "$p: tokens identiques"
  diff <(grep '^output' new_${p}_1.log) <(grep '^output' check_$p.log) && echo "$p: identiques au contrôle"
done
grep -H -E '^(decode|mtp|speculation|verify window)' new_*.log old_*.log
grep -l -E "one-graph draft chain is unavailable|cuStreamWriteValue32 refused" new_*.err   # attendu : rien

# 2. brouillons forcés faux (le test de l'issue) : sorties identiques entre elles et à celles de 1.
for p in p1 p2 p3; do
  $RUN --tokens-file $p.txt --spec-corrupt 3 > newc_$p.log
  env $OLD $RUN --tokens-file $p.txt --spec-corrupt 3 > oldc_$p.log
  diff <(grep '^output' newc_$p.log) <(grep '^output' oldc_$p.log) && \
  diff <(grep '^output' newc_$p.log) <(grep '^output' new_${p}_1.log) && echo "$p: corrupt ok"
done

# 3. mode DMA (seul le point 5 y change quelque chose en plus)
for p in p1 p2 p3; do
  $RUN --tokens-file $p.txt --pcie-mode dma > newd_$p.log 2> newd_$p.err
  env $OLD $RUN --tokens-file $p.txt --pcie-mode dma > oldd_$p.log
  STRATA_OLD_DMA_FLAG=1 $RUN --tokens-file $p.txt --pcie-mode dma > flagd_$p.log
  diff <(grep '^output' newd_$p.log) <(grep '^output' oldd_$p.log) && echo "$p: dma ok"
done
grep -H '^decode' newd_*.log flagd_*.log oldd_*.log   # new vs flagd = effet de cuStreamWriteValue32 seul
grep -l "cuStreamWriteValue32 refused" newd_*.err     # attendu : rien

# 4. cas « zéro pas » : presque tous les tours s'arrêtent au premier brouillon
$RUN --tokens-file p1.txt --spec-min-p 0.95 > newz.log
env STRATA_OLD_MTP_CHAIN=1 $RUN --tokens-file p1.txt --spec-min-p 0.95 > oldz.log
grep -H -E '^(decode|mtp)' newz.log oldz.log         # attendu : mtp ms/round new <= old
```

À vérifier :
1. `STRATA_MTP_CHAIN_CHECK` : `0 differ` sur les trois prompts (c'est le seul test qui exécute la vraie couche MTP
   dans le nœud `WHILE` et la compare à l'ancien chemin, tour par tour).
2. `output  :` identique entre `new_*`, `old_*` et `check_*` pour chaque prompt (glouton, `--adapt-every 100000`).
3. `decode … tok/s` : new ≥ old ; la ligne `mtp … ms/round drafting` doit baisser (manche + chaîne en un graphe),
   y compris dans le cas « zéro pas » (4).
4. Aucun message `the one-graph draft chain is unavailable` ni `cuStreamWriteValue32 refused` sur stderr.
5. Chaque point seul : relancer `new` avec une seule variable `STRATA_OLD_*=1` pour isoler son effet.
6. La ligne `verify window` affiche `commit launch` (nouveau) ou `commit` (ancien) : ne pas comparer ces deux
   nombres (voir 2).
7. Adaptatif actif (non déterministe, voir l'audit) : un passage avec `--adapt-every 4` pour vérifier l'absence
   d'erreur et la ligne `adaptive tier` ; le serveur : `python3 serve/test_server.py`, puis une conversation réelle.

Profil (facultatif) : `nsys profile --cuda-graph-trace=node ./build/strata $ARGS …` sur 20 tours ; le brouillon
passe de 1 + (nombre de pas) lancements et synchronisations à 1 de chaque par tour.

## Ce qui reste / risques

- **Rien n'a tourné sur GPU.** Le point le plus délicat est le nœud `WHILE` avec toute la couche MTP dans son corps :
  le corps ne contient que des kernels, des memset et des copies D2D (`cudaMemcpy2DAsync`), types autorisés ; si
  l'instanciation échoue malgré tout, le moteur revient seul à l'ancien chemin (message sur stderr).
  `mtp_chain_parity` teste le mécanisme (travail de tête avant `init`, init / stage / next, memcpy 2D et memset dans
  le corps) contre la boucle hôte sur 1 120 tours ; il ne rejoue pas la vraie couche MTP (il faudrait le modèle) :
  `STRATA_MTP_CHAIN_CHECK` et l'A/B de tokens ci-dessus le font.
- Non couverts par un test automatique (à valider par l'A/B) : l'ordre du `commit` asynchrone
  (`commit_ev_` / `order_after` / `sync_commit`), `ResidencyUpload`, et le chemin `cuStreamWriteValue32`.
- La variante « toujours calculer les pas et tronquer sur l'hôte » n'a pas été retenue : elle gaspille des pas GPU
  quand la chaîne s'arrête tôt (`--spec-min-p 0.5`).
- Les deux boucles appellent le `Verifier` et le drafter ; la lecture du prompt par fenêtres (`read_windows`) garde
  le `commit` synchrone (le `prefill` MTP le suit sans événement).
- Mémoire : un graphe instancié par taille de fenêtre T (au plus 8), chacun avec la manche et le corps ; c'est
  l'équivalent des anciens graphes de manche et de pas, qui ne sont plus capturés (sauf `STRATA_OLD_MTP_CHAIN=1`,
  repli ou contrôle).
