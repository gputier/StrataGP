# round-sync : moins d'allers-retours hôte↔GPU dans le tour spéculatif (issue #16, O2)

Branche `perf/round-sync`, partie de `perf/base` (b858f9b, moteur 0.1.20). Aucun chiffre GPU n'a été mesuré :
la machine de développement n'a pas de GPU. Tout a été compilé (CUDA 13.0, sm_120) et les tests CPU ont été
lancés ; les tests GPU compilent mais restent à lancer sur la RTX 5090.

## Ce qui change, point par point

| # | Source (issue #16) | Changement | Par défaut | Retour à l'ancien chemin |
|---|---|---|---|---|
| 1 | Brouillon MTP : un graphe + une `cudaStreamSynchronize` par pas | La chaîne entière est **un seul graphe** avec un nœud conditionnel `WHILE` (CUDA 12.4+). Le test `p ≥ min_p` se fait sur le GPU. Un tour MTP = 2 lancements de graphe + 1 synchronisation. | **activé** | `STRATA_OLD_MTP_CHAIN=1` (aussi utilisé automatiquement si le pilote refuse le nœud conditionnel, avec un message sur stderr) |
| 2 | `commit` suivi d'une synchronisation | Dans les deux boucles de décodage (`--serve` et `--spec`), le `commit` n'est plus attendu par l'hôte. La fenêtre suivante suit sur le même flux ; le tour MTP attend un **événement** enregistré après le `commit` (`cudaStreamWaitEvent`). Une seule attente en fin de boucle (`sync_commit`). | **activé** | `STRATA_OLD_COMMIT_SYNC=1` |
| 3 | `apply_pending` : `cudaMemcpy` synchrone de `d_res` | Copie **asynchrone** depuis un tampon épinglé, sur le flux des échanges (derrière les blobs copiés). Aucune fenêtre ne lit `d_res` (la répartition lit `host_res`) ; la fin de boucle attend la copie. `apply_pending(true)` (début de requête serveur) reste synchrone. | **activé** | `STRATA_OLD_RES_UPLOAD=1` |
| 4 | Cache adaptatif : un `std::thread` créé et détruit tous les 4 tours | Un **thread persistant** (`AdaptWorker`, `src/program/round_sync.hpp`) réveillé par une variable de condition. | **activé** | `STRATA_OLD_ADAPT_THREAD=1` |
| 5 | `--pcie-mode dma` : drapeau B par `cudaLaunchHostFunc` | Drapeau B par **`cuStreamWriteValue32`** en file derrière les copies (point d'entrée du pilote résolu à l'exécution, pas d'édition de liens avec `libcuda`). | **activé** (seulement en `--pcie-mode dma`) | `STRATA_OLD_DMA_FLAG=1` (aussi le repli si le pilote refuse l'écriture) |

Aucun de ces changements ne modifie l'arithmétique : mêmes kernels, mêmes valeurs, même ordre des travaux sur le GPU.
Ils sont donc actifs par défaut, avec l'ancien chemin sélectionnable pour l'A/B.

### 1. La chaîne MTP en un graphe (`src/core/mtp.cpp`, `src/kernels/cuda/mtp_chain.cu`)

- `capture_chain()` construit `[init] → WHILE(cond) { stage ; couche MTP (1 ligne) ; mtp_select ; next }` avec
  `build_while_graph()` (capture dans le graphe parent puis dans le corps du nœud via
  `cudaStreamBeginCaptureToGraph`).
- `mtp_chain_init` : `j = 1`, `n = 1`, `cond = (1 < limit && probs[row] ≥ min_p)`.
- `mtp_chain_stage` : copie la fiche de pas et les positions de la ligne `max_t + j − 1` (préparées par l'hôte,
  comme avant) dans une ligne fixe (`max_t`) que le corps lit toujours.
- `mtp_chain_next` : `out[j]`, `out_p[j]`, `j += 1`, `n = j`, `cond = (j < limit && p ≥ min_p)`.
- C'est la boucle hôte `for (j = 1; j < limit && p >= min_p; ++j)` pas pour pas : même comparaison flottante (y
  compris NaN), même compte. `limit` et `min_p` sont lus en mémoire mappée à chaque lancement (pas figés à la capture).
- Les pas écrivent le cache K/V de la couche MTP exactement comme avant (seuls les pas exécutés écrivent).
- Registres (ptxas, sm_120) : `init` 24, `stage` 16, `next` 24, 0 spill.

### 2. `commit` sans synchronisation (`src/core/verify.cpp`, `src/program/generate.cpp`)

- `Verifier::commit(n, err, wait)` : `wait = true` par défaut (inchangé pour la lecture du prompt par fenêtres,
  qui fait un `mtp.prefill` juste après). Les boucles de décodage passent `false`.
- `commit_event()` est donné au drafter (`mtp.order_after`) : l'ordre GPU commit → tour MTP est celui d'avant.
- `run()` synchronise déjà son flux, donc le `commit` précédent est terminé à la fin de chaque fenêtre ;
  `sync_commit()` en fin de boucle rend l'état complet avant les points de contrôle, le hachage d'état, etc.
- Effet sur les statistiques : la ligne `verify window ... commit` ne mesure plus que le lancement.
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

Le drapeau B ne doit jamais redescendre : une écriture directe de l'hôte (couche sans copie) pourrait passer avant
une écriture en file plus petite. Règle : tant qu'aucune levée n'est en file dans la fenêtre, l'hôte écrit
directement (maximum atomique, comme avant) ; dès qu'une levée est en file sur le flux de copie, toutes les
suivantes de la fenêtre y vont aussi (dans l'ordre). Le flux de copie est vidé en fin de fenêtre comme avant.

## Gains attendus

- **ESTIMÉ (audit)** : 1 à 2,5 ms par tour pour l'ensemble O2, soit 6 à 15 % sur une 5090, davantage sous Windows
  (WDDM : 0,3 à 0,4 ms par lancement/synchronisation).
- **HYPOTHÈSE** par point : (1) jusqu'à 2 synchronisations + 2 lancements de moins par tour avec `--spec 4`
  (3 brouillons) — c'est l'essentiel ; (2) une synchronisation de moins par tour ; (3) ~10 à 50 µs tous les
  4 tours ; (4) ~20 à 50 µs tous les 4 tours ; (5) seulement en `--pcie-mode dma`, des dizaines de µs par couche
  avec copies.
- Le banc `mtp_chain_parity --bench` compare, sur un corps synthétique, 6 pas en un lancement contre 6 lancements
  + 6 synchronisations : il donne le coût fixe évité par pas sur la machine.

## Validation sur la RTX 5090

```bash
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON \
      -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_GGML_DIR=/opt/llama.cpp
cmake --build build -j

# tests : la chaîne WHILE contre la boucle hôte (GPU), le thread persistant (CPU)
./build/mtp_chain_parity            # doit afficher PASS
./build/mtp_chain_parity --bench    # coût d'un lancement unique contre un lancement + une synchro par pas
./build/round_sync_test             # PASS
(cd build && ctest --output-on-failure -R "mtp_chain_parity|round_sync_test")
```

A/B : `ARGS` = les arguments de votre configuration (ceux écrits par `setup.py` : `--pack … --native …
--ple-gguf … --expert-profile … --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp … --max-context …`).
Trois prompts tokenisés `p1.txt p2.txt p3.txt` (`--tokens-file`).

```bash
OLD="STRATA_OLD_MTP_CHAIN=1 STRATA_OLD_COMMIT_SYNC=1 STRATA_OLD_RES_UPLOAD=1 STRATA_OLD_ADAPT_THREAD=1 STRATA_OLD_DMA_FLAG=1"
for p in p1 p2 p3; do
  for run in 1 2 3; do
    ./build/strata $ARGS --tokens-file $p.txt --max-new 256 --greedy --adapt-every 100000 > new_${p}_$run.log
    env $OLD ./build/strata $ARGS --tokens-file $p.txt --max-new 256 --greedy --adapt-every 100000 > old_${p}_$run.log
  done
  # 1. tokens identiques
  diff <(grep '^output' new_${p}_1.log) <(grep '^output' old_${p}_1.log) && echo "$p: tokens identiques"
done
# 2. débit et temps de brouillon
grep -H -E '^(decode|mtp|speculation|verify window)' new_*.log old_*.log
```

À vérifier :
1. `output  :` identique entre `new_*` et `old_*` pour chaque prompt (glouton, `--adapt-every 100000`).
2. `decode … tok/s` : new ≥ old ; la ligne `mtp … ms/round drafting` doit baisser (chaîne en un graphe).
3. Aucun message `the one-graph draft chain is unavailable` sur stderr (sinon le pilote refuse le nœud WHILE
   et l'ancien chemin est utilisé).
4. Chaque point seul : relancer `new` avec une seule variable `STRATA_OLD_*=1` pour isoler son effet.
5. Brouillons forcés faux (le test de l'issue) : ajouter `--spec-corrupt 3` aux deux commandes ; les sorties
   doivent rester identiques entre elles et à celles sans `--spec-corrupt`.
6. Mode DMA : refaire l'A/B avec `--pcie-mode dma` (seul le point 5 y change quelque chose) et vérifier l'absence
   de message `cuStreamWriteValue32 refused`.
7. Adaptatif actif (non déterministe, voir l'audit) : un passage avec `--adapt-every 4` pour vérifier l'absence
   d'erreur et la ligne `adaptive tier` ; le serveur : `python3 serve/test_server.py`, puis une conversation réelle.

Profil (facultatif) : `nsys profile --cuda-graph-trace=node ./build/strata $ARGS …` sur 20 tours ; les
`cudaStreamSynchronize` du brouillon passent de 1 + (nombre de pas) à 1 par tour.

## Ce qui reste / risques

- **Rien n'a tourné sur GPU.** Le point le plus délicat est le nœud `WHILE` avec toute la couche MTP dans son corps :
  le corps ne contient que des kernels, des memset et des copies D2D (`cudaMemcpy2DAsync`), types autorisés ; si
  l'instanciation échoue malgré tout, le moteur revient seul à l'ancien chemin (message sur stderr).
  `mtp_chain_parity` teste le mécanisme (init / stage / next, memcpy 2D et memset dans le corps) contre la boucle
  hôte sur 1 120 tours ; il ne rejoue pas la vraie couche MTP (il faudrait le modèle) : l'A/B de tokens ci-dessus
  le fait.
- La variante « toujours calculer les pas et tronquer sur l'hôte » n'a pas été retenue : elle gaspille des pas GPU
  quand la chaîne s'arrête tôt (`--spec-min-p 0.5`).
- Les deux boucles appellent le `Verifier` et le drafter ; la lecture du prompt par fenêtres (`read_windows`) garde
  le `commit` synchrone (le `prefill` MTP le suit sans événement).
- `ms_commit` (ligne `verify window … commit`) n'est plus comparable entre ancien et nouveau chemin.
