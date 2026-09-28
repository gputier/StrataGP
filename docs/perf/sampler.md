# sampler : top-k sans balayage O(k²·V), réparti sur tout le GPU, fin FP64 sur un warp (issue #20, O5)

Branche `perf/sampler`, partie de `perf/base` (b858f9b, moteur 0.1.20). Aucun chiffre GPU n'a été mesuré : la
machine de développement n'a pas de GPU. Tout a été compilé (CUDA 13.0, sm_120) ; `sampler_parity` compile mais
reste à lancer sur la RTX 5090. Seul le chemin **échantillonné** (`temperature > 0`, non glouton) change ; le
glouton (`sampler_greedy_kernel`) n'est pas touché.

## Ce qui change

| # | Source (issue #20) | Changement | Par défaut | Retour à l'ancien chemin |
|---|---|---|---|---|
| 1 | Boucle `taken` : O(k) par logit et par tour, O(k²·V) par ligne sur un seul SM | **Seuil lexicographique** : le tour i ne regarde que les logits strictement *après* le choix du tour i−1 dans l'ordre (valeur décroissante, indice croissant) : `s < prev_v \|\| (s == prev_v && v > prev_i)`. | **activé** | `STRATA_OLD_SAMPLER=1` |
| 2 | Fin FP64 (top_p, température, tirage) calculée par les 1 024 threads | **Un seul warp** : les voies se partagent les `exp` (une par entrée) puis les quotients ; la voie 0 fait seule les deux sommes ordonnées et les deux cumuls. | **activé** | `STRATA_OLD_SAMPLER=1` |
| 3 | Étape 2 de l'issue : top-k en deux étapes sur de nombreux blocs | **Top-k réparti** : chaque ligne est coupée en blocs de 4 096 logits (61 blocs pour 248 320) ; chaque warp garde le top-k de ses 1 024 logits (en registres), le bloc fusionne ses 4 listes ; un warp par ligne fusionne les 61 listes puis fait la fin (point 2). | **activé** (V ≤ 262 144) | `STRATA_SAMPLER_ONE_BLOCK=1` (points 1 et 2 seuls, un bloc par ligne) |

Les trois chemins choisissent **la même liste, dans le même ordre, et le même jeton, au bit près** (par
construction, voir plus bas). Ils sont donc actifs par défaut, les anciens restant sélectionnables pour l'A/B.
`STRATA_OLD_SAMPLER=1` l'emporte sur `STRATA_SAMPLER_ONE_BLOCK=1`. Les variables sont lues une fois par processus.

Le noyau d'un bloc par ligne (`sampler_one_block_kernel`) sert aussi de **repli automatique** du chemin réparti :
flux en cours de capture de graphe (le chemin échantillonné n'est capturé nulle part aujourd'hui), vocabulaire de
plus de 262 144 (64 blocs), plus de 64 lignes par appel (le moteur en échantillonne au plus une fenêtre de
vérification, T ≤ 8), ou mémoire de travail impossible à obtenir.

### Pourquoi c'est identique au bit près

- **Sélection.** L'ancien noyau classe (valeur, id) avant (valeur', id') si valeur > valeur', ou valeur == valeur' et
  id < id' (`>` strict dans le balayage croissant de chaque thread, départage vers l'id le plus petit dans les
  réductions). C'est un ordre total strict (−0 et +0 sont égaux et départagés par l'id, comme avant) ; les choix
  successifs sont donc les premiers de cet ordre, et « pas encore pris » = « strictement après le dernier pris ».
  NaN et −inf ne sont jamais pris (`s > bv` depuis −inf échoue) ; un tour vide donne (−inf, id 0) et tous les tours
  suivants restent vides, dans les deux versions.
- **Réparti.** Les k premiers d'une union sont parmi les k premiers de chaque partie, et la fusion de listes
  ordonnées est ordonnée. Un id n'apparaît que dans une liste (les blocs sont disjoints), donc à chaque tour de
  fusion exactement une tête avance. La réduction de warp par papillon XOR est exacte pour « le premier des deux »
  dans un ordre total strict : toutes les voies finissent avec la même paire.
- **Pénalités.** Chaque bloc construit le masque de présence de *ses* 4 096 logits (512 octets au lieu de 31 Ko) ;
  un logit marqué paie le même comptage exact sur toute la fenêtre ; `apply_penalties` avec un compte nul rend le
  logit inchangé, comme avant.
- **Fin FP64.** Mêmes `exp` des mêmes arguments (fonction déterministe), sommes dans le même ordre (voie 0),
  `cum += e / sum` avec le même quotient arrondi correctement (la division double est IEEE), mêmes comparaisons.
- **Seul écart** : `n_keep == 0` (atteignable seulement avec `min_p > 1`, que le serveur borne à [0, 1] et que la
  CLI ne fixe pas) lisait `sel_ids[-1]` ; il lit `sel_ids[0]`.

### Fichiers

- `src/kernels/cuda/sampler.cu` : `sampled_tail_warp`, `sampler_one_block_kernel`, `sampler_split_part_kernel`
  (étape 1), `warp_merge_lists`, `sampler_split_merge_kernel` (étape 2), choix du chemin et mémoire de travail
  (`split_scratch` : un tampon par (périphérique, flux), agrandi à la demande en doublant au moins, ~0,5 Mo pour
  16 lignes, 2 Mo au plus pour 64 lignes). Un tampon remplacé n'est pas libéré mais mis de côté (un pointeur remis
  à un autre thread hôte sur le même flux peut encore attendre son lancement) : au plus ~4 Mo par flux. Un échec de
  `cudaMalloc` est mémorisé : les appels suivants de cette taille passent directement au noyau d'un bloc sans
  réessayer l'allocation. L'ancien
  `sampler_kernel` est inchangé.
- `src/kernels/sampler_parity.cpp` : fixtures 16, 17 et 18, `--bench`.
- `CMakeLists.txt` : `sampler_parity_one_block` et `sampler_parity_old` (le même binaire avec la variable
  d'environnement de chaque chemin). Chaque test fixe les deux variables (`STRATA_OLD_SAMPLER` et
  `STRATA_SAMPLER_ONE_BLOCK`, à 0 ou 1) : une variable exportée dans le shell pour l'A/B ne change pas le chemin testé.

Registres (ptxas, sm_120, 0 spill) : étape 1 : 80 registres, 128 threads, 2,5 Ko de mémoire partagée ; étape 2 :
52 registres, 32 threads, 33,8 Ko ; un bloc : 59 registres, 1,3 Ko + masque dynamique ; ancien : 55 (inchangé).

## Tests

`sampler_parity` garde ses 15 fixtures (référence hôte, égalité exacte) et en ajoute deux, comparées à un
**miroir de la sémantique du noyau** (−inf et NaN jamais gardés, id 0 quand il ne reste rien) :

- **16 — la liste top-k position par position, avec beaucoup d'égalités.** Chaque ligne contient un +inf : toute
  l'arithmétique de la fin devient NaN, aucune coupe ni tirage n'aboutit et la chaîne rend la *dernière* entrée
  gardée, `sel_ids[k − 1]`. Lancer top_k = 1..64 (et 0, 100, −3 → 64) lit donc la liste entière, ensemble et
  ordre. Lignes : 20 logits à 6,25 et 300 à 6,0 répartis sur tous les blocs, warps et voies ; des zéros signés (−0 aux
  ids pairs) au sommet ; seulement 11 candidats (−inf et NaN ailleurs : l'id 0 sentinelle doit sortir) ; pénalités
  qui posent des jetons pénalisés exactement sur les valeurs d'autres jetons (fenêtre sur les bords de blocs et de
  warps, doublons, ids hors vocabulaire). Vocabulaires 248 320 (dernier bloc partiel), 100 003, 262 144 (le plus
  large réparti), 262 145 (repli un bloc) et 1 000.
- **17 — tirages sous égalités.** Logits au demi-pas (des dizaines de jetons par valeur près du sommet), top_k
  1/20/64, top_p 0,9/1, min_p 0/0,05, température 2,5, pénalités avec et sans (la moitié de chaque fenêtre sur la
  tête de la ligne), 17 lignes (la mémoire de travail est d'abord taillée pour 16 : elle est agrandie), sur le flux
  par défaut et sur un flux créé. Observabilité exigée : des tirages doivent tomber sur un jeton à égalité avec un
  autre jeton gardé.
- **18 — les replis automatiques du chemin par défaut.** Un flux créé, capturé en mode `ThreadLocal` :
  `sample_tokens` (top_k 20, top_p 0,9, pénalités, V = 248 320, 3 lignes) doit passer par le noyau d'un bloc (masque
  de pénalités en mémoire partagée dynamique) dans le graphe ; le graphe rejoué deux fois doit donner les tirages du
  miroir, puis le même flux hors capture (chemin réparti, mémoire de travail allouée après la capture) aussi. Puis
  64 lignes (réparti, le maximum) et 70 lignes (repli un bloc) à V = 512. L'échec d'allocation n'est pas provoqué
  (il faudrait épuiser la VRAM) ; son repli est le même noyau d'un bloc, testé par `sampler_parity_one_block`.

ctest lance le binaire une fois par chemin : `sampler_parity` (réparti), `sampler_parity_one_block`,
`sampler_parity_old`. Les trois doivent passer : même référence, donc mêmes jetons.

Fait ici, sans GPU (hors dépôt) : une émulation hôte voie par voie des nouveaux noyaux (seuil à deux chaînes,
papillon XOR, les deux fusions, sentinelles, fin) comparée à l'ancien noyau transcrit, sur 3 600 cas aléatoires
(égalités, ±0, −inf, NaN, +inf, pénalités, V de 1 à 262 144) : 0 écart, voies toujours d'accord ; des mutations
volontaires (règle d'égalité inversée, seuil `>=`, fusion qui n'avance pas) sont détectées. Puis le source de
`sampler_parity` lié à un runtime CUDA factice dont `sample_tokens` est cette émulation : les 18 groupes de
fixtures passent sur les trois chemins (dont 280 positions sentinelles et 1 057 tirages à égalité ; pour la
fixture 18, le runtime factice enregistre l'appel capturé et le rejoue au lancement du graphe). Cela valide la
logique et les fixtures, pas le code CUDA lui-même : c'est le rôle de `sampler_parity` sur la 5090.

## Gains attendus

- **ESTIMÉ (audit)** : 0,5 à 0,8 ms par fenêtre, soit 2 à 4 % (5070) et 4 à 6 % (5090) sur les requêtes avec
  `temperature > 0`. Le glouton n'est pas concerné.
- **HYPOTHÈSE (calcul, par ligne, top_k 20, V = 248 320)** :
  - ancien : balayage `taken` ~47 M comparaisons en mémoire partagée sur un SM (~0,4 à 0,6 ms) + fin FP64 sur
    1 024 threads (~45 `exp` et ~15 divisions double par thread, au débit FP64 de 2 opérations/cycle/SM :
    ~0,2 ms) + 20 lectures de 1 Mo ;
  - un bloc (`STRATA_SAMPLER_ONE_BLOCK=1`) : 20 passes de 1 Mo sur un SM, ~8 µs chacune : ~0,15 ms ;
  - réparti (défaut) : étape 1 ~3 à 5 µs sur 61 SM, étape 2 + fin ~3 à 6 µs, plus un lancement : ~10 à 20 µs.
- **top_k 64** (`top_k` 0, « off », ou > 64, que le serveur ramène à 64) : l'ancien fait ~500 M comparaisons par
  ligne (plusieurs ms par fenêtre) ; réparti ~20 à 40 µs. C'est là que l'écart est le plus grand.
- Une fenêtre de vérification de T lignes coûtait déjà le temps d'une ligne (un SM par ligne) ; le réparti occupe
  61 × T blocs, donc tout le GPU dès T ≥ 3.

## Validation sur la RTX 5090

```bash
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON \
      -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_GGML_DIR=/opt/llama.cpp
cmake --build build -j

# 1. parité : les trois chemins contre la même référence (doit afficher "sampler_parity OK" trois fois)
(cd build && ctest --output-on-failure -R sampler_parity)

# 2. micro-banc du chemin échantillonné seul (V = 248 320 ; 1, 4, 8 lignes ; top_k 20 et 64 ; us par appel)
./build/sampler_parity --bench
STRATA_SAMPLER_ONE_BLOCK=1 ./build/sampler_parity --bench
STRATA_OLD_SAMPLER=1 ./build/sampler_parity --bench

# 3. (facultatif) accès mémoire et barrières
compute-sanitizer --tool memcheck ./build/sampler_parity --selftest
compute-sanitizer --tool synccheck ./build/sampler_parity --selftest
```

A/B moteur. `ARGS` = les arguments de votre configuration (ceux écrits par `setup.py` : `--pack … --native …
--ple-gguf … --expert-profile … --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp …
--max-context …`) ; `ARGS_NOSPEC` = les mêmes **sans** `--spec`, `--spec-min-p` et `--mtp` : hors serveur, `--spec`
n'échantillonne que le premier jeton (voir l'audit), alors que sans `--spec` chaque jeton passe par le sampler.
Trois prompts tokenisés `p1.txt p2.txt p3.txt` (`--tokens-file`).

```bash
S="--seed 1 --temperature 0.7 --top-k 20 --top-p 0.95"
for p in p1 p2 p3; do
  for run in 1 2 3; do
    # protocole standard, glouton : le sampler glouton n'a pas changé -> mêmes sorties, même débit
    ./build/strata $ARGS --tokens-file $p.txt --max-new 256 --greedy --adapt-every 100000 --stats > g_new_${p}_$run.log
    STRATA_OLD_SAMPLER=1 ./build/strata $ARGS --tokens-file $p.txt --max-new 256 --greedy --adapt-every 100000 \
      --stats > g_old_${p}_$run.log
    # échantillonné : c'est ici que le changement agit
    ./build/strata $ARGS_NOSPEC --tokens-file $p.txt --max-new 256 $S --adapt-every 100000 --stats > s_new_${p}_$run.log
    STRATA_SAMPLER_ONE_BLOCK=1 ./build/strata $ARGS_NOSPEC --tokens-file $p.txt --max-new 256 $S \
      --adapt-every 100000 --stats > s_one_${p}_$run.log
    STRATA_OLD_SAMPLER=1 ./build/strata $ARGS_NOSPEC --tokens-file $p.txt --max-new 256 $S \
      --adapt-every 100000 --stats > s_old_${p}_$run.log
  done
  # tokens identiques entre les chemins (même graine, même sélection, même tirage)
  diff <(grep '^output' g_new_${p}_1.log) <(grep '^output' g_old_${p}_1.log) && echo "$p glouton : identiques"
  diff <(grep '^output' s_new_${p}_1.log) <(grep '^output' s_old_${p}_1.log) && echo "$p échantillonné : identiques"
  diff <(grep '^output' s_one_${p}_1.log) <(grep '^output' s_old_${p}_1.log) && echo "$p un bloc : identiques"
done
# débit et phase "sample" (ms par jeton)
grep -H -E '^decode|token host phases' g_*.log s_*.log
```

À vérifier :
1. `output  :` identique entre `s_new`, `s_one` et `s_old` pour chaque prompt, et entre `g_new` et `g_old`.
2. Ligne `token host phases … sample X` des runs échantillonnés : `s_old` ~0,7 à 1 ms, `s_one` ~0,2 ms, `s_new`
   proche du glouton (quelques centièmes de ms : l'essentiel est alors la synchronisation qui l'entoure).
3. `decode … tok/s` : `s_new` ≥ `s_one` ≥ `s_old` ; `g_new` = `g_old` aux fluctuations près.
4. `--top-k 0` (liste de 64) : refaire un prompt avec `S="--seed 1 --temperature 0.7 --top-k 0 --top-p 0.95"` ;
   l'écart `s_old` / `s_new` sur la phase `sample` doit être de plusieurs ms.
5. Serveur (fenêtres de vérification échantillonnées, T lignes par fenêtre) : lancer le serveur une fois tel quel
   et une fois avec `STRATA_OLD_SAMPLER=1` dans l'environnement du moteur, envoyer la même requête avec
   `"temperature": 0.7, "seed": 1` (moteur avec `--adapt-every 100000`) et comparer texte et tokens/s. Les tests
   Python (`python3 serve/test_server.py`) ne sont pas concernés.

## Ce qui reste

- Aucune mesure GPU : tous les gains ci-dessus sont des estimations à confirmer par `--bench` et l'A/B.
- Les deux étapes sont deux lancements ; les fusionner (dernier bloc qui fusionne, compteur atomique) ou les
  enchaîner en PDL gagnerait encore quelques µs, non fait.
- L'argmax **glouton** multi-blocs fusionné dans la tête (audit, section 7) n'est pas dans cette branche : le
  glouton est inchangé.
