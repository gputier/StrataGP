# ple-io : lectures n-gram (PLE) hors du chemin critique (#15, O1 de l'audit)

Branche `perf/ple-io`, partie de `perf/base` (b858f9b, moteur 0.1.20). Aucune mesure GPU n'a été faite : la
machine de développement n'a pas de GPU. Tout ce qui touche au GPU a été compilé (sm_120, CUDA 13.0) mais pas
exécuté. Les chiffres de gain sont **ESTIMÉ** (calcul de l'audit) ou **HYPOTHÈSE** (à confirmer au profileur).

## Ce qui change

### (a) Fenêtre de vérification : lancer le graphe d'abord (`src/core/verify.cpp`)

Avant : `Verifier::run` lisait les T×16 lignes PLE (`gather_batch`, bloquant) **puis** lançait le graphe, qui
copiait `m_ple_` en tête. Toute la latence SSD s'ajoutait à chaque fenêtre.

Maintenant (défaut) :

1. l'hôte calcule les lignes et **démarre** les lectures (`PleTable::issue_batch`, non bloquant) ;
2. il lance le graphe ;
3. le graphe ne copie plus la PLE en tête : juste avant la couche 1 (dans `pre(1, premier groupe)`), il attend un
   nouveau drapeau mappé `P` (`wait_flag_ge(m_flagP_, 1)`), puis copie `m_ple_` → `ple_` ;
4. l'hôte récupère les lignes (`collect_batch`) **après avoir servi le premier groupe de la couche 0** (les
   experts CPU de la couche 0), écrit `h_ple_`, fait une barrière, puis lève `P`.

Les lectures se recouvrent donc avec l'embedding, la couche 0 GPU et les experts CPU de la couche 0. Dans une
fenêtre scindée (`--spec-split`), `P` est levé après le groupe A de la couche 0 : le groupe B de la couche 0 est
servi pendant que le GPU est déjà en couche 1, comme avant.

Mêmes octets, mêmes kernels, même ordre d'arithmétique : **identique au bit près par construction**, donc actif par
défaut. Si une lecture échoue, la fenêtre va quand même à son terme (un graphe arrêté au milieu bloquerait le flux)
et l'appel échoue ensuite. Un garde RAII récupère un lot resté en vol sur tout retour anticipé.

- Ancien chemin (A/B) : **`STRATA_OLD_PLE_STAGING=1`** (lecture bloquante avant le lancement, copie en tête).
  La valeur est lue : variable absente, vide ou `0` = nouveau chemin ; toute autre valeur = ancien chemin.
- Diagnostic du chien de garde : la ligne `verify window: ...` affiche aussi `ple rows (P)`.
- Répartition des couches sur plusieurs GPU (0.1.21, [../MULTI_GPU.md](../MULTI_GPU.md)) : seul l'étage qui porte
  la couche 1 (celui de CUDA0, les points de coupure valant 2 ou plus) démarre les lectures, lève `P` et attend `P`
  dans son graphe ; les étages suivants ne lisent rien. Un étage qui commencerait à la couche 1 lirait ses lignes
  avant le lancement (l'ancien chemin), car son graphe attendrait `P` avant le premier anneau.

### (b) Linux : lectures concurrentes (`src/platform/direct_file.cpp`)

Avant : `DirectFile::submit` appelait `pread` directement ; les 16 lectures d'un token étaient séquentielles,
`wake()` était vide et `--ple-inflight` n'avait aucun effet.

Maintenant, trois backends avec le même contrat (une requête en file produit exactement une complétion ; `wake`
réveille un `wait` bloqué) :

| backend   | description                                                                              | choix           |
|-----------|------------------------------------------------------------------------------------------|-----------------|
| `uring`   | io_uring par appels système bruts (pas de liburing), anneaux dimensionnés sur `--ple-inflight` | **défaut** si le noyau l'accepte (≥ 5.6) |
| `threads` | pool de `min(--ple-inflight, 16)` threads `pread` (profondeur 16 avec le défaut 64)       | repli automatique si `io_uring_setup` échoue |
| `sync`    | `pread` dans `submit`, appelé sous le verrou du lecteur : l'ancien chemin exact          | A/B uniquement  |

- Sélection : **`STRATA_PLE_IO_BACKEND=uring|threads|sync`** ; **`STRATA_PLE_IO_THREADS=N`** (1..64) règle la
  taille du pool `threads`.
- La ligne `ple io: ...` se termine maintenant par `backend <nom>` (`iocp` sous Windows).
- Windows n'est pas modifié (IOCP était déjà parallèle).

### (b bis) Plus de `submit` sous le verrou du lecteur, sauf avec `sync` (`src/ngram/ple_reader.cpp`)

Avec `uring` et `threads` (et `iocp`), le thread d'E/S de `PleReader` prend les emplacements sous `mu`, puis fait
ses appels `submit` **sans** le verrou (`pump_worker`) : un `submit` ne retient plus `issue`/`collect` du thread
hôte. Le backend `sync` garde **l'ancienne pompe sous verrou** (`pread` sous `mu`, et `finish` relance aussitôt la
lecture suivante) : c'est ce qui en fait un bras B fidèle à `perf/base`.

Corrigé au passage :

- `issue` lisait `ts.pending` après avoir relâché le verrou (course signalée par ThreadSanitizer, sans effet
  visible mais réelle) ;
- injection de délai seulement (`--ple-delay-us`) : le thread d'E/S libère les complétions retenues **avant** de
  pomper, sinon les emplacements qu'elles libèrent restaient vides jusqu'à la complétion suivante ;
- `pump` (mode sans thread d'E/S, et `sync`) rend l'emplacement d'une lecture que `submit` a refusée (aucune
  complétion ne viendrait le libérer) ;
- **erreur de lecture** (lecture échouée ou courte) : avant, `finish` sortait sans rendre l'emplacement et
  `process` abandonnait les complétions suivantes du même lot ; le thread d'E/S tournait alors à vide pour
  toujours et `close()` (donc la fin du programme) ne revenait jamais. Défaut présent depuis `perf/base`, atteint
  seulement si une lecture échoue ; il rendait faux le « l'appel échoue ensuite » de (a), puisque le programme
  restait ensuite bloqué à sa fermeture. Maintenant chaque complétion rend son emplacement, et après une erreur plus rien n'est copié (tout
  `collect` échoue de toute façon, et le tampon d'un ticket en échec peut déjà être libéré). Aucun effet sur le
  chemin sans erreur.

### API

- `PleTable::issue_batch(rows, n_tokens, err)` / `PleTable::collect_batch(out, err)` : `gather_batch` en deux
  moitiés, mêmes octets. Un seul lot en vol par table ; `issue` et `gather_batch` sont refusés pendant ce temps.
- `DirectFile::open(path, err, queue_depth = 16)`, `DirectFile::backend()`, `PleReader::backend()`.

## Gains attendus

- **ESTIMÉ (audit)** : la couche 0 dure ~0,3 à 0,6 ms de GPU, plus les experts CPU de la couche 0 : c'est ce que la
  lecture peut recouvrir. Linux : 1 à 2,5 ms par tour, soit **3 à 7 % (RTX 5070) et 6 à 15 % (RTX 5090)** en
  décodage spéculatif. Windows : 1 à 2 % (seule la partie (a) s'applique).
- **Mesuré ici, sur CPU seulement** (VM, disque virtio, O_DIRECT, fichier synthétique de 45 Mo, 300 tickets de 16
  lignes aléatoires, cache de lignes coupé ; chiffres bruités d'un lancement à l'autre) : `sync` 890 à 1320 µs par
  ticket, `threads` 380 à 580 µs, `uring` 195 à 610 µs (190 à 280 µs une fois chaud). Ce n'est pas un NVMe : sur la
  machine cible, **HYPOTHÈSE** d'un rapport similaire (lectures parallèles au lieu de 16 lectures en série).

## Tests

Sur la machine de développement (CPU seulement) :

- `ple_reader_selftest` : étendu. Il tourne maintenant **pour chaque backend Linux** (`uring`, `threads`, `sync`)
  toute la matrice existante (thread d'E/S oui/non, cache, profondeur 1/8/64, lignes à cheval sur deux pages,
  doublons, hors bornes, gros lot, deux tickets croisés, injection de délai), vérifie que le backend demandé est
  celui utilisé, et compare sur un GGUF synthétique `issue_batch`+`collect_batch` à `gather_batch` (Direct) et au
  mode mmap, au bit près. Nouveau cas : injection de délai avec plus de pages que d'emplacements (4 emplacements,
  32 pages, 1 ms) : octets identiques, 32 lectures, durée ≥ 8 ms. Nouveau cas : erreur de lecture (table tronquée
  sous le lecteur, avec et sans thread d'E/S, pour chaque backend) : le ticket échoue et `close()` revient (sinon
  le test s'arrête en échec au bout de 20 s au lieu de rester bloqué ; vérifié : lié à l'ancien `ple_reader.cpp`,
  il échoue ainsi, « close() hangs after a read error »). Le test écrit maintenant ses fichiers dans
  un sous-dossier unique du dossier temporaire : deux arbres de construction qui le lançaient en même temps
  réécrivaient la même table sous le lecteur de l'autre (c'est ainsi que le défaut ci-dessus est apparu ici). Le
  délai injecté du cas « lecture retenue » passe de 3 à 100 ms : sur une machine surchargée (charge 40 à 60 sur
  4 coeurs pendant ces essais), le thread qui récolte les lectures pouvait rester hors CPU plus de 3 ms, et même
  plus de 20 ms, et plus aucune complétion n'arrivait assez tôt pour être retenue (échec intermittent, sans rapport
  avec le code testé).
  **OK** (aussi sous ThreadSanitizer et AddressSanitizer/UBSan, compilé à la main).
- `platform_memory_test`, `pool_stress`, `expert_multi_test`, `suffix_drafter_test`, `draft_policy_test`,
  `controller_test`, `conv_cache_test` : **OK**.

Nouveau test GPU, **compilé, pas exécuté** : `ple_stage_parity` (ctest). Sur un graphe capturé et rejoué comme
dans le vérificateur, pour T = 1, 3, 8 et 6 fenêtres : le graphe attend bien le drapeau, la copie tardive livre
exactement les octets écrits après le lancement, et la copie en tête (ancien chemin) donne les mêmes lignes au bit
près.

## À valider sur la RTX 5090

```bash
# construction
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON \
      -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j

# tests
ctest --test-dir build -R "ple_stage_parity|ple_reader_selftest" --output-on-failure
# le vrai tableau, par backend (octets identiques au mmap, latences) :
for b in uring threads sync; do
  STRATA_PLE_IO_BACKEND=$b ./build/ple_reader_test --gguf <SHARD2.gguf> --rows 20000 --direct-only
done
```

Benchmark A/B : 3 prompts × 3 exécutions, glouton, 256 tokens, résidence statique. Remplacer `<OPTS>` par la ligne
habituelle (`--pack`, `--ple-gguf`, `--native`, `--expert-profile`, `--expert-cache`, `--spec`, `--mtp`, ...).
`--stats` est nécessaire pour la ligne `ple io`.

Trois bras :

- **old** : `STRATA_OLD_PLE_STAGING=1 STRATA_PLE_IO_BACKEND=sync`, le comportement de `perf/base` (lecture bloquante
  avant le lancement, `pread` en série sous le verrou du lecteur). Seule différence restante : la course sur
  `ts.pending` corrigée dans `issue`, sans effet sur les octets ni sur l'ordre des lectures ;
- **io** : `STRATA_OLD_PLE_STAGING=1`, seulement les lectures concurrentes (b et b bis) ;
- **new** : défaut, tout.

```bash
mkdir -p ab
OPTS="<OPTS> --greedy --max-new 256 --adapt-every 100000 --stats"
run() {  # run <bras> <suffixe> <options en plus> : écrit ab/<bras><suffixe>_<prompt>_<exécution>.txt
  local arm=$1 sfx=$2; shift 2
  local env=""
  case $arm in
    old) env="STRATA_OLD_PLE_STAGING=1 STRATA_PLE_IO_BACKEND=sync" ;;
    io)  env="STRATA_OLD_PLE_STAGING=1" ;;
  esac
  env $env ./build/strata $OPTS "$@" --tokens-file prompts/$p.txt > ab/${arm}${sfx}_${p}_$r.txt 2>&1
}
# 1. débit (fenêtre non scindée, le défaut), 3 exécutions
for p in p1 p2 p3; do for r in 1 2 3; do for a in old io new; do run $a "" ; done; done; done
# 2. exactitude, fenêtre scindée (P est levé après le groupe A de la couche 0)
for p in p1 p2 p3; do r=1; for a in old new; do run $a _split --spec-split; done; done
# 3. exactitude quand le graphe attend VRAIMENT P (chaque lecture dure >= 3 ms, plus que la couche 0)
for p in p1 p2 p3; do r=1; for a in old new; do
  run $a _slow --ple-delay-us 3000; run $a _slowsplit --ple-delay-us 3000 --spec-split
done; done

# tokens générés : chaque fichier doit donner la même ligne "output  :" que old (même prompt, même découpage)
for p in p1 p2 p3; do
  for f in ab/io_${p}_*.txt ab/new_${p}_*.txt ab/old_${p}_[23].txt; do
    cmp -s <(grep "^output  :" ab/old_${p}_1.txt) <(grep "^output  :" $f) || echo "DIFF $f"
  done
  for s in _split _slow _slowsplit; do
    cmp -s <(grep "^output  :" ab/old${s}_${p}_1.txt) <(grep "^output  :" ab/new${s}_${p}_1.txt) \
      || echo "DIFF new${s} $p"
  done
done
grep -H "ple io:" ab/*.txt       # "blocked ... ms" doit chuter en new ; "backend uring" en io/new, "sync" en old
grep -H "^decode \|tokens per round" ab/*.txt   # débit et tokens par tour
```

Empreinte d'état : `STRATA_STATE_HASH` n'est imprimé qu'en mode `--serve` (après chaque requête, cache de prompts
actif, le défaut). Même requête dans les deux bras, fenêtre scindée ou non :

```bash
for p in p1 p2 p3; do
  ids=$(tr -s ' ,\n\r\t' ',' < prompts/$p.txt | sed 's/^,//; s/,$//')
  for split in "" --spec-split; do
    for a in old new; do
      env=""; [ $a = old ] && env="STRATA_OLD_PLE_STAGING=1 STRATA_PLE_IO_BACKEND=sync"
      printf 'GEN 256 %s\nQUIT\n' "$ids" | env $env STRATA_STATE_HASH=1 \
        ./build/strata <OPTS> --adapt-every 100000 $split --serve 2>&1 | grep "STATE_HASH" > ab/hash_${a}${split}_$p.txt
    done
    cmp -s ab/hash_old${split}_$p.txt ab/hash_new${split}_$p.txt || echo "HASH DIFF $p $split"
  done
done
```

Critères :

- aucune ligne `DIFF` ni `HASH DIFF` : tokens générés et empreintes identiques entre `old`, `io` et `new`, avec et
  sans `--spec-split`, et avec `--ple-delay-us 3000` (le cas où le graphe attend à coup sûr `P` devant la
  couche 1 : il valide le placement de l'attente et la barrière hôte) ;
- `blocked ... ms` de la ligne `ple io` nettement plus bas en `new` (il ne mesure plus que l'attente résiduelle
  après la couche 0) ;
- débit : attendu +6 à 15 % en `new` sur la 5090 (ESTIMÉ) ;
- pour isoler (a) sous Windows : `STRATA_OLD_PLE_STAGING=1` contre défaut.

## Ce qui reste

- `ple_stage_parity` ne teste la poignée de main (attente de `P`, copie tardive) que sur un graphe synthétique de
  3 noeuds : `Verifier::run` et `record_window` eux-mêmes (P placé dans `pre(1, groupe 0)`, levé après `post` de
  `k == 0`, fenêtre scindée, garde `PleGuard`, erreur de lecture rendue après la synchronisation) ne sont couverts
  que par l'A/B ci-dessus. Un test au niveau du vérificateur demanderait des poids (un `WeightTable` complet) :
  pas faisable sans modèle ni GPU ici. **À faire avant fusion : l'A/B complet ci-dessus sur la 5090.**
- Le profil `nsys` (`--cuda-graph-trace=node`) doit confirmer que le noeud `wait_flag_ge` avant la couche 1 attend
  peu ou pas ; s'il attend, le SSD est plus lent que la couche 0 + experts CPU, et il faudrait lancer les lectures
  de la fenêtre suivante plus tôt (dès que ses tokens sont connus, pendant le brouillon MTP), ce qui touche au code
  spéculatif de `generate.cpp` (hors de ce paquet).
- `uring` fait un `io_uring_enter` par lecture (~4 à 13 µs ici, sur le thread d'E/S, pas sur le thread hôte). Un
  envoi groupé (toutes les SQE d'un ticket puis un seul `enter`) est possible mais demande une API de lot dans
  `DirectFile` ; non fait.
- Le chemin de décodage un token à la fois (`ple_issue_token`/`ple_finish_token`) et le préremplissage profitent
  des lectures concurrentes (b) sans autre changement ; leur ordonnancement n'est pas modifié.
