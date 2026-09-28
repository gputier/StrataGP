# prefill-host : MoE du prefill, indexeur, re-remplissage, points de reprise, chercheur de suffixe, démarrage

Issues #36 (P1), #40 (P5), #42 (P7), #45 (E2), #46 (E3), #48 (E5) de l'audit (`docs/AUDIT-PERF.md`).

Branche `perf/prefill-host`, partie de `perf/base` (b858f9b, moteur 0.1.20). **Aucune mesure GPU n'a été faite** : la
machine de développement n'a pas de GPU. Tout ce qui touche au GPU a été compilé (sm_120, CUDA 13.0, sans
débordement de registres d'après `ptxas -v`) mais **pas exécuté**. Les gains sont **ESTIMÉ** (calcul) ou
**HYPOTHÈSE** (à confirmer au profileur) ; les seuls chiffres **MESURÉ** ci-dessous viennent du CPU du conteneur de
développement, pas de la machine cible.

## Résumé

| Issue | Changement | Par défaut | Interrupteur (ancien chemin / option) | Identique au bit près | Test |
|---|---|---|---|---|---|
| #40 P5 | Ajout des clés de l'indexeur QSA pour tout un morceau en 2 lancements (au lieu d'un par token) | **oui** | `STRATA_OLD_IDX_APPEND=1` | oui, par construction | `qsa_indexer_chunk_parity` (GPU) |
| #36 P1 | Paires (token, k) groupées par expert sur le GPU ; attente hôte recouverte par l'expert partagé ; marche « par groupe » (une attente, un rassemblement, une libération par groupe de 16) ; copies fusionnées des experts contigus | **oui** | `STRATA_OLD_MOE_GROUP=1` ; `STRATA_PREFILL_COALESCE=0` (copies une par une) | oui, par construction | `moe_group_parity` (GPU) |
| #46 E3 | Chercheur de suffixe : effacement paresseux par époque, extension quand la requête prolonge l'historique | **oui** | `STRATA_OLD_SFX_RESET=1` | oui (mêmes propositions) | `suffix_drafter_test` (CPU, **OK ici**) |
| #45 E2 | Points de reprise : pool de tampons épinglés réutilisés + copies asynchrones sur un flux dédié | **oui** | `STRATA_OLD_CKPT=1` | oui (mêmes octets) | A/B serveur (voir plus bas) |
| #48 E5 | Remplissage du profil au démarrage mis en file sur un flux (une seule attente) ; relecture du premier **et du dernier** emplacement | **oui** | `STRATA_OLD_PROFILE_FILL=1` | oui (mêmes octets) | la relecture elle-même, au démarrage |
| #42 P7 | Re-remplissage **asynchrone** des emplacements prêtés au prefill (mécanisme `pending` + événement du cache adaptatif) | **non (opt-in)** | `STRATA_ASYNC_REFILL=1` | **non** (voir #42) | A/B TTFT |

## #40 (P5) : l'indexeur QSA ajouté par morceau

**Avant.** `prefill.cpp` lançait `native_qsa_indexer_append` une fois par token : 8 192 × 12 couches ≈ 98 000
lancements par morceau, chacun avec ses vérifications côté hôte (7 plages, 21 tests de recouvrement).

**Maintenant.** `native_qsa_indexer_append_chunk` (dans `src/kernels/cuda/native_qsa_indexer.cu`, donc compilé avec les
mêmes options `--use_fast_math` que le noyau par token) :

- 1er lancement : un bloc par *événement* du morceau (la clé de secours si le morceau commence à la cellule 0, puis
  chaque bloc de 4 cellules que le morceau complète). Chaque bloc refait **instruction pour instruction** l'arithmétique
  du noyau par token (arrondi F16 des clés brutes, somme `__fadd_rn` dans le même ordre, réduction en papillon
  identique, `rsqrtf`, RoPE). Il lit la queue telle que le morceau l'a trouvée pour les cellules d'avant, et les clés
  du morceau pour les autres. Aucun événement ne lit ce qu'un autre écrit, donc ils tournent en parallèle.
- 2e lancement (un bloc) : ce que seul le *dernier* événement de la boucle laissait (la ligne de secours après le
  dernier bloc complété, `block_pos`) et la queue (la dernière cellule de chaque emplacement).
- Les cellules au-delà de la capacité sont ignorées, comme dans le noyau par token.

J'ai comparé le PTX : la séquence arithmétique du nouveau noyau est celle du noyau par token (mêmes `add.rn`, même
`fma.rn` de 0,25, même papillon, même `div.approx`/`rsqrt.approx`, mêmes `sin/cos.approx`, même motif
`mul/mul/sub` et `mul/fma` pour la rotation). Le PTX du noyau par token est inchangé octet pour octet (il sert
toujours au décodage et à la fenêtre de vérification). **Le vrai contrôle reste `qsa_indexer_chunk_parity` sur le
GPU** : deux jeux de tampons partant du même bruit, l'un rempli token par token, l'autre par morceaux, comparés octet
par octet après chaque morceau (toutes les phases par rapport au bloc de 4, morceaux de 1 à 8 192, table M-RoPE,
cellules au-delà de la capacité, capacité non multiple de 4).

**Gain (HYPOTHÈSE, audit) :** 2 à 6 % du débit de prefill ; 98 000 lancements → 24 par morceau. Visible dans la
phase `qsa indexer` de `STRATA_PREFILL_TIMING=1`.

## #36 (P1) : le MoE du prefill

### Ce qui a été fait (par défaut, identique au bit près)

1. **Regroupement sur le GPU** (`src/prefill/moe_group.cu`, nouveau fichier). Le même tri par comptage que l'hôte :
   comptes par tranche de 2 048 paires (atomiques en mémoire partagée), un scan (un thread par expert), puis chaque
   tranche placée **dans l'ordre des paires** par un warp (`__match_any_sync` : le rang d'une paire parmi les lanes
   plus basses de même expert). Les tables `slot`/`src` sont donc exactement celles du tri hôte, entrée par entrée :
   c'est indispensable, car l'ordre des lignes d'un expert change le découpage *stream-k* de MMQ et donc le dernier
   bit du résultat. Seuls les 512 comptes (+ un drapeau « id hors bornes ») reviennent, par une copie asynchrone vers
   de la mémoire épinglée ; le brouillon est pris dans `Dm` (écrasé ensuite par les produits).
2. **L'attente hôte est recouverte** : l'événement est enregistré juste après le regroupement, *avant* l'expert
   partagé ; l'hôte attend l'événement (plus `cudaStreamSynchronize`) pendant que le GPU calcule l'expert partagé.
   Les bornes des groupes montent depuis un tampon épinglé (une source paginable peut bloquer l'hôte jusqu'à ce que
   le flux atteigne la copie).
3. **Marche par groupe** (morceaux ≥ 2 048 tokens en `stream_all`, toutes les couches en MMQ — sinon l'ancienne
   marche par expert reste utilisée) :
   - un groupe de 16 experts routés attend **une fois** la copie de sa dernière entrée (le flux de copie est en
     ordre), est rassemblé en **un seul lancement** (`gather_native_batch` / `gather_strata_q2_batch`, octet pour
     octet les rassemblements par expert — vérifié par `moe_group_parity`), et rend ses emplacements de l'anneau (et
     ceux des experts non routés d'avant) avec **un seul événement** ;
   - jusqu'à 16 experts consécutifs d'une couche, contigus dans l'arène épinglée, arrivent en **une copie 2D**
     (`cudaMemcpy2DAsync`, pas de destination = pas de l'anneau) quand les emplacements de l'anneau sont également
     espacés (région empruntée) ; `STRATA_PREFILL_COALESCE=0` revient à une copie par expert ;
   - les produits MMQ sont lancés **exactement** comme avant (mêmes groupes, même `max_rows`, mêmes bornes), d'où
     l'identité au bit près ;
   - l'événement `copied` d'une copie fusionnée est enregistré sur le **dernier** emplacement de la série (le premier
     peut être rendu et re-rempli dès la libération du lot qui contient la première entrée, alors qu'un lot suivant
     attend encore la série : il aurait attendu une copie jusqu'à `ring` entrées plus loin) ;
   - en fin de morceau, chaque emplacement retrouve son propre événement `used` (ce qu'attendent la marche par
     expert et les petits morceaux) ; un morceau interrompu synchronise les deux flux au morceau suivant.
4. `Prefill::run` synchronise aussi le flux de copie avant de rendre la main, **sur chaque retour** (garde de
   portée : succès, erreur, et `cancelled` quand un STOP arrive entre deux morceaux) : la copie d'un expert non routé n'était
   jamais attendue et atterrissait dans les emplacements empruntés que l'appelant re-remplit ensuite (course
   théorique déjà présente, nécessaire pour #42).

Appels d'API CUDA par couche MoE (ESTIMÉ, 8 192 tokens, ~480 experts diffusés) : ~3 000 → ~400 (par groupe : une
attente, un rassemblement, un événement, 2 `memset`, les lancements MMQ ; côté copie : ~1 copie + 1 attente + 1
événement par série de 16).

**Mesuré ici (CPU du conteneur)** : le tri hôte supprimé coûtait 0,18 ms par couche, soit 8,7 ms par morceau de
8 192 (48 couches) — faible ; l'essentiel attendu vient des appels d'API par expert et de l'attente non recouverte.

**Gain (HYPOTHÈSE, audit) :** 5 à 10 % du débit de prefill, davantage sur la 5090 (le GPU y va plus vite que l'hôte).
À lire dans `STRATA_PREFILL_TIMING=1` : phases `host grouping` (désormais : le GPU qui attend l'hôte après l'expert
partagé ; le regroupement GPU est compté dans `router+shared`), `gather`, `wait copy`, `dequant`.

### Ce qui n'a pas été fait, et pourquoi

- **Supprimer complètement la synchronisation / couche capturable en graphe / groupes à plages d'ids fixes.** Les
  lancements MMQ de llama.cpp dépendent de `max_rows` (le plus grand nombre de lignes d'un expert du groupe) : il fixe
  la largeur de tuile `J` et la grille, donc le découpage *stream-k* de la réduction (activé pour tous les types sur
  Ampere et Blackwell dans cette version de llama.cpp). Des groupes à plages d'ids fixes, ou une borne fixe
  (`max_rows = T`) pour se passer des comptes côté hôte, changeraient ce découpage et donc les arrondis : ce serait
  une option (opt-in) à valider par KL, et il faudrait en plus des noyaux `swiglu`/quantification dont le nombre de
  lignes vient de la mémoire du GPU. Non fait : l'attente restante est courte et recouverte par l'expert partagé.
- La marche par groupe ne couvre pas les couches FP16 de repli (IQ1_M) ni les morceaux < 2 048 tokens : ils gardent
  la marche par expert (avec le regroupement GPU quand même).

## #46 (E3) : le chercheur de suffixe gardé d'une requête à l'autre

- `SuffixDrafter::reset()` est O(1) : chaque case porte l'époque où elle a été écrite, une case d'une autre époque se
  lit comme vide (la table n'est vraiment effacée qu'au rebouclage du compteur 32 bits). La case reste à 32 octets.
- `SuffixDrafter::sync(tokens)` : si l'historique est un préfixe des tokens de la requête (une conversation qui
  continue, réponse comprise), seuls les nouveaux tokens sont ajoutés ; sinon `reset()` + tout ajouter.
- L'index ne dépend que des tokens ajoutés depuis le dernier `reset`, donc les propositions sont celles d'un
  chercheur reconstruit. `suffix_drafter_test` le vérifie contre des chercheurs neufs (historique étendu, reconstruit,
  après 1 000 `reset`, et une petite table remplie à chaque époque — ce dernier cas échoue si les cases périmées
  restent occupées : vérifié en mutant le code).
- `STRATA_OLD_SFX_RESET=1` reconstruit l'historique à partir de tout le prompt à chaque requête.

**Mesuré ici (CPU du conteneur)**, prompt de 100 000 tokens dont 1 000 nouveaux : réécriture de la table + re-hachage
4,0 ms (contexte 128K) et 6,5 ms (262K) → `sync` 0,1 ms. **ESTIMÉ** sur la machine cible : 5 à 15 ms de moins avant le
premier token sur les longues conversations (audit).

## #45 (E2) : points de reprise de conversation

- Chaque point de reprise prend **un** tampon d'un pool (`CkptPool`, `generate.cpp`) : épinglé (`cudaHostAlloc`),
  rendu au pool quand le point de reprise est abandonné ; paginable si plus rien ne peut être épinglé.
- **Empreinte épinglée** : au plus `--prompt-cache` + 1 tampons de ~118 Mo (~826 Mo avec la valeur par défaut 6)
  tant que les points de reprise vivent. À chaque requête, après l'abandon des points de reprise qui ne servent plus
  (nouvelle conversation, changement de vecteur de contrôle, préfixe différent), le pool libère ses tampons inactifs
  au-delà d'**un** (`CkptPool::trim(1)`) : la mémoire épinglée suit les points de reprise vivants.
- Les morceaux (GDN, historique PLE, queues de l'indexeur) sont copiés par `cudaMemcpyAsync` sur le flux du pool,
  avec **une** attente ; la restauration fait le chemin inverse (après un `cudaDeviceSynchronize`, comme avant).
- Mêmes octets. `STRATA_OLD_CKPT=1` : vecteurs paginables réalloués et `cudaMemcpy` synchrones, comme avant.

La copie reste attendue par l'hôte (la synchronisation de l'appareil avant la copie aussi) : la faire tourner
pendant le morceau suivant demanderait que le flux du vérificateur attende l'événement de copie (`verify.cpp`, hors
de ce lot) ; le gain restant serait de 3 à 5 ms. **Gain (ESTIMÉ, audit) :** 10 à 30 ms par point de reprise (à chaque
changement de tour et tous les 16 384 tokens de prompt) : DMA à pleine vitesse, plus d'allocation ni de défauts de
page.

## #48 (E5) : le démarrage

**Constat : la prémisse de l'issue est inexacte.** `verify_slot` n'est appelé qu'**une** fois au démarrage (sur
l'emplacement 0, 1,38 Mo), pas sur chaque emplacement ; c'est vrai en 0.1.18 comme en 0.1.20 (et depuis le premier
commit du dépôt). Le vrai coût du démarrage est le remplissage lui-même : **une copie bloquante par emplacement**
(4 105 sur une carte de 12 Go, ~16 000 sur la 5090).

- Les remplissages sont maintenant mis en file sur un flux et attendus une fois (`ExpertCache::fill_slot`, déjà
  existant). Le moteur de copie n'attend plus l'hôte entre deux emplacements.
- Puisque les copies sont asynchrones, la relecture vérifie aussi le **dernier** emplacement rempli (en plus du
  premier) : +1,38 Mo lus.
- La durée du remplissage est affichée : `pre-filled N of M slots from the profile in X ms; the first and the last
  verified`.
- Le flux de remplissage est non bloquant, donc **pas** ordonné après le flux 0 : un `cudaDeviceSynchronize` le
  précède, sinon la mise à zéro de l'arène (`cudaMemset` de `ExpertCache::open`, jamais attendue avec
  `--expert-cache N` fixe) pourrait atterrir sur les copies.
- `STRATA_OLD_PROFILE_FILL=1` : les copies bloquantes.

**Gain (HYPOTHÈSE) :** 5 à 15 % du temps de remplissage (l'écart entre deux copies bloquantes), soit ~0,05 à 0,3 s sur
la 5090. La vérification par échantillonnage/hash demandée n'a pas de sens ici : il n'y a rien à échantillonner.

## #42 (P7) : re-remplissage asynchrone des emplacements prêtés (opt-in)

Avec `STRATA_ASYNC_REFILL=1` :

- **serveur** : `refill()` met les copies en file sur le flux de re-remplissage, enregistre `adapt_ev` et pousse les
  emplacements dans `pending` ; `apply_pending(false)` (déjà appelé avant chaque fenêtre de décodage, et maintenant
  aussi avant chaque fenêtre de lecture du prompt) les ré-admet quand les copies ont atterri. Un nouvel emprunt
  (`lend`) attend d'abord ces copies (`apply_pending(true)`), et `adapt()` ne tourne pas tant que `pending` n'est pas
  vide ;
- **ligne de commande** (pack natif, suivi de la boucle spéculative) : les emplacements prêtés sont re-remplis de la
  même façon au début de la boucle spéculative.

En attendant, le CPU calcule ces experts. **Pourquoi opt-in :** un expert calculé par le CPU arrondit différemment
de celui calculé par le GPU (`bench/results/2026-09-27-cache-parity`), et le moment où les copies atterrissent
dépend du minutage : les premières fenêtres ne sont plus reproductibles au bit près. **Gain (ESTIMÉ, audit) :** 0,2 à
0,5 s de délai avant le premier token par requête.

## Tests faits ici (CPU seulement)

- Construction complète `cmake --build build-wp` (CUDA 13.0, sm_120) : **OK**.
- `ple_reader_selftest`, `platform_memory_test`, `pool_stress`, `expert_multi_test`, `suffix_drafter_test`
  (étendu), `draft_policy_test`, `controller_test`, `conv_cache_test` : **OK**. `pool_test` échoue comme attendu
  (pas de pack de modèle ici).
- `python3 serve/test_server.py` : **OK** (26 tests).
- Nouveaux tests GPU, **compilés, pas exécutés** : `qsa_indexer_chunk_parity`, `moe_group_parity` (enregistrés dans
  ctest).

## À valider sur la RTX 5090

### Construction et tests

```bash
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON \
      -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j

ctest --test-dir build -R "qsa_indexer_chunk_parity|moe_group_parity|qsa_parity|suffix_drafter_test|conv_cache_test" \
      --output-on-failure
```

### Identité au bit près du prefill (#40, #36)

Même prompt long (32K), même commande, ancien et nouveau chemin ; les empreintes GDN, les résidus finaux et les tokens
doivent être **identiques**. Remplacer `<OPTS>` par la ligne habituelle (`--pack`, `--ple-gguf`, `--native`,
`--expert-profile`, `--expert-cache auto`, `--spec`, `--mtp`, `--prefill auto`, ...).

```bash
OPTS="<OPTS> --greedy --max-new 256 --adapt-every 100000"
STRATA_STATE_HASH_GDN=1 STRATA_PREFILL_DUMP_R=new.r ./build/strata $OPTS --tokens-file <PROMPT_32K.ids> > new.txt 2> new.err
STRATA_OLD_IDX_APPEND=1 STRATA_OLD_MOE_GROUP=1 STRATA_STATE_HASH_GDN=1 STRATA_PREFILL_DUMP_R=old.r \
  ./build/strata $OPTS --tokens-file <PROMPT_32K.ids> > old.txt 2> old.err
grep GDN_HASH new.err old.err          # les deux lignes identiques
cmp new.r old.r && echo "résidus identiques"
diff <(grep '^output' new.txt) <(grep '^output' old.txt) && echo "tokens identiques"
# chaque changement séparément, si une différence apparaît :
#   STRATA_OLD_IDX_APPEND=1 seul (#40), STRATA_OLD_MOE_GROUP=1 seul (#36), STRATA_PREFILL_COALESCE=0 (copies)

# la tenue de l'anneau de la marche par groupe (issue_runs / flush / événements) n'a pas de test unitaire : elle
# n'est couverte que par cette comparaison, À FAIRE AVANT LA FUSION.  Copies fusionnées ou non, et un anneau
# plus petit que le nombre d'experts diffusés d'une couche (~480 à 8 192 tokens), séries à cheval sur deux lots
# (une variable vide n'est pas « absente » : `env ${ring:+...}` ne la pose que si elle a une valeur) :
for ring in "" 16 40; do for co in 1 0; do
  env ${ring:+STRATA_PREFILL_RING=$ring} STRATA_PREFILL_COALESCE=$co STRATA_STATE_HASH_GDN=1 STRATA_PREFILL_DUMP_R=r_${ring}_$co.r \
    ./build/strata $OPTS --tokens-file <PROMPT_32K.ids> > t_${ring}_$co.txt 2> t_${ring}_$co.err
  env ${ring:+STRATA_PREFILL_RING=$ring} STRATA_OLD_MOE_GROUP=1 STRATA_STATE_HASH_GDN=1 STRATA_PREFILL_DUMP_R=o_${ring}_$co.r \
    ./build/strata $OPTS --tokens-file <PROMPT_32K.ids> > o_${ring}_$co.txt 2> o_${ring}_$co.err
  cmp r_${ring}_$co.r o_${ring}_$co.r && diff <(grep GDN_HASH t_${ring}_$co.err) <(grep GDN_HASH o_${ring}_$co.err) \
    && echo "ring=${ring:-défaut} coalesce=$co identiques"
done; done
```

### Benchmark A/B (3 prompts × 3 exécutions, glouton, 256 tokens)

```bash
OPTS="<OPTS> --greedy --max-new 256 --adapt-every 100000"
mkdir -p ab
for p in p1 p2 p3; do for r in 1 2 3; do
  STRATA_OLD_IDX_APPEND=1 STRATA_OLD_MOE_GROUP=1 STRATA_OLD_PROFILE_FILL=1 \
    ./build/strata $OPTS --tokens-file prompts/$p.ids > ab/old_${p}_$r.txt 2>&1
  ./build/strata $OPTS --tokens-file prompts/$p.ids > ab/new_${p}_$r.txt 2>&1
  STRATA_ASYNC_REFILL=1 ./build/strata $OPTS --tokens-file prompts/$p.ids > ab/async_${p}_$r.txt 2>&1
done; done
# débit de prefill et délai avant le premier token ; durée du remplissage du profil ; re-remplissage
grep -h "generate: prefill\|pre-filled\|lent slots" ab/*.txt
# les tokens : identiques entre old et new ; async peut différer légèrement (#42)
for p in p1 p2 p3; do diff <(grep '^output' ab/old_${p}_1.txt) <(grep '^output' ab/new_${p}_1.txt) && echo "$p ok"; done
```

Répartition du prefill (à faire une fois par chemin) :

```bash
STRATA_PREFILL_TIMING=1 ./build/strata $OPTS --tokens-file <PROMPT_32K.ids> 2>&1 | grep "prefill timing"
STRATA_OLD_IDX_APPEND=1 STRATA_OLD_MOE_GROUP=1 STRATA_PREFILL_TIMING=1 ./build/strata $OPTS \
  --tokens-file <PROMPT_32K.ids> 2>&1 | grep "prefill timing"
```

À comparer : `qsa indexer`, `host grouping`, `gather`, `wait copy`, et le total.

### Serveur (#45, #46, #42)

Lancer le serveur comme d'habitude (`python3 serve/server.py --engine strata --config <config.json>`), une fois avec
`STRATA_OLD_CKPT=1 STRATA_OLD_SFX_RESET=1` dans l'environnement du moteur, une fois sans, une fois avec
`STRATA_ASYNC_REFILL=1` ; envoyer la même conversation de 3 tours (premier message long, > 16K tokens). Comparer par
requête le `prompt_ms` de la ligne `DONE` (ou le délai avant le premier token vu par le client) et, avec
`STRATA_TRACE=1`, les lignes `refill start` / `prompt done`. Les réponses doivent être identiques entre les deux
premiers lancements. `STRATA_CKPT_REREAD=1` reste le contrôle des points de reprise (relecture au lieu de
restauration, même réponse attendue avec `--adapt-swaps 0`).
