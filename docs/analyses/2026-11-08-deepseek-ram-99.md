# DeepSeek-V4.1 sur la .99 : arrêts « faute de RAM » du 07 et du 08/11/2026

Analyse en lecture seule, faite le 08/11/2026 vers 10 h. Rien n'a été modifié sur la machine.

## En bref

- Windows n'a jamais manqué de mémoire. Aucun événement `Resource-Exhaustion-Detector` depuis le 18/08/2026, et aucun plantage applicatif de `ds41_generate.exe` dans le journal Application.
- Quatre arrêts sur cinq lancements, tous faits par le garde mémoire `E:\llm\services\deepseek\guard.ps1`. Il tue tout l'arbre du serveur dès que la RAM disponible passe sous 8 Gio.
- Trois de ces arrêts (07/11 à 18 h 14 et 21 h 15, 08/11 à 06 h 35) tombent à 7,86 ou 7,96 Gio disponibles. La machine tenait encore, rien ne swappait. C'est le garde qui a coupé.
- Le quatrième (08/11 à 09 h 36) est un vrai plantage du moteur, `GGML_ASSERT ... tensor write out of bounds`, juste après une requête annulée. La RAM disponible est tombée à 0,86 Gio et le fichier d'échange s'est rempli.
- Cause de fond : la configuration actuelle laisse trop peu de marge. Au démarrage, il reste 24 à 26 Gio disponibles. Une fois le cache de conversations rempli, il en reste 11 à 15. Un creux normal de 4 à 6 Gio pendant le décodage suffit alors à passer sous les 8 Gio du garde.

## Ce qui tourne (relevé du 08/11/2026)

- `ds41_generate.exe` (`E:\llm\xolotl\build-684d`, branche locale `strata-127-684`, base amont `684e3c06`), lancé à 09 h 52 :
  - mémoire privée 102,2 Gio ;
  - working set 81,6 Gio, dont 70,5 Gio privés.
- Système :
  - RAM disponible 21,3 Gio, presque entièrement en cache standby ;
  - mémoire engagée 120,7 Gio sur une limite de 149,2 Gio ;
  - pool non paginé 2,7 Gio.
- Fichier d'échange : `C:\pagefile.sys`, géré par Windows, 24 181 Mo alloués, utilisation courante 1,7 Go. **Pic à 23 935 Mo** depuis le démarrage du 28/10 : il s'est rempli une fois.
- VM Hyper-V : `gitlab-runner-140`, `-141` et `-142` arrêtées. Leur `AutomaticStartAction` vaut `Start` : elles repartent au prochain redémarrage de Windows et prendraient alors 3 × 16 Gio.
- WSL : aucun `vmmem` actif, `.wslconfig` borné à 16 Gio.

## Le budget mémoire du moteur

Tout vient de `E:\llm\services\deepseek\strata-ds41-deepseek.json` et de la source `E:\llm\xolotl\src-684`.

| Poste | Taille | Nature |
|---|---|---|
| Arène d'experts (`--arena-gib 64`) | 64 Gio | verrouillée (`cudaHostRegister`), allouée au chargement |
| Cache de conversations (`--conversation-cache-mib 8192`) | 0 au départ, jusqu'à 8 Gio | verrouillée (`cudaHostAlloc`), **grossit par extents au fil de la session** (`ds41_conv_pool.cpp`, `grow()`) et ne rend rien |
| Tampons d'étape (`h_fstage`, `h_def`, `c_hparts`, `c_bounce`) | variable | verrouillés, alloués au premier besoin puis gardés (« stay », `ds41_moe.cpp:6556`) |
| Reste de la mémoire privée | environ 30 Gio | engagée, en partie non résidente |

Le commentaire du json note « 24 GiB free after the run » au passage du 28/10, juste après l'ajout du cache de conversations. Cette mesure a été faite sur un passage court. Le cache était donc presque vide.

Depuis, les sessions d'agent sont longues, avec des contextes de 100 000 à 193 000 tokens et jusqu'à 40 conversations parquées. Le journal du moteur donne 0,94 à 1,2 Go de données utiles parquées. La mémoire verrouillée réservée par le cache, elle, n'est pas journalisée. Elle avance par extents et ne redescend pas.

Tendance relevée dans `guard-deepseek.log`, un point toutes les 10 minutes :

- au démarrage, 94 Gio disponibles, puis 24 à 26 Gio une fois le modèle chargé ;
- pendant une session, 11 à 15 Gio en régime établi ;
- avant chaque arrêt, une descente de 13 à 8 Gio en 2 à 3 minutes, pendant un décodage qui suit la lecture d'un long prompt.

## Les quatre arrêts

| Heure | RAM dispo au kill | Dernière action du moteur |
|---|---|---|
| 07/11 18 h 14 | 7,96 Gio | décodage après un prompt de 42 497 tokens |
| 07/11 21 h 15 | 7,86 Gio | décodage après un prompt de 193 484 tokens (192 984 repris du cache) |
| 08/11 06 h 35 | 7,95 Gio | décodage après un prompt de 14 634 tokens |
| 08/11 09 h 36 | 0,86 Gio | voir ci-dessous |

### Le plantage de 09 h 36

Journal du moteur (`strata-ds41.log`), dans l'ordre :

1. Une requête est annulée en pleine lecture de prompt : `done: 0 tokens in 1 s (cancel, cancel=True)`. Juste avant, la ligne `prompt phases: 9234 new = 9234 batch (0.52 s ...) + DSpark tail 0` montre une lecture de 9 234 tokens coupée en 0,52 s.
2. La requête suivante restaure du pool une conversation de 142 536 tokens et ajoute 1 293 tokens neufs.
3. La lecture du prompt reste bloquée plus de 31 s. Pendant ce temps, la RAM disponible tombe de 15,8 à 0,86 Gio en moins de 46 s, et le garde n'arrive même plus à écrire ses relevés toutes les 2,5 s.
4. Le moteur meurt sur `ggml-backend.cpp:485: GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds") failed`.

C'est un défaut du moteur, et non un manque de RAM de la machine. Un état laissé incohérent par l'annulation semble mal dimensionner une écriture. Le pic de mémoire en découle ou l'accompagne : l'ordre exact n'est pas prouvé. Scénario à rejouer pour le confirmer (non reproduit) : annuler une requête pendant la lecture d'un long prompt, puis relancer une conversation parquée.

## Corrections proposées

Par ordre de priorité.

**Déjà fait le 08/11/2026 vers 11 h, sur ordre de Guillaume.** Le garde mémoire a été supprimé. Il n'aurait jamais dû exister : il empêche le système de paginer et cache les vrais défauts. Le garde qui tournait a été arrêté, sans toucher à DeepSeek, dont `/health` répond toujours `loaded`. `guard.ps1` a été retiré de la .99 et de `~/.claude/bin/strata.d`. `start-deepseek.ps1` n'en lance plus, et la version déployée est identique à la source. 

Le point 1 est appliqué le même jour : `--conversation-cache-mib 3072` est écrit dans la source et déployé sur la .99, avec la même empreinte des deux côtés. Comme Guillaume l'a demandé, DeepSeek n'a pas été redémarré : le réglage prendra effet au prochain démarrage. Le point 2 est appliqué sur le go de Guillaume : `C:\pagefile.sys` passe à 64 Gio fixes et la gestion automatique est coupée. Le réglage, relu par CIM, prend effet au prochain redémarrage de Windows, qui n'a pas été fait. Point 3 : Guillaume a répondu « Laisser en auto » le 08/11/2026, et les VM gardent leur démarrage automatique. Le point 4 est suivi dans gputier/StrataGP#136.

### 1. Plafonner le cache de conversations à 3 Gio

Dans `strata-ds41-deepseek.json`, passer `--conversation-cache-mib` de `8192` à `3072`.

- Gain : jusqu'à 5 Gio de mémoire verrouillée en moins en régime établi. Une mémoire verrouillée ne se pagine jamais : c'est autant de place rendue au reste du système.
- Coût attendu : nul sur l'usage observé, où 1,2 Go au plus étaient occupés. Au-delà de 3 Gio, le pool évince la conversation la plus ancienne, qui sera relue en entier si elle revient.
- Preuve à faire : relancer, rejouer une session d'agent longue et vérifier que la RAM disponible reste au-dessus de 15 Gio.

### 2. Fixer le fichier d'échange à 64 Gio

Sans garde, c'est le fichier d'échange qui absorbe les pics. Aujourd'hui, `C:\pagefile.sys` est géré par Windows et fait 24 Gio, et il a déjà été plein une fois. Le fixer à 64 Gio sur C:, qui a 608 Go libres, porte la limite d'engagement d'environ 149 à 189 Gio.

- Gain : un pic passager pagine au lieu de faire échouer une allocation. Le moteur juge lui-même sur `min(ullAvailPhys, ullAvailPageFile)` (`conversation_memory.cpp:286`), et c'est l'engagement qui a saturé à 09 h 36.
- Coût : 64 Go de disque. Le réglage demande un redémarrage de Windows.
- Limite : la RAM verrouillée (arène et pool) ne part jamais dans le fichier d'échange. Le point 1 reste donc nécessaire.

### 3. Passer les VM runners en démarrage manuel

`Set-VM gitlab-runner-14x -AutomaticStartAction Nothing`, si elles ne doivent pas repartir seules. Sinon, un redémarrage de Windows rend 48 Gio aux VM et DeepSeek ne charge plus son arène. Ce choix touche le runner CI 25 : à trancher avant de le faire.

### 4. Le plantage du moteur

Ce défaut se corrige dans Xolotl, pas par un réglage. Voir la section Xolotl : soit l'amont l'a déjà corrigé, soit il faut ouvrir une issue avec ce scénario.

En attendant, éviter d'annuler une requête pendant la lecture d'un long prompt.

## Xolotl : correctifs et optimisations publiés depuis `684e3c06`

Amont : `github.com/Indras-Mirror/Xolotl`. Le relevé a été fait le 08/11/2026. Il y a 2 898 commits depuis `684e3c06`. 2 586 viennent de la fusion Strata `c7a1c73f` et 45 seulement touchent `tools/v41`. Le dernier commit est `1a6242db`. Il n'y a ni tag ni release.

Le détail, avec les citations des commits, est dans `xolotl-upstream-report.md`. Ce fichier est dans le scratchpad de la session.

### Sur nos deux problèmes : rien en amont

- **RAM** : `ds41_conv_pool.hpp`, le pool de conversations sur une seule carte, n'a reçu aucun commit depuis la base. Le plafond reste à poser chez nous, par réglage (point 1 ci-dessus).
- **Assert de 09 h 36** : les appels à `ggml_backend_tensor_set` sont identiques entre la base et `main` dans `ds41_dense.cpp`, `ds41_dspark.cpp`, `ds41_conv_reuse.hpp` et `ds41_batch_slots.hpp`. Le défaut est donc à diagnostiquer chez nous, puis à signaler en amont.

### À prendre

- `5b06a59e` (31/10) : au-delà de 65 535 clés de contexte, le noyau CUDA `pad` F32 laissait des lignes non initialisées. La .99 y est sans doute peu exposée, car la voie `STRATA_ATTN_F16_GATHER`, active par défaut, ne l'utilise pas. Le correctif est petit, et nos contextes dépassent 140 000 tokens.
- `a8f0e8c2` (08/11) : une reprise après une réponse interrompue relisait jusqu'à 10 593 tokens, soit 25 s. C'est voisin de notre scénario d'annulation, mais l'amont précise « NOT yet run on the model ».
- `2d1df99d` (#100) : obligatoire si l'on monte de version. La fusion Strata renvoyait une 503 au lieu de mettre la requête en attente.

### Gains de vitesse à mesurer (aucun n'a été mesuré sur une 5090 sous Windows)

- `1510ca5a` : read-ahead de l'engram et expert partagé tardif, actifs par défaut. Le commit annonce +11 % en prefill multi-chunk et +2,7 % en décodage.
- `ce19d56e` (`STRATA_CHUNK_CPU=1`, opt-in) : environ 0,5 s de moins par petit tour d'agent. Il prend un peu plus de RAM verrouillée, ce qui va à contre-courant du point 1.
- `3b4d95b8` et `998575e9` : cache de requantification dense, actif par défaut et compilable sous Windows. Il évite environ 25 s de conversion à chaque chargement et écrit environ 5,5 Go sur disque. Son dossier se déduit du dernier `/` du chemin : avec un chemin `E:\...`, non vérifié.
- `XOLOTL_SHADOW` (Q2_K, avec perte) : +27 % en décodage, mais le top-1 passe de 87,01 à 86,16 %. À écarter, puisque la config actuelle est réglée sans perte.

### Coût d'une montée de version

- Le correctif #127 (tokens émis après le STOP) n'a pas d'équivalent en amont. Il faudra le reporter sur un `serve/server.py` passé de 7 327 à 11 179 lignes, où la lecture DSML a été déplacée dans `serve/dsml.py`.
- `STRATA_BLOB_MIRROR2` risque d'entrer en conflit avec les sidecars `STRATA_SHADOW`, qui réécrivent `Ds41FileBlobs::open` et `read` (non vérifié contre notre code).
- Aucun de nos drapeaux ni aucune de nos variables `STRATA_*` n'est renommé ou retiré. Les alias `XOLOTL_*` et le chargement rapide (pin-after, pipe 4) ne marchent que sous Linux.

Conclusion : monter de version ne règle pas les arrêts. Cela peut attendre un chantier à part, avec rebase de #127 et de MIRROR2, puis une série de mesures sur la .99.
