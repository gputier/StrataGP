# correctness : corrections latentes et tests de parité manquants (#2, #3, #5, #8, #9, #10, #13)

Branche `perf/correctness`, partie de `perf/base` (b858f9b, moteur 0.1.20). Aucune mesure GPU n'a été faite : la
machine de développement n'a pas de GPU. Tout ce qui touche au GPU a été **compilé** (sm_120, CUDA 13.0, build vert)
mais **pas exécuté**. Les tolérances des nouveaux tests GPU viennent d'un modèle d'erreur écrit dans le code (et,
quand c'était possible, vérifié par une simulation hôte), pas d'une mesure. Les gains sont **ESTIMÉ** (calcul) ou
**HYPOTHÈSE** (à confirmer).

## Résumé

| Issue | Audit | Changement | Par défaut | Sélecteur |
|---|---|---|---|---|
| #2 | B2 | `s_gemv_q8_split_kernel` : la barrière avant le `return` des warps sans ligne | **oui** (identique au bit) | aucun (même kernel) |
| #13 | B15 | `bf16_from_f32` et `f2bf` (dequant_bf16) : un NaN reste un NaN, comme ggml | **oui** (identique hors NaN) | aucun |
| #10 | B11 | `--spec` sans `--mtp` ni `--spec-oracle` : fenêtres de 1 token, plus de brouillons 0 | **non** (opt-in tant que l'A/B GPU n'a pas confirmé la sortie identique) | `STRATA_SPEC_T1=1` |
| #3 | B4 | `rope_parity` : RoPE natif de 0 à 262 144 contre FP64 ; angle FP64 en option | test ; option **non** | `STRATA_ROPE_F64=1` |
| #5 | B6 | softplus natif = llama.cpp, vérifié et documenté ; pas de changement de calcul | — | — |
| #8 | B9 | 5 tests étendus, 4 nouveaux tests GPU, 1 test CPU | — | — |
| #9 | B10 | `tools/prefill_chunk_check.py` : même prompt, plusieurs tailles de morceau | — | — |

## Ce qui change, par issue

### #2 (B2) : barrière sautée dans `s_gemv_q8_split_kernel` (`src/kernels/cuda/s_gemv.cu`)

Le `if (o >= n_out) return;` précédait le chargement du codebook en mémoire partagée et le `__syncthreads()`. Dès
que `n_out % 8 != 0`, les warps sans ligne du dernier bloc quittaient avant une barrière que les autres attendaient
(comportement indéfini, latent car toutes les formes du pack sont des multiples de 8). Le chargement et la barrière
viennent maintenant d'abord, comme dans `s_gemv_q8k_kernel` ; le `return` reste uniforme par warp (`o` est par warp),
donc le masque plein des `__shfl_down_sync` reste valide.

- Arithmétique inchangée, **identique au bit près par construction** ; `ptxas` : 48 registres, 0 spill, comme avant
  (les 4 instanciations).
- Au passage : le lanceur Q8_0 vérifiait `group_elems % 4` alors qu'une itération prend 16 éléments sous une seule
  échelle ; il refuse maintenant tout sauf un multiple de 16, comme le lanceur Q8_K (toutes les tailles du format
  sont 16, 32 ou 64 : aucun pack réel n'est refusé).
- Pas de sélecteur : c'est le même kernel, seule la place du `return` change.

### #13 (B15) : `bf16_from_f32` transformait NaN en −0 (`bf16_bits.hpp`, `dequant_bf16.cu`)

L'addition d'arrondi débordait pour un NaN : `0x7FFFFFFF` → `0x8000` (−0), `0x7F800001` → `0x7F80` (+inf). Les deux
fonctions testent maintenant `(u & 0x7fffffff) > 0x7f800000` et renvoient `(u >> 16) | 64`, exactement
`ggml_compute_fp32_to_bf16`. Pour toute entrée non-NaN (infinis compris), le résultat est **identique au bit près** :
c'est **prouvé exhaustivement sur les 2^32 motifs** par `bf16_bits_test` (exécuté ici : OK ; l'ancienne règle
masquait 131 072 NaN sur 16 777 214). Coût : une comparaison, dans des kernels limités par la mémoire.

Non modifiés (fichiers d'autres paquets, même défaut) : `bf16_bits` de `src/kernels/cuda/ple.cu`, `bf` de
`src/prefill/kernels.cu`, et la copie hôte de `gr_parity.cpp`. À aligner lors de la fusion.

### #10 (B11) : `--spec` sans `--mtp` remplissait les fenêtres de token 0 (`src/program/generate.cpp`)

Sans MTP ni oracle, chaque fenêtre contenait T−1 brouillons `0`, rejetés aussitôt, chacun un token entier de travail
de vérification. Maintenant la fenêtre sans brouillon est le dernier token seul (T = 1) ; la recherche de suffixe
(`--suffix-draft`) peut toujours l'élargir (la `DraftPolicy` compare alors T = 1 et la fenêtre de recherche). Un
brouillon rejeté n'émet jamais rien : la sortie ne reste identique que si le résultat d'une ligne de vérification ne
dépend pas de T au bit près (T = 1 et T = 4 passent par des kernels multi-token et un batching CPU différents). Rien
ne l'assure dans le code et rien ne l'a vérifié sur GPU : le changement est donc **opt-in**, à passer par défaut
seulement après l'A/B ci-dessous (tokens identiques). Une ligne sur stderr indique le mode.

- Nouveau comportement : **`STRATA_SPEC_T1=1`** (par défaut : fenêtres complétées de token 0, comme avant).
- Le mode serveur n'est pas concerné (`--serve` exige `--mtp`), ni la configuration de `setup.py` (elle passe `--mtp`).
- Gain **ESTIMÉ** pour cette configuration (pack natif sans `--mtp`, `--spec 4`) : une fenêtre de 4 coûte ~2,05× une
  fenêtre de 1 (`kShape` de `draft_policy.cpp`, mesuré sur RTX 5070 avec les experts manqués sur CPU) pour le même
  token émis, soit jusqu'à **~2× de débit** ; **HYPOTHÈSE** sur la 5090 (moins d'experts manqués : gain plus faible).

### #3 (B4) : RoPE natif aux grandes positions (`src/kernels/rope_parity.cpp`, `native_rope.cu`)

`rope_parity` exécute maintenant `native_rope` (le chemin `--native`, l'angle f32 de llama.cpp sous
`--use_fast_math`) aux positions 0, 1, 31, 1 024 … 131 072, 200 000, 262 143, 262 144. Une ligne-sonde par position
(x = 1 sur la première moitié, 0 sur la seconde) renvoie (cos θ', sin θ') : l'erreur de phase des 32 paires est lue
directement (atan2) et imprimée par position, avec l'erreur des valeurs tournées (lignes aléatoires) et le bit à bit
de l'appel en place (x == out, comme tous les appelants).

- **Tolérance documentée** (modèle, pas mesure) : la pire paire perd au plus ~4,3e-7 × pos radians (theta_scale
  arrondi en f32 puis élevé à la puissance p, `lg2`/`ex2` approchés de `__powf`, produit f32, fraction de tour f32
  arrondie vers zéro avant le sinus matériel). Le test admet **1e-4 + 1e-6 × pos rad** (2,3× le pire cas cumulé,
  ~0,26 rad à 262K) et reste ~4× sous l'erreur structurelle la plus petite (position décalée de 1 : 1 rad sur la
  paire 0). À 131 072 le modèle borne à ~0,056 rad ; l'audit estimait ~1e-2. Le vrai chiffre s'imprimera sur la 5090.
- **Option `STRATA_ROPE_F64=1`** (ou `native_rope_set_f64_angle(true)`), **désactivée par défaut** : une seconde
  instanciation du kernel calcule pos × base^(−2i/64) et son sinus/cosinus en double (l'arithmétique du chemin par
  table ; fast-math ne touche pas au double). L'instanciation par défaut est compilée comme avant (18 registres, pas
  de pile) et reproduit toujours llama.cpp. `rope_parity` vérifie aussi ce mode : exact à l'arrondi f32 de cos/sin
  près (1e-6 rad) à toutes les positions. C'est l'outil de la mesure demandée par l'issue (KL forcé et aiguilles à
  128K), voir plus bas.
- **Constat (non corrigé, fichier d'un autre paquet)** : le RoPE du chemin de prompt (`rope_kernel` de
  `src/prefill/kernels.cu`) n'est **pas** compilé en fast-math (`powf`/`cosf`/`sinf` précis, angle f32 : ~1,2e-7 × θ,
  ~0,01 rad à 128K), alors que le décodage `--native` utilise l'angle fast-math. Les K du prompt et les Q du décodage
  ne sont donc pas tournés avec la même arithmétique ; l'écart croît avec la position. `STRATA_ROPE_F64` ne touche
  que le décodage. Piste : aligner les deux chemins (même instanciation) après mesure.

### #5 (B6) : softplus (`src/kernels/cuda/native_gdn_preprocess.cu`, commentaire seulement)

Vérifié contre llama.cpp au commit épinglé (`/opt/llama.cpp`, 3cf03257f) : `ggml-cuda/unary.cu` `op_softplus` est
`(x > 20.0f) ? x : logf(1.0f + expf(x))`, compilé en `-use_fast_math`. Le chemin natif (`native_gdn_gate`) est
**exactement cette expression** avec le même seuil et les mêmes options : il reproduit llama.cpp, il reste tel quel
(un commentaire le dit). Les trois formes du moteur :

| chemin | expression | quand |
|---|---|---|
| natif (`native_gdn_gate`) | `logf(1 + expf(v))`, fast-math = llama.cpp | `--native --no-fused-gdn` |
| fusionné (`fused_gdn_ab`) | `log1pf(__expf(v))` | **`--native` par défaut** |
| historique (`gdn_gate`) | `log1pf(expf(v))` | sans `--native` |

Conséquence : le chemin de production par défaut (fusionné) **ne reproduit pas** llama.cpp sous v ≈ −5 ; il est plus
précis (le natif vaut exactement 0 sous v ≈ −17, décroissance = 1). L'écart absolu sur la porte est de l'ordre de
|ssm_a| × 4e-7 : faible, mais systématique. Aucun calcul n'a été changé (le changer altérerait la numérique ; une
option pour aligner le fusionné sur llama.cpp serait possible si la mesure le justifie). `gdn_fused_parity` affirme le
natif contre l'expression llama.cpp (hôte), le fusionné contre FP64, et imprime l'écart par plage de v.

### #8 (B9) : tests de parité manquants

Tous enregistrés dans ctest, tous compilés ; aucun n'a pu tourner ici (sauf le test CPU).

| test | ce qu'il affirme |
|---|---|
| `gdn_fused_parity` (nouveau) | `gdn_step`, `native_gdn_step` et `fused_gdn_step_norm` sur **256 pas** contre un état FP64, deux régimes de portes (mixte ; mémoire longue, décroissance 0,98–0,9995), dérive imprimée aux pas 1/16/64/256, tolérance 5e-6 + 2e-7 × pas (une simulation hôte de l'ordre fusionné donne ~2e-7 de dérive, 1,3e-5 si chaque décroissance est biaisée de 2 ulp dans le même sens) ; les trois softplus (#5) ; `fused_gdn_ab` contre FP64 sur tous les régimes de v ; `fused_gdn_conv_l2` ; `native_gdn_conv_silu`, `native_gdn_l2_norm`, `native_gdn_beta_gate`, `native_gdn_out_norm` |
| `fused_gr_parity` (nouveau) | `fused_gr_read` contre une transcription FP64 du contrat (rs, lo, inject, mixed, R mis à jour), avec/sans écriture repliée et lignes d'injection ; `fused_gr_read_multi` **bit à bit** égal à `fused_gr_read` de chaque token pour T = 1..8 (promesse de l'en-tête dont dépend l'exactitude des fenêtres) |
| `qsa_decode_parity` (nouveau) | `qsa_block_scores` contre FP64 ; `qsa_block_topk` **exact** (ids) sur les scores du GPU et sur des scores pleins d'égalités (+0/−0), identité sous la largeur (la référence a été vérifiée ici contre un portage hôte du radix select) ; `qsa_decode_attn` FP16, INT8 et Q4_0 via une table de pages permutée, 1/65/2051 cellules, contre une attention FP64, et la forme lot bit à bit égale à la forme par requête |
| `bf16_bits_test` (nouveau, CPU) | #13 sur les 2^32 motifs (exécuté ici : OK) |
| `router_top10_parity` (étendu) | le routeur natif (`native_router_top10`) sur les 4 distributions : ids exacts (égalités → plus petit indice), poids à 1e-5 |
| `bf16_gemv_parity` (étendu) | le MMVF natif (`bf16_gemv_fp32_mmvf`) **bit à bit** contre une émulation hôte de son ordre (paires, deux fmaf ordonnés, papillon xor par warp puis entre warps), tailles de bloc 32/128/160/256, et à 1e-5 du FP64 |
| `quantize_act_parity` (étendu) | `quantize_q8_0_scaled` **bit à bit** contre le vrai `act_quant_q8_1` du CPU (codes int8 et échelles fp32 ; forme AVX2 sans AVX-512), dont un cas d'égalités exactes k + 0,5 (lie désormais `strata_kernels_cpu`) |
| `kv_q4_parity` (étendu) | la reconstruction est **affirmée** : bit à bit la déquantification hôte du bloc hôte, et sous la borne Q4_0 (1,01 × |d16| + |x|/1024) |
| `s_gemv_q8k_parity` (étendu) | #2 : `n_out` = 1, 7, 9, 61 pour les deux kernels split, contre la référence hôte, erreur L1 normalisée par Σ\|w·x\| de chaque ligne (une ligne dont la somme s'annule ne fait pas échouer un kernel correct), avec une bande de garde après `n_out` qui doit rester intacte |
| `rope_parity` (étendu) | #3, ci-dessus |
| `elementwise_parity` (étendu) | #13 sur GPU : `f32_to_bf16_bulk` contre la règle ggml (NaN, inf, motifs aléatoires), et un bloc Q8_0 à échelle NaN via `dequant_bf16` |

Non couverts ici : `pool_test` `run_split`/`run_split_multi` bit à bit (paquet CPU), `iq_parity` au-delà de
`ncols = 2`, les tests oracle `bench/micro/*` (non publiés).

Observation (non modifiée, conforme à llama.cpp) : `native_router.cu` fait `if (threadIdx.y != 0) return;` avant un
`__syncthreads()` qui ne protège rien (pas de mémoire partagée) ; même classe que #2, sans effet en pratique (des warps
entiers sortent), et `topk-moe.cu` de llama.cpp a la même forme.

### #9 (B10) : étalon du prefill (`tools/prefill_chunk_check.py`)

L'outil lit le même prompt avec chaque taille de morceau et compare, à la première taille (ou au chemin token par
token avec `--token-path`) :

- le hash d'état GDN par couche récurrente après le chemin de prompt (`STRATA_STATE_HASH_GDN`, identique ou non) ;
- les logits des `--tail` dernières positions du prompt, passées par le chemin token (`--prefill-until`, forçage par
  l'enseignant) : top-1 identique, KL moyen et dernier, écart max ;
- les tokens gloutons générés après le prompt.

La configuration de référence tourne deux fois : si la relance n'est pas identique au bit près, l'outil le signale
(le cache adaptatif est figé par `--adapt-every 100000`). Prompt synthétique déterministe de **9 000 tokens** par
défaut (`--length`, `--seed`) ou `--ids FICHIER` : un morceau ne coupe le prompt que si le prompt est plus long que
lui (9 000 → 2 morceaux à 8192, 3 à 4096, 9 à 1024 ; un prompt de 2K serait **un seul** morceau à 8192 comme à
4096). Un pack natif (IQ) ne sort pas de logits de ses fenêtres de vérification : hash et tokens seulement, `--tail 1`.
`--ab-env CLE=VALEUR` relance la référence avec une variable d'environnement (sert à l'A/B RoPE de #3).
`tools/test_prefill_chunk_check.py` le teste avec un moteur factice (exécuté ici : OK).

## Tests exécutés ici (CPU seulement)

- `bf16_bits_test` : **OK** (2^32 motifs, 1,5 s).
- `platform_memory_test`, `ple_reader_selftest`, `suffix_drafter_test`, `controller_test`, `draft_policy_test`,
  `conv_cache_test`, `expert_multi_test`, `pool_stress` : **OK**. `pool_test` : échoue comme prévu (pas de pack :
  « cannot read expert 0 of layer 0 from pack/full/experts.bin »).
- `python3 -m unittest tools.test_prefill_chunk_check tools.test_calibrate` : **OK** (17 tests) ;
  `python3 serve/test_server.py` : **OK** (26 tests).
- Build complet `cmake --build build-wp` : **vert**. `ptxas -v` : `s_gemv_q8_split_kernel` 48 registres, 0 spill
  (inchangé) ; `native_rope` défaut 18 registres, pas de pile (inchangé), instanciation FP64 32 registres, 40 o de pile.
- Hors dépôt : simulation hôte de la dérive GDN (ordre du kernel fusionné) pour dimensionner la tolérance ; portage
  hôte de `block_topk_kernel` comparé à la référence du test (identiques sur les deux jeux de scores).
- Tous les tests GPU (`*_parity`) : **compilés, pas exécutés**.

## À exécuter sur la RTX 5090

### Construction et tests

```bash
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON \
      -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j

# les tests nouveaux et étendus de ce paquet
ctest --test-dir build --output-on-failure -R \
  "gdn_fused_parity|fused_gr_parity|qsa_decode_parity|bf16_bits_test|router_top10_parity|bf16_gemv_parity|quantize_act_parity|kv_q4_parity|s_gemv_q8k_parity|rope_parity|elementwise_parity|gdn_parity"

# les chiffres à relever (erreur de phase du RoPE par position, dérive GDN, écart des softplus)
./build/rope_parity --selftest
./build/gdn_fused_parity --selftest

# #2 : la barrière elle-même (doit dire 0 erreur, y compris sur les n_out 1/7/9/61)
compute-sanitizer --tool synccheck ./build/s_gemv_q8k_parity --selftest
# facultatif : courses en mémoire partagée des kernels fusionnés testés ici
compute-sanitizer --tool racecheck ./build/fused_gr_parity --selftest
compute-sanitizer --tool racecheck ./build/gdn_fused_parity --selftest
```

Si un test échoue sur une **tolérance** (et non sur un résultat structurellement faux), noter la valeur imprimée :
les tolérances viennent d'un modèle d'erreur, pas d'une mesure sur ce GPU.

### A/B de référence (non-régression)

3 prompts × 3 exécutions, glouton, 256 tokens, cache adaptatif figé. `<OPTS>` = la ligne habituelle (`--pack`,
`--ple-gguf`, `--native`, `--expert-profile`, `--expert-cache`, `--prefill`, `--spec`, `--mtp`, ...).

```bash
OPTS="<OPTS> --greedy --max-new 256 --adapt-every 100000"
mkdir -p ab
for p in p1 p2 p3; do for r in 1 2 3; do
  STRATA_STATE_HASH=1 ./build/strata $OPTS --tokens-file prompts/$p.txt > ab/new_${p}_$r.txt 2>&1
done; done
# même boucle avec le binaire de perf/base -> ab/base_${p}_$r.txt
grep -H "^output" ab/*.txt        # tokens IDENTIQUES entre base et new (#2 et #13 sont identiques au bit près)
grep -H "tok/s\|tokens per round" ab/*.txt
```

### A/B #10 (`--spec` sans `--mtp`)

```bash
OPTS_NOMTP="<OPTS sans --mtp> --greedy --max-new 256 --adapt-every 100000"
for p in p1 p2 p3; do for r in 1 2 3; do
  STRATA_SPEC_T1=1 ./build/strata $OPTS_NOMTP --tokens-file prompts/$p.txt > ab/nomtp_new_${p}_$r.txt 2>&1
  ./build/strata $OPTS_NOMTP --tokens-file prompts/$p.txt > ab/nomtp_old_${p}_$r.txt 2>&1
done; done
grep -H "^output" ab/nomtp_*.txt   # tokens identiques entre old et new : condition pour le passer par défaut
grep -H "decode\|speculation" ab/nomtp_*.txt   # new : "rounds of", tok/s nettement plus haut
```

### #3 : effet de l'angle FP64 à 128K

```bash
# KL forcé par l'enseignant sur les 256 dernières positions d'un prompt de 128K (pack canonique ; --max-context >= 131328)
python3 tools/prefill_chunk_check.py --config strata-q2_0.json --chunks 8192 --length 131072 --tail 256 \
    --ab-env STRATA_ROPE_F64=1 --keep ab/rope128k
# aiguilles : serveur lancé une fois normalement, une fois avec STRATA_ROPE_F64=1 dans l'environnement
# (le serveur transmet son environnement au moteur ; contexte >= 128K dans la config)
python3 -m serve.server --engine strata --config strata-q2_0.json --port 8080 &
python3 tools/needle_bench.py --lengths 128k,262k --depths 10,50,90 --out ab/needles_fast.json
# arrêter le serveur, puis :
STRATA_ROPE_F64=1 python3 -m serve.server --engine strata --config strata-q2_0.json --port 8080 &
python3 tools/needle_bench.py --lengths 128k,262k --depths 10,50,90 --out ab/needles_f64.json
```

### #9 : tailles de morceau

```bash
# hash GDN + dernières positions + tokens, 8192 / 4096 / 1024, contre le chemin token par token
python3 tools/prefill_chunk_check.py --config strata-q2_0.json --length 9000 --tail 256 --token-path --keep ab/chunks
# pack natif (IQ) : pas de logits, hash et tokens
python3 tools/prefill_chunk_check.py --config strata-iq3_s.json --length 9000 --keep ab/chunks_iq
```

Lecture : si 8192 et 4096 diffèrent déjà sur le hash GDN (ou si le KL moyen contre le chemin token dépasse nettement
celui entre deux exécutions identiques, c'est-à-dire 0), l'écart vient des frontières de morceau et l'étalon
« 6144 contre 8192 » de `bench/results/2026-09-28-prefill-speed` mesure ce défaut plutôt qu'un ordre de sommation.

## Ce qui reste

- **Tout ce qui est GPU est à exécuter** : les tolérances de `rope_parity` (natif), `gdn_fused_parity`,
  `qsa_decode_parity` et `fused_gr_parity` (l'égalité bit à bit multi/simple est une promesse de l'en-tête, pas une
  mesure) peuvent se révéler trop serrées ou trop lâches ; les valeurs imprimées permettent de les recaler.
- #3 : la mesure elle-même (KL et aiguilles à 128K) demande le modèle ; l'option et l'outil sont prêts. Le RoPE du
  chemin de prompt (`src/prefill/kernels.cu`, précis, non fast-math) diffère de celui du décodage : à traiter avec le
  paquet prefill.
- #5 : aligner (ou non) le softplus du chemin fusionné sur llama.cpp est une décision de numérique, à prendre après
  la mesure ; rien n'a été changé.
- #13 : deux autres copies de l'arrondi bf16 (`ple.cu`, `prefill/kernels.cu`) gardent le défaut NaN (autres paquets).
- #9 : l'outil ne fait que mesurer ; si un défaut de frontière apparaît, sa correction est un autre travail.
- #8 : `pool_test` (`run_split`), `iq_parity` (`ncols` > 2) et les oracles llama.cpp restent hors de ce paquet.
