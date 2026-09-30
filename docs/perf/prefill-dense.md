# prefill-dense : projections denses du prefill en MMQ int8 (#39, P4) et part de l'attention QSA (#41, P6)

*29/09/2026 : ce document a été écrit sur une machine sans GPU ; la première exécution sur une RTX 5090 est dans
[PERF-CHANGES.md, section 6](../PERF-CHANGES.md#6-première-exécution-sur-gpu-rtx-5090-29092026).*

Branche `perf/prefill-dense`, partie de `perf/base` (b858f9b, moteur 0.1.20). La machine de développement n'a pas
de GPU : aucune mesure GPU n'a été faite. Tout ce qui touche au GPU a été compilé (sm_120, CUDA 13.0), rien n'a été
exécuté. Les gains sont **ESTIMÉ** (calcul) ou **HYPOTHÈSE** (à confirmer au profileur).

| Changement | Défaut | Commutateur |
|---|---|---|
| Projections denses natives en MMQ int8 (#39) | **désactivé** (change l'arrondi) | `--prefill-dense-mmq` ou `STRATA_PREFILL_DENSE_MMQ=1` |
| Phase « dense proj » et lignes par morceau dans `STRATA_PREFILL_TIMING` (#41) | actif dès que la variable est posée | `STRATA_PREFILL_TIMING=1` (mesure seulement) |
| En-tête de `gemm.hpp` : la GEMM des poids quantifiés est en FP16, pas en BF16 | - | - |

Sans `--prefill-dense-mmq`, le calcul est **identique au bit près** à `perf/base` : `Gemm::native_mmq` refuse
l'appel tant qu'aucun contexte MMQ ne lui a été donné, et `native_proj` retombe sur le même `Gemm::native`.

## #39 (P4) : les projections denses natives en MMQ

### Avant

Les projections natives (GGUF) du prefill passaient par `Gemm::native` : déquantification du poids en FP16 dans un
tampon de 64 Mio, activations arrondies en FP16, GEMM cuBLAS FP16 avec accumulation FP32 (demi-débit sur GeForce).
Concerné, par morceau : GDN `attn_qkv`, `attn_gate`, `ssm_out` (36 couches) ; QSA `attn_q`, `attn_k`, `attn_v`,
`attn_output` (12 couches) ; expert partagé `ffn_gate_shexp`, `ffn_up_shexp`, `ffn_down_shexp` (48 couches).

### Maintenant, avec `--prefill-dense-mmq`

`native_proj` (`src/prefill/prefill.cpp`) appelle d'abord `Gemm::native_mmq` (`src/prefill/gemm.cu`) :

1. **Activations → q8_1** avec le quantificateur MMQ de llama.cpp (`mmq::quantize`, déjà utilisé pour les experts).
   Source FP32 quand le chemin la garde : `mixed` pour qkv, gate, q, k, v et gate/up de l'expert partagé ; `y` (sortie
   GDN) pour `ssm_out`. Pour `attn_output` (`attn_h`) et `ffn_down_shexp` (`sh_h`), le prefill ne garde que la
   version FP16 : elle est élargie en FP32 (`widen_kernel`, 10 registres, aucun spill) puis quantifiée. L'arrondi FP16
   intermédiaire (~2^-11 relatif) est petit devant celui du q8_1 (~0,65 % RMS relatif sur des données synthétiques).
2. **Produit MMQ** (`mmq::Context::dense`, `src/prefill/moe_mmq.cu`) : les poids restent dans leurs blocs GGUF,
   produits int8 sur tenseurs. Le chemin utilise le **mode dense** du kernel MMQ, c'est-à-dire le lancement que llama.cpp fait
   lui-même pour un produit dense (`ggml_cuda_mul_mat_q` sans `ids` : un canal, pas de bornes, pas de table de
   lignes), plutôt que le mode « un seul expert » (`bounds = {0, T}`, ids identité) suggéré par l'audit. C'est le même
   kernel et le même calcul (colonne j du produit = ligne j de Y dans les deux cas), mais sans les deux tableaux sur
   le GPU. Ceux-ci vivraient dans les emplacements prêtés du cache d'experts et devraient être réécrits à chaque
   prompt, comme `ids_identity`.
3. **Mémoire** : pas de nouveau tampon de travail en VRAM (voir la réserve sur le « stream-k fixup » ci-dessous). Le tampon de déquantification (64 Mio, déjà emprunté) accueille les lignes
   q8_1. Si les lignes d'un morceau n'y tiennent pas, le produit est fait par tranches de tokens (multiples de 128).
   À 8192 tokens, seul `attn_output` (K = 6144, source FP16 élargie) est coupé, en 4 tranches de 2048. Le contexte
   de lancement MMQ (et son pool pour le « stream-k fixup ») est celui des experts.
   **Réserve** : ce pool (`CachingPool`, `src/prefill/ggml_cuda_host.cu`) fait un `cudaMalloc` au premier usage,
   pendant le prompt, jusqu'à nsm × 128 × 128 × 4 o (~11 Mo sur une 5090), et `CUDA_CHECK` arrête le processus si la
   VRAM est épuisée. Quand les experts passent déjà par MMQ, ce tampon est partagé et ne coûte rien de plus. Quand ils
   n'y passent pas (`STRATA_PREFILL_MMQ=0`, ou experts IQ1_M : `mmq_plan().any` faux), `--prefill-dense-mmq` ajoute
   cette allocation, que `bytes_needed` ne compte pas : garder ~16 Mo de marge VRAM dans ce cas.
   Quand `native_mmq` refuse une projection (type non couvert, K % 4 != 0, tampon trop petit, build sans MMQ), le
   prefill l'écrit sur stderr une fois par couple (type, raison), avec la raison, et la projection reste en FP16.
4. **Lecture au-delà du poids** : MMQ lit une ligne de poids par tuiles de 256 valeurs (llama.cpp remplit chaque
   ligne quantifiée jusqu'à un multiple de 512 pour cela). Quand K n'est pas un multiple de 512 (`ffn_down_shexp`,
   K = 640), la dernière tuile de la dernière ligne déborde du tenseur. Ces valeurs rencontrent des activations
   nulles, ce qui n'est sans danger que si elles décodent en nombres finis (une échelle NaN donne NaN × 0 = NaN).
   Un tel poids est donc copié dans le tampon, suivi de 4 Kio de zéros (~1,7 Mo par couche MoE en Q8_0 : ~2 µs).
5. **Types couverts** : Q4_0, Q5_0, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, Q2_0, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S,
   IQ4_NL et IQ4_XS (`mmq::dense_supported`). **IQ1_M** (non couvert par MMQ), `beta ≠ 0` et une construction sans
   MMQ restent sur le chemin FP16 cuBLAS, avec un message une fois par type
   (`--prefill-dense-mmq: <tenseur> (GGML type N) stays on the FP16 GEMM`). Au démarrage, le moteur affiche
   `strata prefill: dense projections through MMQ (q8_1 activations, --prefill-dense-mmq)`.
6. La clé PLE native du bloc PLE (`m.gemm.native` dans le bloc PLE) n'est pas touchée : elle reste en cuBLAS.

La bibliothèque `strata_mmq` compile 7 instances MMQ de plus (q4_0, q5_0, q8_0, q3_k, q4_k, q5_k, q6_k) : environ
2 minutes de compilation en plus à -j3 sur la machine de développement. `ptxas -v` sur l'instance q8_0 : les
variantes J = 112 et 128 utilisent 255 registres et débordent de 8 à 32 octets sur la pile. L'instance q2_0, déjà
utilisée pour les experts, fait pareil (32 à 40 octets) : c'est le code amont de llama.cpp, inchangé.

### Gains attendus

Paramètres natifs par morceau : 36 × 57,7 M (GDN) + 12 × 49,8 M (QSA) + 48 × 4,9 M (expert partagé) ≈ 2,9 G, soit
**~48 TFLOP par morceau de 8192 tokens** (ESTIMÉ). Les 59 TFLOP de l'audit comptent aussi les lectures GR, le
routeur et l'indexeur, qui restent en BF16 cuBLAS.

- **RTX 5070 (ESTIMÉ)** : FP16 avec accumulation FP32 à ~62 TFLOPS crête (~1 s par morceau), INT8 à ~247 TOPS crête.
  Avec MMQ à 40-60 % de la crête, il faut 0,3 à 0,5 s. Le gain est de 0,5 à 0,7 s sur ~6,35 s par morceau, soit
  **8 à 11 %**, dans la ligne des ~10 % de l'audit.
- **RTX 5090 (HYPOTHÈSE)** : FP16/FP32 à ~209 TFLOPS crête (~0,3 s), INT8 à ~838 TOPS crête (0,11 à 0,15 s). Le gain
  est de ~0,2 s par morceau. Les parties séquentielles ne s'accélèrent pas autant ; avec ~3 s par morceau, cela fait
  **4 à 7 %** de débit de prefill.
- **Coût ajouté** (compris dans « dense proj ») : la quantification des activations, ~40 Go de trafic mémoire par
  morceau, soit ~30 ms sur une 5090 et ~60 ms sur une 5070. `mixed` est quantifié une fois par projection (2 fois
  par couche GDN, 3 fois par couche QSA, 2 fois pour l'expert partagé) : voir « Reste à faire ».
- Décodage : aucun effet (le chemin des fenêtres de vérification n'est pas touché).

### Précision

L'arrondi change : activations q8_1 au lieu de FP16. C'est donc **opt-in**. Il faut le juger comme les experts MMQ
l'ont été (`bench/results/2026-09-28-prefill-speed/README.md`), par KL forcé par l'enseignant sur les 256 dernières
positions d'un prompt de 32K. L'étalon est l'écart que produit déjà un simple changement de taille de morceau :
chunk 6144 contre 8192, top-1 identique à 89,8 %, KL moyen 0,33. Les experts MMQ ont donné 85,9 % et 0,38.

Sur données synthétiques, émulé sur CPU (poids gaussiens quantifiés par ggml, activations gaussiennes avec 1 % de
valeurs ×4, K = 1024) : q8_1 contre produit exact, **0,63 à 0,66 % RMS relatif** pour les 15 types. Activations FP16
(chemin actuel) : 0,021 %.

## #41 (P6) : la part de l'attention QSA dans `STRATA_PREFILL_TIMING`

La phase `qsa attn` existait déjà dans la ligne de synthèse. Ce qui manquait pour décider d'un nouveau noyau
d'attention :

- **Phase `dense proj`** : chaque projection native (GDN qkv/gate/out, QSA q/k/v/o, expert partagé) est chronométrée
  à part. Elle est sortie de `gdn`, `qsa proj` et `router+shared`, qui ne la contiennent donc plus. C'est aussi la
  phase que `--prefill-dense-mmq` change.
- **Par morceau** : chaque marque porte son morceau. Après la ligne par phase, le moteur affiche une ligne pour le
  prompt entier, puis une par morceau (si plus d'un), avec le temps GPU, `qsa attn` (le kernel de décodage, 32
  requêtes par lancement), `qsa indexer+select` et `dense proj`, en ms et en % :

```
strata prefill timing: 32768 tokens, GPU timeline ... ms, wall ... ms, host staging ... ms: embed+steps ... dense proj ...
strata prefill timing: prompt, GPU 12345 ms: qsa attn 1234 ms (10.0%), qsa indexer+select 456 ms (3.7%), dense proj 2345 ms (19.0%)
strata prefill timing: chunk 0, positions 0-8191, GPU ... ms: qsa attn ... ms (...%), ...
strata prefill timing: chunk 3, positions 24576-32767, GPU ... ms: qsa attn ... ms (...%), ...
```

Un morceau attend toutes les positions qui le précèdent : la part de l'attention croît avec la profondeur. Les
lignes par morceau montrent cette croissance ; c'est la mesure que demande #41 avant tout noyau flash clairsemé. Le
noyau lui-même n'est pas écrit ici. Règle de décision proposée : s'il dépasse ~10-15 % du GPU sur les derniers
morceaux d'un prompt de 64K à 128K sur la 5090, le noyau flash sur l'union des blocs d'une tuile de 64 requêtes
vaut l'effort L.

Coût : deux événements de plus par projection native, uniquement quand `STRATA_PREFILL_TIMING` est posée. Sans elle,
rien n'est enregistré.

## Tests

Sur la machine de développement (CPU seulement) :

- Construction complète (`cmake --build build-wp -j 3`, Release, CUDA 13.0, sm_120) : **OK**.
- `ple_reader_selftest`, `platform_memory_test`, `pool_stress`, `expert_multi_test`, `suffix_drafter_test`,
  `draft_policy_test`, `controller_test`, `conv_cache_test` : **OK**. `pool_test` échoue comme attendu (il lui faut un
  pack de modèle). `python3 serve/test_server.py` : **OK** (26 tests). Note : `ple_reader_selftest` passe quand il
  tourne seul, mais écrit un fichier fixe (`/tmp/ple_reader_selftest.bin`) et peut échouer s'il est lancé en même temps
  depuis un autre arbre de travail (il a échoué une fois ainsi ; relancé seul, OK).
- Calibrage des bornes du test GPU sur CPU (programme jetable, non versionné) : voir « Précision ».

Nouveau test GPU, **compilé, pas exécuté** : `prefill_dense_mmq_parity` (ctest,
`src/prefill/dense_mmq_parity.cpp`). Ici, il s'arrête avec « no CUDA device ». Poids synthétiques quantifiés par
ggml, pour les 15 types couverts :

- chaque sortie est finie, et à moins de 3 % RMS relatif du produit exact FP64 (attendu ~0,65 %) ;
- pour les types de disposition D4 (tous sauf Q4_0, Q4_K, Q5_K), à moins de 2e-3 du produit avec les activations
  arrondies comme le fait le quantificateur MMQ (émulé sur l'hôte), c'est-à-dire à l'ordre de sommation FP32 près ;
- une seconde exécution est identique au bit près, et les colonnes au-delà de N (ldy > N) ne sont pas touchées ;
- N non multiple de 128 (tuiles vérifiées de MMQ), T = 1, 128 et 129 ;
- activations FP16 élargies ; tranches de tokens (petit tampon) contre une seule passe (< 1e-5) ;
- forme K = 640 avec des octets NaN posés juste après le poids : la copie avec zéros doit les tenir à l'écart ;
- formes réelles à petit morceau (GDN qkv, QSA q, `ssm_out` en FP16), référence échantillonnée ;
- appels à refuser sans rien lancer : pas de contexte, IQ1_M, beta = 1, tampon trop petit ;
- le chemin FP16 cuBLAS est aussi comparé au produit exact (< 5e-3), ce qui valide la référence du test.

## À valider sur la RTX 5090

```bash
# construction
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON \
      -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j

# tests
ctest --test-dir build -R "prefill_dense_mmq_parity" --output-on-failure
# conseillé une fois : accès mémoire hors bornes (lectures MMQ au-delà des tampons)
compute-sanitizer --tool memcheck ./build/prefill_dense_mmq_parity
```

**Benchmark A/B** : 3 prompts × 3 exécutions, glouton, 256 tokens, résidence statique. Prendre des prompts longs
(par exemple 8K, 32K et 64K tokens) pour que le prefill compte. `<OPTS>` est la ligne habituelle (`args` de
`strata-<modèle>.json` : `--pack`, `--native`, `--ple-gguf`, `--expert-profile`, `--expert-cache auto`,
`--prefill auto`, `--spec`, `--mtp`, ...).

```bash
OPTS="<OPTS> --greedy --max-new 256 --adapt-every 100000"
mkdir -p ab
for p in p1 p2 p3; do for r in 1 2 3; do
  ./build/strata $OPTS --tokens-file prompts/$p.txt                     > ab/fp16_${p}_$r.txt 2>&1
  ./build/strata $OPTS --tokens-file prompts/$p.txt --prefill-dense-mmq > ab/mmq_${p}_$r.txt  2>&1
done; done
grep -H "^prefill" ab/*.txt        # "prefill N tokens in X ms -> Y tok/s (time to first token Z ms)"
grep -H "^decode"  ab/*.txt        # le décodage ne doit pas bouger
grep -H "dense projections through MMQ\|stays on the FP16 GEMM" ab/mmq_*.txt   # le chemin est bien pris
```

**Répartition du prefill** (#41, et pour voir le gain de #39 dans la phase `dense proj`) : une exécution de chaque
bras avec la mesure. Les événements coûtent un peu, donc ne pas comparer ces tok/s aux précédents.

```bash
STRATA_PREFILL_TIMING=1 ./build/strata $OPTS --tokens-file prompts/p3.txt                     2>&1 | grep "prefill timing"
STRATA_PREFILL_TIMING=1 ./build/strata $OPTS --tokens-file prompts/p3.txt --prefill-dense-mmq 2>&1 | grep "prefill timing"
```

**Précision** (#39) : KL forcé par l'enseignant sur un prompt de 32K. Le prompt est lu par le prefill jusqu'aux 256
dernières positions, qui passent par le chemin token avec leurs logits écrits. `<OPTS_KL>` est `<OPTS>` sans
`--spec`, `--spec-min-p` ni `--mtp`.

```bash
P=prompts/p32k.txt
N=$(python3 -c "import re,sys; print(len(re.findall(r'\d+', open('$P').read())))")
KL="<OPTS_KL> --greedy --max-new 1 --adapt-every 100000 --prefill-until $((N-256)) --tokens-file $P"
./build/strata $KL --prefill 8192                     --dump-logits ab/ref.bin
./build/strata $KL --prefill 8192 --prefill-dense-mmq --dump-logits ab/mmq.bin
./build/strata $KL --prefill 6144                     --dump-logits ab/c6144.bin   # l'étalon
cat > ab/kl.py <<'EOF'
import sys, numpy as np
def rows(p):
    v = int(np.fromfile(p, dtype=np.int32, count=2)[0])
    x = np.fromfile(p, dtype=np.float32, offset=8)
    return x[: x.size // v * v].reshape(-1, v)
a, b = rows(sys.argv[1]), rows(sys.argv[2])
n = min(len(a), len(b), 256)
kl, top = [], 0
for x, y in zip(a[-n:].astype(np.float64), b[-n:].astype(np.float64)):
    lx = x - x.max(); lx -= np.log(np.exp(lx).sum())
    ly = y - y.max(); ly -= np.log(np.exp(ly).sum())
    kl.append((np.exp(lx) * (lx - ly)).sum()); top += x.argmax() == y.argmax()
print(f"{sys.argv[2]}: {n} positions, top-1 identique {100 * top / n:.1f} %, KL moyen {np.mean(kl):.3f}")
EOF
python3 ab/kl.py ab/ref.bin ab/c6144.bin    # étalon attendu : ~89.8 %, ~0.33 (README prefill-speed)
python3 ab/kl.py ab/ref.bin ab/mmq.bin      # à accepter si du même ordre que l'étalon
```

Puis les aiguilles, comme pour les experts MMQ (`tools/needle_bench.py`, 1K à 128K, avec `--prefill-dense-mmq`
dans les arguments du moteur).

## Reste à faire, limites

- **Aucune mesure GPU.** Le test de parité, le gain et la précision sont à confirmer sur la 5090. Le chemin reste
  opt-in tant que le KL n'a pas été mesuré.
- **Quantifier `mixed` une fois par couche** (au lieu d'une fois par projection) quand les types des projections
  partagent la même disposition q8_1 : ~2/3 du coût de quantification en moins. Non fait : il faudrait une
  invalidation explicite par couche, et le gain (~20 ms par morceau sur une 5090) ne justifie pas le risque avant
  d'avoir la mesure.
- **FP32 de `attn_h` et `sh_h`** : garder une sortie FP32 de `gate_attn` et `swiglu_pair` éviterait l'élargissement
  et le double arrondi. Non fait, pour ne pas toucher `src/prefill/kernels.cu`, que d'autres paquets modifient.
- **Nouveau noyau d'attention (P6)** : volontairement pas écrit. À décider sur les chiffres par morceau ci-dessus.
