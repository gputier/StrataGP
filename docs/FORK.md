# StrataGP, le fork

StrataGP est un fork de [Strata](https://github.com/Niko1221/Strata), le moteur de Niko1221. Depuis le 3 octobre 2026,
`main` repart de la version amont `v0.1.37` (`CMakeLists.txt` annonce `0.1.37`). Le README, les autres documents de
`docs/` et le moteur sont ceux de l'amont, sauf les quatre ajouts ci-dessous.

## Ce que le fork ajoute

Quatre branches, fusionnées dans `main` et proposées une à une à l'amont :

- [Niko1221/Strata#567](https://github.com/Niko1221/Strata/pull/567), cache d'encodage du prompt. Le serveur ne
  retokenise que la fin du prompt, à partir du dernier jeton spécial qu'il partage avec un prompt récent
  (`PromptEncoder` dans `tools/strata_tokenizer.py`, appelé par `Service.encode_prompt` dans `serve/server.py`). Les
  identifiants obtenus sont les mêmes qu'avec un encodage complet. Mesuré en conteneur : 235 ms ramenées à 1,5 ms par
  tour à 100 000 jetons.
- [Niko1221/Strata#568](https://github.com/Niko1221/Strata/pull/568), tests de parité. `ple_parity` compare le bloc PLE
  à une référence en double précision écrite dans le test, au lieu des fichiers `ple_in.bin` et `ple_out.bin` absents du
  dépôt. `gr_parity` et `ple_parity` attendent la fin de leur préparation avant de lancer leurs kernels sur un flux non
  bloquant.
- [Niko1221/Strata#569](https://github.com/Niko1221/Strata/pull/569), clé d'API vide refusée : par le champ `api_key` du
  fichier de configuration du serveur (`serve/server.py`, avant le chargement du modèle) et par `setup.py --api-key`.
- [Niko1221/Strata#570](https://github.com/Niko1221/Strata/pull/570), `--gguf-dir` en lecture seule : si le dossier
  n'accepte pas l'écriture, `setup.py` avertit au lieu de s'arrêter quand il ne peut pas poser la marque de fin sur un
  fichier complet.

## Pourquoi un redémarrage depuis la 0.1.37

L'ancien `main` portait seize lots de performance sur la base 0.1.21 puis 0.1.28. Mesuré le 2 octobre sur la RTX 5090,
modèle IQ3_S, trois passages par cas, la 0.1.37 nue lit le prompt plus vite que cet ancien `main` : 1 742 contre
1 584 jetons par seconde à 2 520 jetons, 2 491 contre 2 180 à 23 020. La génération est à 3 % près. Notre marche
groupée du prefill apportait +12 % sur l'ancien `main`, qui restait pourtant sous la 0.1.37. L'amont a repris les lots
Q2_0 groupé et grille IQ dans sa 0.1.31 (anciennes PR 241 et 242), et d'autres sous une autre forme.

`STRATA_PF_FUSED=1` reste disponible sur la 0.1.37 (lue dans `src/prefill/prefill.cpp`, décrite dans `docs/DETAILS.md`).
Mesuré : lecture du prompt +7,5 % à 2 520 jetons et +17,5 % à 23 020, mais des jetons différents et une acceptation des
brouillons de 0,612 contre 0,760 à 2 520. La variable n'est donc pas activée par défaut.

## Où lire l'ancien travail

Le commit `98ff354` de l'ancien `main` porte le tag `sauvegarde/main-avant-0.1.37`. Il contient l'audit
(`docs/AUDIT-PERF.md`), le détail issue par issue (`docs/PERF-CHANGES.md`) et un document par lot (`docs/perf/`), absents
de l'arbre actuel :

```sh
git show sauvegarde/main-avant-0.1.37:docs/PERF-CHANGES.md
git ls-tree --name-only sauvegarde/main-avant-0.1.37 docs/perf/
```

`main` rattache cet historique par une fusion qui n'en garde pas le contenu : `git log` le montre, l'arbre non.

## Compiler et valider sous Windows

Le moteur précompilé que télécharge `START-HERE.bat` vient des releases de l'amont : il ne contient pas ces ajouts. Pour
compiler celui du fork, avec les Build Tools de Visual Studio et le CUDA Toolkit :

```bat
START-HERE.bat --setup --build
```

`--build` compile le moteur au lieu d'utiliser le moteur prêt à l'emploi (`setup.py`). Sur la station de test (la `.99`,
RTX 5090), `deploy-build.bat` ne reconstruit que la cible `strata` dans le dossier de build du clone, voir la page
StrataGP du wiki.

Validation, depuis le dossier de build du clone : `ctest` pour les tests de parité (53 sur 53 le 3 octobre sur la `.99`),
puis les tests Python `serve/test_*.py` et `tools/test_*.py`. Chaque ajout a ses tests : `ple_parity` et `gr_parity`
pour #568, `serve/test_server.py` et `tools/test_strata_tokenizer.py` pour #567 et #569, `tools/test_setup_choices.py`
pour #569 et #570.
