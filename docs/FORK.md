# StrataGP, le fork

StrataGP est un fork de [Strata](https://github.com/Niko1221/Strata), le moteur de Niko1221. Depuis le 5 octobre 2026,
`main` repart de l'amont à la version `v0.1.39` (tag `6f32ec0`). Le README, les autres documents de `docs/` et le moteur
sont ceux de l'amont, sauf les cinq ajouts ci-dessous. Mesuré par `git diff --stat v0.1.39 main`, le delta du fork est
de 13 fichiers, ce document compris.

`main` a été reconstruit comme à la 0.1.37 : une branche neuve depuis le tag `v0.1.39`, la fusion des cinq branches de
PR, ce document, puis une fusion qui rattache l'ancien `main` sans garder son contenu (`git merge -s ours`). L'ancien
`main` porte le tag `sauvegarde/main-avant-0.1.39`.

## Ce que le fork ajoute

Cinq branches, fusionnées dans `main` et proposées une à une à l'amont. La 0.1.39 n'en reprend aucune.

- [Niko1221/Strata#567](https://github.com/Niko1221/Strata/pull/567), cache d'encodage du prompt. Le serveur ne
  retokenise que la fin du prompt, à partir du dernier jeton spécial qu'il partage avec un prompt récent
  (`PromptEncoder` dans `tools/strata_tokenizer.py`, appelé par `Service.encode_prompt` dans `serve/server.py`). Les
  identifiants obtenus sont les mêmes qu'avec un encodage complet. Mesuré en conteneur : 235 ms ramenées à 1,5 ms par
  tour à 100 000 jetons. Sur la 0.1.39, les passages en texte brut de la #537 (une balise `</think>` citée dans un
  message reste du texte) décident aussi des jetons spéciaux retenus. Le préfixe commun s'arrête donc là où ces
  passages diffèrent entre les deux prompts. Ce report est celui de blange48, repris dans notre commit avec lui en
  co-auteur. Niko1221 relira la PR dans une version suivante, avec les autres changements du cache de prompt.
- [Niko1221/Strata#568](https://github.com/Niko1221/Strata/pull/568), tests de parité. `ple_parity` compare le bloc PLE
  à une référence en double précision écrite dans le test, au lieu des fichiers `ple_in.bin` et `ple_out.bin` absents du
  dépôt. `gr_parity` et `ple_parity` attendent la fin de leur préparation avant de lancer leurs kernels sur un flux non
  bloquant.
- [Niko1221/Strata#569](https://github.com/Niko1221/Strata/pull/569), clé d'API vide refusée : par le champ `api_key` du
  fichier de configuration du serveur (`serve/server.py`, avant le chargement du modèle) et par `setup.py --api-key`.
- [Niko1221/Strata#570](https://github.com/Niko1221/Strata/pull/570), `--gguf-dir` en lecture seule : si le dossier
  n'accepte pas l'écriture, `setup.py` avertit au lieu de s'arrêter quand il ne peut pas poser la marque de fin sur un
  fichier complet.
- [Niko1221/Strata#594](https://github.com/Niko1221/Strata/pull/594), corps de requête lu avant la fermeture. Le
  serveur HTTP/1.0 répond parfois sans avoir lu le corps (401 mauvaise clé, 403 origine ou `Host`, 413, 501 méthode sans
  gestionnaire) puis ferme la connexion. Sous Windows, fermer sur des octets non lus envoie un RST : un client Python
  (`http.client`, `urllib`, `requests`), qui envoie le corps après les en-têtes, reçoit `WinError 10054` au lieu de la
  réponse. `curl` n'est pas touché. Après chaque réponse, `_drain_body` dans `serve/server.py` lit et jette le corps
  qu'aucun gestionnaire n'a pris. La limite est de 5 s pour tout le corps, sans limite en octets, et la lecture se fait
  une lecture de socket à la fois (`read1`), sinon un client qui envoie un octet de temps en temps n'est jamais lâché.
  Un corps chunked reste non lu, comme avant. La 0.1.39 lit déjà le corps de `/load` et `/unload` (#630) : ces deux
  routes et `/config` passent maintenant par `_body`, qui marque le corps comme lu, sinon la connexion se fermait 5 s
  trop tard. Un corps de `/load` qui arrive en plus de 2 s reçoit un 400 : `_body` ne le marque alors pas comme lu, et
  le drain lit la suite sur la socket, car Python refuse toute lecture de `rfile` après un dépassement de délai. La
  classe `AnswerBeforeTheBody` de `serve/test_server.py` compte 10 tests.

## La liste de brouillon française (#664)

Le fork portait une liste de brouillon française (`setup.py --draft-vocab fr`), proposée dans la #664 à la demande de
l'amont (issue #597). Niko1221 a construit la sienne dans la 0.1.39 à partir de notre description, avec le même réglage
`--draft-vocab fr` et l'option `tools/draft_vocab.py --corpus ... --coverage 0.99`, puis a fermé la PR. Le fork prend
celle de l'amont. Sa mesure, sur une RTX 5070 avec IQ3_XXS et 8 prompts français : acceptation des brouillons de 0,505
avec `en` à 0,597 avec `fr`, sans perte en anglais ni en code. La nôtre, sur la RTX 5090 avec IQ3_S et 12 prompts : de
0,52 à 0,64, génération de 141 à 158 jetons par seconde. L'ancienne version reste lisible sous le tag
`sauvegarde/main-avant-0.1.39`.

## Pourquoi un redémarrage depuis la 0.1.37

L'ancien `main` portait seize lots de performance sur la base 0.1.21 puis 0.1.28. Mesuré le 2 octobre sur la RTX 5090,
modèle IQ3_S, trois passages par cas, la 0.1.37 nue lit le prompt plus vite que cet ancien `main` : 1 742 contre
1 584 jetons par seconde à 2 520 jetons, 2 491 contre 2 180 à 23 020. La génération est à 3 % près. Notre marche
groupée du prefill apportait +12 % sur l'ancien `main`, qui restait pourtant sous la 0.1.37. L'amont a repris les lots
Q2_0 groupé et grille IQ dans sa 0.1.31 (anciennes PR 241 et 242), et d'autres sous une autre forme.

Sur la 0.1.37, `STRATA_PF_FUSED=1` existe (lue dans `src/prefill/prefill.cpp`, décrite dans `docs/DETAILS.md`).
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

Validation, depuis le dossier de build du clone : `ctest` pour les tests de parité, puis les tests Python
`serve/test_*.py` et `tools/test_*.py`. Chaque ajout a ses tests : `ple_parity` et `gr_parity` pour #568,
`serve/test_prompt_encoder.py`, `serve/test_server.py` et `tools/test_strata_tokenizer.py` pour #567,
`serve/test_server.py` pour #569 et #594, `tools/test_setup_choices.py` pour #569 et #570.

Mesuré le 5 octobre sur `main` (commit `dce5834`, base 0.1.39), en conteneur Linux, Python 3.13, avec le vocabulaire
qwen35 de llama.cpp : la suite `serve` lance 290 tests, 9 ignorés, avec 1 erreur dans `serve/test_responses.py`
(`test_json_schema_text_format`) qui sort aussi sur la 0.1.39 nue ; `tools/test_strata_tokenizer.py` passe 11 tests, 1
ignoré. `tools/test_setup_amd.py`, `tools/test_setup_choices.py` et `tools/test_setup_golden.py` ont des échecs dans ce
conteneur, les mêmes test pour test sur la 0.1.39 nue. Pas encore mesuré sur cette base : `ctest` et les tests sous
Windows sur la `.99`. Le 3 octobre, sur la 0.1.38 (commit `236388d`), `ctest` passait 61 tests sur 61.
