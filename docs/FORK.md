# StrataGP, le fork

StrataGP est un fork de [Strata](https://github.com/Niko1221/Strata), le moteur de Niko1221. Depuis le 10 octobre 2026,
`main` repart de l'amont à la version `v0.1.41` (tag `fb58e0d`). Le README, les autres documents de `docs/` et le moteur
sont ceux de l'amont, sauf l'ajout ci-dessous. Mesuré par `git diff --stat v0.1.41 main`, le delta du fork est de
7 fichiers (786 ajouts, 11 suppressions), ce document compris.

`main` a été reconstruit comme à la 0.1.37 et à la 0.1.39 : une branche neuve depuis le tag `v0.1.41`, la fusion de la
branche de PR, ce document, puis une fusion qui rattache l'ancien `main` sans garder son contenu (`git merge -s ours`).
L'ancien `main` porte le tag `sauvegarde/main-avant-0.1.41`.

## Ce que le fork ajoute

Une branche, fusionnée dans `main` et proposée à l'amont.

- [Niko1221/Strata#1852](https://github.com/Niko1221/Strata/pull/1852), cache d'encodage du prompt. C'est la #567
  soumise de nouveau, sur la 0.1.41. Le serveur ne retokenise que la fin du prompt, à partir du dernier jeton spécial
  qu'il partage avec un prompt récent (`PromptEncoder` dans `tools/strata_tokenizer.py`, appelé par
  `Service.encode_prompt` dans `serve/server.py`). Les identifiants obtenus sont les mêmes qu'avec un encodage complet,
  et les mêmes qu'avec la 0.1.41 nue. Mesuré dans un conteneur Linux de 10 processeurs : à 125 000 jetons renvoyés avec
  un tour de plus, 42,5 ms ramenées à 11,0 ms par tour ; à 89 000 jetons renvoyés, 29,5 ms ramenées à 6,6 ms. Sur la
  0.1.41, les passages en texte brut des #537 et #931 (une balise `</think>` ou un `<|im_end|>` cité dans un message
  reste du texte) décident aussi des jetons spéciaux retenus. Le préfixe commun s'arrête donc là où ces passages
  diffèrent entre les deux prompts. Ce report est celui de blange48, repris dans notre commit avec lui en co-auteur. La
  PR compte cinq commits : le cache, les tests de serveur recalés sur un `encode_prompt` complet, un test des vecteurs
  de llama.cpp qui prend le vocabulaire qwen35 au lieu de celui d'un paquet choisi par `$STRATA_TOKENIZER`, le
  nettoyage de `ByteTokenizer`, et la règle des prompts gardés.

  Cette règle vient de ce que tout prompt rendu commence par l'en-tête du modèle de conversation, une frontière de jeton
  spécial. Le prompt d'une autre conversation passait pour une suite du prompt gardé et le remplaçait : la requête d'un
  sous-agent coûtait sa réutilisation à la conversation principale au tour suivant (8 276 caractères réutilisés, puis
  236). Un prompt ne remplace plus que l'entrée dont il reprend plus de la moitié du texte. Sinon il devient une entrée
  neuve, et le cache en garde 4. Un prompt refusé parce qu'il ne tient pas dans le contexte n'est pas gardé, comme ses
  jetons ne l'étaient pas avant la #567. Les autres refus le gardent, car le client renvoie le même prompt.

## Ce que l'amont a repris

Dans la 0.1.40 et la 0.1.40.2 (Niko1221), quatre de nos anciennes branches ont été reprises, en tout ou
en partie. Le fork n'en porte plus rien : il prend la version de l'amont.

- [Niko1221/Strata#594](https://github.com/Niko1221/Strata/pull/594), corps de requête lu avant la fermeture. Le
  premier commit (`_drain_body` dans `serve/server.py`) est dans la 0.1.40, avec celui de Niko1221 qui marque comme lus
  les corps de `/config` et de `/load`. Sous Windows, fermer une connexion sur des octets non lus envoie un RST, et un
  client Python reçoit `WinError 10054` au lieu de la réponse.
- [Niko1221/Strata#570](https://github.com/Niko1221/Strata/pull/570), `--gguf-dir` en lecture seule. Notre commit est
  repris tel quel, avec son auteur, dans la 0.1.40.2. Niko1221 y ajoute un second commit : `mark()` avertit et continue
  pour toutes les marques de fin, pas seulement celle d'un fichier complet.
- [Niko1221/Strata#569](https://github.com/Niko1221/Strata/pull/569), clé d'API vide, reprise en réduit. Une clé vide
  dans le champ `api_key` du fichier de configuration donne un avertissement au démarrage, pas un refus. Une clé vide
  donnée par `--api-key` ou `STRATA_API_KEY` est déjà refusée (#213), et une clé blanche l'est depuis la #956.
- [Niko1221/Strata#568](https://github.com/Niko1221/Strata/pull/568), tests de parité, reprise en partie. Le premier
  commit est pris : `gr_parity` et `ple_parity` attendent la fin de leur préparation avant de lancer leurs kernels sur
  un flux non bloquant.

Deux choses ne sont pas reprises, donc perdues pour le fork. Le second commit de la #568 : `ple_parity` reste lié aux
fichiers `ple_in.bin` et `ple_out.bin`, absents du dépôt, au lieu de la référence en double précision écrite dans le
test. Et le refus de `setup.py --api-key` quand la clé est vide.

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

Remesuré le 10 octobre sur la 0.1.41, modèle IQ3_S : la divergence de KL moyenne entre la marche par défaut et la marche
fusionnée est de 6,9e-5 à 2 048 jetons et de 6,3e-5 à 4 096, le plancher est à 0, et le jeton le plus probable est le
même pour 16 positions sur 16. Le constat de la 0.1.37 (des jetons différents) ne se retrouve pas tel quel : sur le
prompt p2, deux passages par défaut diffèrent entre eux au jeton 226, alors que la marche par défaut et la marche
fusionnée donnent les mêmes jetons. La mesure est publiée dans le
[commentaire de l'issue 519](https://github.com/Niko1221/Strata/issues/519#issuecomment-6098236429).

## Où lire l'ancien travail

Le commit `98ff354` de l'ancien `main` porte le tag `sauvegarde/main-avant-0.1.37`. Il contient l'audit
(`docs/AUDIT-PERF.md`), le détail issue par issue (`docs/PERF-CHANGES.md`) et un document par lot (`docs/perf/`), absents
de l'arbre actuel :

```sh
git show sauvegarde/main-avant-0.1.37:docs/PERF-CHANGES.md
git ls-tree --name-only sauvegarde/main-avant-0.1.37 docs/perf/
```

Chaque redémarrage laisse un tag : `sauvegarde/main-avant-0.1.37`, `sauvegarde/main-avant-0.1.39` et
`sauvegarde/main-avant-0.1.41`. `main` rattache cet historique par une fusion qui n'en garde pas le contenu : `git log`
le montre, l'arbre non.

## Compiler et valider sous Windows

Le moteur précompilé que télécharge `START-HERE.bat` vient des releases de l'amont : il ne contient pas cet ajout. Pour
compiler celui du fork, avec les Build Tools de Visual Studio et le CUDA Toolkit :

```bat
START-HERE.bat --setup --build
```

`--build` compile le moteur au lieu d'utiliser le moteur prêt à l'emploi (`setup.py`). Sur la station de test (la `.99`,
RTX 5090), `deploy-build.bat` ne reconstruit que la cible `strata` dans le dossier de build du clone, voir la page
StrataGP du wiki.

Validation, depuis le dossier de build du clone : `ctest` pour les tests de parité, puis les tests Python
`serve/test_*.py` et `tools/test_*.py`. L'ajout a ses tests : `serve/test_prompt_encoder.py`, `serve/test_server.py` et
`tools/test_strata_tokenizer.py`.

Mesuré le 10 octobre sur `fork/main-0.1.41` (fusion `a62a1924`), en conteneur Linux, Python 3.12, `pip install -r
requirements.txt` : `python -m pytest -q serve tools/test_strata_tokenizer.py` passe 598 tests, 18 ignorés, 0 échec.
