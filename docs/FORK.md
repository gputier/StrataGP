# StrataGP, le fork

StrataGP est un fork de [Strata](https://github.com/Niko1221/Strata), le moteur de Niko1221. Depuis le 17 octobre 2026,
`main` repart de l'amont à la version `v0.1.45`. Le fork n'ajoute plus aucun code. Le README, les autres documents de
`docs/` et le moteur sont ceux de l'amont. Son seul écart avec l'amont est ce document : `git diff --stat v0.1.45 main`
ne montre que `docs/FORK.md`.

`main` a été reconstruit comme à la 0.1.37, à la 0.1.39, à la 0.1.41 et à la 0.1.42 : une branche neuve depuis le tag
`v0.1.45`, ce document, puis une fusion qui rattache l'ancien `main` sans garder son contenu (`git merge -s ours`). L'ancien
`main` porte le tag `sauvegarde/main-avant-0.1.45`.

## Ce que l'amont a repris

Le fork n'a plus de code à lui. Ce qu'il ajoutait se trouve dans l'amont.

- [Niko1221/Strata#1852](https://github.com/Niko1221/Strata/pull/1852), cache d'encodage du prompt, la #567, qui
  reprenait le port de blange48, soumise de nouveau. Niko1221 l'a reprise dans la v0.1.45 par le commit `f13a870d5`,
  avec gputier et blange48 en co-auteurs. Le travail a été rebasé à la main sur la 0.1.44.1, où `prepare()` prend un
  nouvel argument `cancel`. Les fichiers `tools/strata_tokenizer.py`, `serve/test_prompt_encoder.py`,
  `tools/test_strata_tokenizer.py` et `SECURITY.md` sont identiques à ceux de notre branche. `serve/server.py` porte les
  mêmes changements. Mesure de Niko1221 : sur une conversation synthétique de 140 000 jetons, l'encodage à chaud passe
  de 22,6 ms à 5,4 ms, avec les mêmes identifiants. Les PR #1455 (W1nge) et #1823 (xrip), encore ouvertes, font le même
  travail, et Niko1221 les dit couvertes par celle-ci.

  Le serveur ne retokenise que la fin du prompt, à partir du dernier jeton spécial qu'il partage avec un prompt récent
  (`PromptEncoder` dans `tools/strata_tokenizer.py`, appelé par `Service.encode_prompt` dans `serve/server.py`). Les
  identifiants obtenus sont les mêmes qu'avec un encodage complet, et étaient les mêmes qu'avec la 0.1.42 nue. Mesuré sur la
  0.1.41, dans un conteneur Linux de 10 processeurs : à 125 000 jetons renvoyés avec un tour de plus, 42,5 ms ramenées
  à 11,0 ms par tour ; à 89 000 jetons renvoyés, 29,5 ms ramenées à 6,6 ms. Sur la 0.1.42, les passages en texte brut
  des #537 et #931 (une balise `</think>` ou un `<|im_end|>` cité dans un message reste du texte) décident aussi des
  jetons spéciaux retenus. Le préfixe commun s'arrête donc là où ces passages diffèrent entre les deux prompts. Ce
  report est de blange48 : il figure dans le commit `f13a870d5`, avec lui en co-auteur.

  La PR comptait huit commits : le cache, les tests de serveur recalés sur un `encode_prompt` complet, un test des
  vecteurs de llama.cpp qui prend le vocabulaire qwen35 au lieu de celui d'un paquet choisi par `$STRATA_TOKENIZER`, le
  nettoyage de `ByteTokenizer`, la règle des prompts gardés, et deux corrections de revue. La liste n'en compte que
  sept parce que la première correction tient en deux commits : les prompts gardés suivent les emplacements du cache
  de conversation du moteur, et le second commit couvre le démarrage paresseux. La seconde correction : le comptage de
  jetons ne stocke pas de prompt.

  Cette règle vient de ce que tout prompt rendu commence par l'en-tête du modèle de conversation, une frontière de jeton
  spécial. Le prompt d'une autre conversation passait pour une suite du prompt gardé et le remplaçait : la requête d'un
  sous-agent coûtait sa réutilisation à la conversation principale au tour suivant (8 276 caractères réutilisés, puis
  236). Un prompt ne remplace plus que l'entrée dont il reprend plus de la moitié du texte. Sinon il devient une entrée
  neuve. Le cache garde une entrée par emplacement du cache de conversation du moteur (`--conversation-cache-slots`),
  4 sans ce réglage. Un prompt refusé parce qu'il ne tient pas dans le contexte n'est pas gardé, comme ses jetons ne
  l'étaient pas avant la #567. Les autres refus le gardent, car le client renvoie le même prompt.

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

## Ce qui n'a pas été repris

Deux choses ne sont pas reprises, donc perdues pour le fork. Le second commit de la #568 : `ple_parity` reste lié aux
fichiers `ple_in.bin` et `ple_out.bin`, absents du dépôt, au lieu de la référence en double précision écrite dans le
test. Et, de la #569, le refus de `setup.py --api-key` quand la clé est vide.

La #1934 n'a pas été reprise. Elle élargit la recherche d'échelle Q2_0 dans `tools/mtp_pack.py`, de 0,15 à 1,0 fois
max|w|, avec deux signes.

Mesure de Niko1221 sur une RTX 3060, 4 prompts anglais de 200 jetons, 5 tours alternés, `--spec 4` : acceptation de
2355/3567 (0,660) avec l'ancien pack, 2325/3585 (0,648) avec le nouveau. Décodage de 50,9 contre 52,4 jetons par
seconde : l'écart tient dans le bruit de mesure. Le pack prend 9 min 15 au lieu de 2 min 20.

Notre mesure de la #1934, rapportée dans le corps de la PR, sur la RTX 5090 avec la liste de brouillon française en
service, donne sur 8 paires une acceptation de 0,716 à 0,732 et un débit de sortie de +9,4 % sur un prompt de 22 785
jetons (8 paires sur 8). Les prompts courts et la lecture à froid de 99K jetons sont dans le bruit. Nous avons fermé la
PR le 17/10/2026. La conclusion : un gain mesuré sur une seule carte, avec la liste de brouillon française, ne justifie
pas un pack quatre fois plus long, et Niko1221 ne le retrouve pas sur la RTX 3060. La branche est gardée sous le tag
`sauvegarde/pr-mtp-scale-0.1.42`. La tête de brouillon française construite avec ce packer (`mtp-fr-ms`) reste en
service chez nous.

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

Chaque redémarrage laisse un tag : `sauvegarde/main-avant-0.1.37`, `sauvegarde/main-avant-0.1.39`,
`sauvegarde/main-avant-0.1.41`, `sauvegarde/main-avant-0.1.42` et `sauvegarde/main-avant-0.1.45`. `main` rattache cet
historique par une fusion qui n'en garde pas le contenu : `git log` le montre, l'arbre non.

## Compiler et valider sous Windows

Le moteur précompilé que télécharge `setup.py`, lancé par `START-HERE.bat`, vient des releases de l'amont
(`setup.py:175`, `PREBUILT_URL`). Le fork n'ayant plus de code à lui, ce moteur et le serveur de l'amont sont ceux du
fork. Pour compiler le moteur soi-même, avec les Build Tools de Visual Studio et le CUDA Toolkit :

```bat
START-HERE.bat --setup --build
```

`--build` compile le moteur au lieu d'utiliser le moteur prêt à l'emploi (`setup.py`). Sur la station de test (la `.99`,
RTX 5090), chaque version se construit par un script `gpNNN-build.bat` à côté du clone (Release, CUDA 13.4, sm_120),
voir la page StrataGP du wiki.

Validation, depuis le dossier de build du clone : `ctest` pour les tests de parité, puis les tests Python
`serve/test_*.py` et `tools/test_*.py`.
