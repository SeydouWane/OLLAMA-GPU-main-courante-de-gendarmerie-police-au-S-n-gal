# OCR local de main courante manuscrite (gendarmerie / police, Sénégal) avec Ollama + GPU

Extraire le texte manuscrit d'une page de main courante (registre à 3 colonnes : **numéro de mention**, **heure**, **objet**) avec un modèle de vision exécuté **en local** par [Ollama](https://ollama.com) sur un GPU d'ordinateur portable. Aucune donnée ne quitte la machine pendant l'extraction.

Deux modèles ont été comparés sur la même photo, dans les mêmes conditions, contre une transcription faite à la main :

| | `qwen3-vl:4b-instruct` | **`glm-ocr`** ✅ |
|---|---|---|
| Temps par page | 12,4 – 13,3 s | **7,6 – 8,8 s** |
| VRAM utilisée | 3,88 Go | **2,26 Go** |
| Repères exacts (n° de mention, heures, dates, téléphone, noms…) | 16 / 29 | **23 / 29** |
| Mentions inventées | 2 | **0** |
| Mots retrouvés (rappel) | 59 – 60 % | 50 – 59 % |
| Mots corrects (précision) | 62 % | 50 – 63 % |

**Meilleur modèle : `glm-ocr`.** Il est plus rapide, prend 40 % de VRAM en moins et surtout il **n'invente pas** : ses erreurs sont des lettres mal lues (« Bassinou » pour « Bassirou »), visibles à la relecture. `qwen3-vl` produit au contraire des phrases fluides mais fausses (« Nous remercions Néton pour l'intérêt fourni »), ajoute des mentions qui n'existent pas et décale les numéros de mention. C'est le modèle par défaut de [`extract.py`](extract.py).

> ⚠️ Même avec `glm-ocr`, environ 4 mots sur 10 restent faux sur cette photo (image WhatsApp compressée, 1280 px pour une double page). Une relecture humaine reste nécessaire, en particulier pour les noms propres.

---

## Sommaire

1. [Ressources utilisées](#1-ressources-utilisées)
2. [Document testé](#2-document-testé)
3. [Méthode](#3-méthode)
4. [Résultats détaillés](#4-résultats-détaillés)
5. [Transcriptions de chaque modèle](#5-transcriptions-de-chaque-modèle)
6. [Problèmes rencontrés et optimisations](#6-problèmes-rencontrés-et-optimisations)
7. [Utilisation](#7-utilisation)
8. [Limites et pistes](#8-limites-et-pistes)
9. [Structure du dépôt](#9-structure-du-dépôt)

---

## 1. Ressources utilisées

### Matériel et logiciels

| Élément | Valeur |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop, **6 Go VRAM** (pilote 592.82, backend CUDA 13) |
| CPU | Intel Core i5-13420H (13ᵉ gén.) |
| RAM | 15,6 Go |
| Système | Windows 11 Famille |
| Ollama | 0.40.16 |
| Python | 3.11.6 (Pillow 10.4.0, requests 2.32.5) |
| Date des tests | 25 novembre 2026 |

### Modèles

| | `qwen3-vl:4b-instruct` | `glm-ocr` |
|---|---|---|
| Type | Modèle vision-langage généraliste (Qwen3-VL) | Modèle **spécialisé OCR** (GLM-OCR) |
| Paramètres | 4,0 milliards + encodeur d'image 415 M | 0,9 milliard + encodeur d'image 434 M |
| Quantification | Q4_K_M | F16 |
| Taille sur disque | 3,3 Go | 2,7 Go |
| VRAM en fonctionnement | 3,88 Go (contexte 6144) | 2,26 Go (contexte 8192) |
| Répartition | 100 % GPU | 100 % GPU |
| Tokens pour une demi-page (1600 px) | 2 027 | 2 581 |
| Lecture de l'image (encodeur sur GPU) | 2,8 – 3,5 s | 2,9 – 4,0 s |
| Vitesse d'écriture du texte | 36 tokens/s | 75 tokens/s |
| **Temps par page** (modèle chargé) | **12,4 – 13,3 s** | **7,6 – 8,8 s** |
| Photo complète (2 pages + chargement) | ≈ 40 s | **≈ 21 s** |

---

## 2. Document testé

Photo WhatsApp d'une double page de main courante (pages du 20/12/2021, page n° 65 à droite) : 1280 × 822 px, écriture cursive au stylo bleu, numéros et annotations de marge en rouge, texte en biais dans la marge. La photo n'est pas publiée dans ce dépôt.

---

## 3. Méthode

1. **Découpage** : la photo en paysage est coupée en deux pages (gauche / droite), chacune agrandie à 1600 px de haut pour donner plus de pixels par ligne manuscrite.
2. **Mêmes conditions pour les deux modèles** : même image, `temperature = 0`, toutes les couches sur le GPU (`num_gpu = 99`), encodeur d'image vérifié sur GPU avant de commencer, un seul modèle chargé à la fois.
3. **Consigne** :
   - `glm-ocr` : sa consigne native `Text Recognition:` ;
   - `qwen3-vl` : « Transcris mot pour mot tout le texte manuscrit de cette page de registre… Ne résume pas, ne corrige pas. Mot illisible = [illisible]. »
4. **Référence** : transcription faite à la main, par lecture zoomée de chaque zone de la photo ([`benchmark/reference.json`](benchmark/reference.json)). Quelques noms propres y restent incertains (Ousso Gaye, Mouma, Amidou Wilane, plaque DL 1564 E).
5. **Mesures** (script [`benchmark/compare.py`](benchmark/compare.py)) :
   - **rappel** : part des mots de la référence retrouvés par le modèle ;
   - **précision** : part des mots du modèle présents dans la référence (une précision faible signale des mots inventés) ;
   - **repères** : 29 informations clés (numéros de mention, heures, dates, n° de page, téléphone, plaque, noms) recherchées à l'identique ;
   - temps, VRAM et vitesse sont lus dans les réponses de l'API Ollama.

---

## 4. Résultats détaillés

| Page | Modèle | Temps | Lecture image | Tokens sortis | Rappel | Précision | Repères | Repères manqués |
|---|---|---|---|---|---|---|---|---|
| Gauche | qwen3-vl | 12,4 s | 2,8 s | 330 | 59,7 % | 61,8 % | 5/12 | 09H20, 20-12-2021, quatre, Spark 7, Camon 12, Dianka, Habib Diallo |
| Gauche | **glm-ocr** | 7,6 s | 2,9 s | 321 | 49,7 % | 50,0 % | **9/12** | 20-12-2021, Camon 12, Habib Diallo |
| Droite | qwen3-vl | 13,3 s | 2,8 s | 358 | 59,0 % | 61,7 % | 11/17 | 20/12/2021, 65, 0123, DL 1564 E, Bassirou Ndoye, Wilane |
| Droite | **glm-ocr** | 8,8 s | 3,0 s | 399 | 59,0 % | 62,9 % | **14/17** | 10H40, DL 1564 E, Bassirou Ndoye |

Ce que les chiffres ne montrent pas :

| Critère | qwen3-vl | glm-ocr |
|---|---|---|
| Mentions inventées | **2** : « 0137 – 09H45 » en bas de la page gauche, « 0139 » sur la page droite | aucune |
| Numéros de mention attachés à la bonne mention | non : sur la page droite, 0137 est attribué à la suite de la mention précédente, 0138 au véhicule | oui |
| Type d'erreur | phrases fluides mais fausses, difficiles à repérer | lettres mal lues, faciles à repérer |
| Découpage des mots | phrases recomposées | une ligne par ligne du registre, coupures en fin de ligne (« dé- ») |

Les mots coupés en fin de ligne expliquent en partie le rappel plus bas de `glm-ocr` sur la page gauche.

Pour mémoire, la première version du script (`qwen3-vl` avec une sortie JSON imposée) obtenait un rappel de 60 % et une précision de 47 à 58 % : la contrainte JSON l'a poussé à créer une mention fictive en double.

Quelques passages côte à côte :

| Ce qui est écrit | qwen3-vl | glm-ocr |
|---|---|---|
| quatre (04) téléphones portables (01 iPhone 7+ 02 Spark 7 et 01 Camon 12) | Sautu (S) Téléphone portable (S) iPhone 7+ 02 Stébèle 7 et 01 Carmon 12 | quatre telephones portables (OL APhone 7+02 Spark 7 et OL Carmon 12) |
| 0136 — 09H20 | 0136 — 09H**30** | 0136 — 09H 20 |
| Suite à la mention 0123 … véhicule immobilisé | Suite à la mention **0137** … véhicule immatriculé | Suite à la mention 0123 … véhicule immobilisé |
| Mention portée à toutes fins utiles | Mention portée à l'antéfina utile | Mention portée toute fine utilise |
| … qui nous signe ici | Nous remercions Néton pour l'intérêt fourni *(inventé)* | Pour nous à reicier |

---

## 5. Transcriptions de chaque modèle

### 5.1 Référence (transcription manuelle)

**Page gauche**

| N° | Heure | Marge | Objet |
|---|---|---|---|
| *(suite)* | | *(annotation rouge peu lisible)* | Ce dernier nous déclare être victime de vol dans la nuit du 19 au 20 déc courant à son domicile. Le ou les voleurs sont entrés par la fenêtre de derrière et ont emporté quatre (04) téléphones portables (01 iPhone 7+ 02 Spark 7 et 01 Camon 12). Mention portée à toutes fins utiles. |
| 0135 | 08H15 | Vu | Mentionnons à l'heure indiquée le départ des éléments de la B.R sous la conduite de l'A/P Dianka pour une patrouille pédestre dans notre secteur de compétence. Mention portée à toutes fins utiles. |
| 0136 | 09H20 | 10 Déférés — Sur — Vu | Faisant suite à la mention N° 0130 du 20-12-2021 relative à des 10 déférés par VI sur les instructions du chef de service à nous par le chef de poste relaxons les sus-nommés : Ousso Gaye - Omar Aliou Diallo - Habib Diallo - Ismaïla Ba - El Hadji Diallo - Pape Faye - *(suite page droite)* |

**Page droite** — 20/12/2021, page 65

| N° | Heure | Marge | Objet |
|---|---|---|---|
| *(suite)* | | Instruction du chef de service | Thierno Mouma Ba - Aliou Kandé - Mouma Diallo - Lamine Bakhoum. Mention à toutes fins utiles. |
| 0137 | 10H40 | Levée immobilisation — ccu | Suite à la mention 0123 du 19 Décembre 2021 relative à un véhicule immobilisé immatriculé DL 1564 E pour transport illégal de carburant avec une provenance douteuse. Sur instruction du CDT avons levé l'immobilisation dudit véhicule après payement amendes forfaitaires de douze mille francs suivant AF n° 00805578/81, 79, 80 et 81 du 20.12.2021 à 10H30. Avons remis au chauffeur répondant au nom de Bassirou Ndoye CNI n° 106779… tel 77 208 53 66 qui nous signe ici. Mention portée à toutes fins utiles. |
| 0138 | 12H20 | Restitution de matériels de la salle de feu — Vu | Suite à la mention 0010 du 18/12/2021 relative à la sortie de la Brigade de recherches pour la patrouille nocturne, mentionnons avoir remis sur les instructions du sieur Amidou Wilane en 1977. |

### 5.2 `glm-ocr` (meilleur modèle)

Sortie brute ([`benchmark/resultats/glm_ocr_gauche.txt`](benchmark/resultats/glm_ocr_gauche.txt), [`glm_ocr_droite.txt`](benchmark/resultats/glm_ocr_droite.txt)). La version découpée par mention produite par `extract.py` est dans [`extractions/`](extractions/).

<details open>
<summary><b>Page gauche</b> — 7,6 s</summary>

```text
NUMERO
DE LA
MENTION
HEURE
OBJET
Le dernier pous d'claire est victime de
vol dans le quit du 19 au 20alle con
à Son domicile.
Le on des volets ont entier par le fini
de dérivée et ont emporté quatre
telephones portables (OL APhone 7+02
Spark 7 et OL Carmon 12)
Mention portée toute fine utilise

0135
08H 15
Vu
Mention non à Phure indiqué le dé-
sélément de la B.R. Jones la Pouille
de l'AFP dianka pour une patrouille
Apedestre dans votre secteur de compétence
Mention portée toute fine utilise

0136
09H 20
Sue
Tocignet seule p le markier N°0130
cle 20-du 22. 2021 regolise p des 10
écloques par VI pour les
instructions, die clef de seVia
a nous per le cliché de porte
pre Coron con sus Romnie-
Bois Gourn Omair Alain D'allo-
Habib D'allo- Ismail 30 -
É/ Hodji D'allo- Pape Faye-
```
</details>

<details open>
<summary><b>Page droite</b> — 8,8 s</summary>

```text
20/12 / 2021
NUMERO
DE LA
MENTION
HEURE
OBJET
65
instruction
du droit
de la
diction
Thierno Mouro Ba- Acian
Kandé - Mouro Diello-
damne Bakhrou-
Mentien a toutes Gino Ubler
0137
10H00
Suite à la mention 0123 du 19 décembre
2021 relative à un véhicule immobilisé
immatriculé du 1504E pour transport illégal
de Carburant avec une provenance britannique
Sur instruction du CMT avons laissé l'om-
plication du dit véhicule après payement
amendes les forfaitaires de douze milliores
suivant PN 00805578/29749, 800181
du 20.12.2021 à 10H30.
Avis ne limite au chéffeur responsable du
nom de Bassinou Nolaye Pcn 1047791
tel 772085366 Pour nous à reicier
Mention poste fins illite
0138
12H20
Restitution
do matériels
do fallé de
Ten
Suite à la mention 0010 du 18/12/2021
relative à la Surtri de la Brigade
de cuche du frou le porto ville moctume,
mentionnons avoir permis sur le instituer
du sein Amidon Wilane en 1977.
```
</details>

### 5.3 `qwen3-vl:4b-instruct`

Sortie brute ([`benchmark/resultats/qwen3_vl_4b_instruct_gauche.txt`](benchmark/resultats/qwen3_vl_4b_instruct_gauche.txt), [`qwen3_vl_4b_instruct_droite.txt`](benchmark/resultats/qwen3_vl_4b_instruct_droite.txt)).

<details open>
<summary><b>Page gauche</b> — 12,4 s</summary>

```text
NUMERO DE LA MENTION | HEURE | OBJET
---|---|---
[illisible] | [illisible] | Le dernier mois déclare être victime du vol dans le nuit du 19 au 20 avec son domicile. Les volumes sont entrés par la fenêtre de derrière et ont emporté Sautu (S) Téléphone portable (S) iPhone 7+ 02 Stébèle 7 et 01 Carmon 12). Mention portée à l'antéfina utile.
0135 | 08H15 | Mention non répétée. L'heure indiquée de départ des éléments de la B.R. Jules Baud de l'AP Sianka pour une patrouille d'opération dans notre secteur de compétence. Mention portée à l'antéfina utile.
0136 | 09H30 | Tournant suite à la mort de N°0130 chez 20-42-222. Récolte par des 10 d'opérations pour V1 par les instructions du chef de police et non par le chef de poste. Précaution par les commissaires. Bédo Gouna, Omaï Aïdat Dallou, Habib Dallou, Ismaili Ba, F. Hafsi, Océille, Pape Faye.
0137 | 09H45 | [illisible]
```
</details>

<details open>
<summary><b>Page droite</b> — 13,3 s</summary>

```text
0137 10H40
Résolution du chef de service
Thierry Moumou Ba-Acine
Kandé-Moussa Diello.
dame Bakhrou.
Mentien a toutes Bins Akla.

0138 12H20
Résolution du chef de service
Suite à la mention 0137 du 19 décembre 2021 relative à un véhicule immatriculé DL 1584E pour transport illégal de Carburant avec une mouvance d'attente.
Sur instruction du CDT avons levée l'immatriculation du dit véhicule après paiement
amendes forfaitaires de douze milliers
suivant PV no 00805578/21-79, 80281
du 20.12.2021 à 10H30.
D'après les avis du chauffeur répondent au
nom de Bastion Ndaye ?no 106:799
tel 772085366. Nous remercions
Néton pour l'intérêt fourni.

0139 12H20
Résolution du chef de service
Suite à la mention 0010 du 18/12/2021
Relative à la sortie de la Brigade
du caché des armes de la patrouille nocturne.
Mentionnons avoir mis sur la instruction
de la suite d'Amidou Wilouane en 1977.
```
</details>

---

## 6. Problèmes rencontrés et optimisations

### Historique des temps (double page complète)

| Étape | Configuration | Temps | Constat |
|---|---|---|---|
| 1 | `qwen3-vl`, contexte 8192, réglages par défaut | 269 s | Ollama ne place que 34 couches sur 37 sur le GPU (24 % CPU / 76 % GPU) |
| 2 | contexte réduit à 4096 | — | JSON tronqué : image (~2 000 tokens) + consigne + réponse dépassent 4096 |
| 3 | contexte 6144 + `num_gpu: 99` (toutes les couches sur le GPU) | 47 s | 100 % GPU |
| 4 | `glm-ocr`, encodeur d'image sur le CPU | 376 s | lecture de l'image : 137 à 229 s par page |
| 5 | `glm-ocr` + vérification de l'encodeur (`load_on_gpu`) | **21 s** | version actuelle |

### Piège principal : l'encodeur d'image qui tombe sur le CPU

Le modèle de texte peut être à « 100 % GPU » dans `ollama ps` alors que **l'encodeur d'image tourne sur le CPU**. La lecture d'une page passe alors de 3 s à plus de 2 minutes :

| Modèle | Encodeur d'image | Lecture d'une demi-page |
|---|---|---|
| qwen3-vl | CPU | 63 – 137 s |
| qwen3-vl | GPU | 2,8 – 3,5 s |
| glm-ocr | CPU | 129 – 229 s |
| glm-ocr | GPU | 2,9 – 4,0 s |

**Cause**, visible dans `%LOCALAPPDATA%\Ollama\server.log` : avant chaque chargement, Ollama mesure la VRAM libre. Sur cette machine la mesure échoue (`llama-server GPU discovery watchdog timed out` puis `unable to refresh free memory, using old values`). Ollama se fie alors à une ancienne valeur ; si elle est basse (par ex. 0,9 Go), il désactive l'encodeur sur GPU (`disabling multimodal projector offload reason=limited-vram`, option `--no-mmproj-offload`). Libérer la VRAM ne suffit donc pas.

**Solution** dans [`extract.py`](extract.py) (`load_on_gpu`) : après le chargement, une petite image test (448 × 448) est envoyée. Sur GPU l'encodeur traite 180 à 600 tokens/s, sur CPU 11 à 34 tokens/s. En dessous de 80 tokens/s, le script décharge et recharge le modèle (3 essais au maximum). En pratique, un seul rechargement suffit :

```text
Modèle : glm-ocr
  encodeur d'image sur CPU (34 tokens/s) -> rechargement 1/3
  encodeur d'image sur GPU (189 tokens/s)
  WhatsApp Image 2026-11-25 at 20.38.11.jpeg page 1 : 7.5s (lecture image 2.9s)
  WhatsApp Image 2026-11-25 at 20.38.11.jpeg page 2 : 8.7s (lecture image 2.9s)
```

Les options (`num_ctx`, `num_gpu`) sont identiques pour l'image test et pour les pages, sinon Ollama rechargerait le modèle et le problème pourrait revenir.

### Téléchargement de `glm-ocr` bloqué

`ollama pull glm-ocr` restait bloqué (0 octet reçu, puis arrêts après quelques Mo, débit ≈ 0,3 Mo/s). Ollama reprend un téléchargement interrompu : relancer la commande à chaque blocage a permis de récupérer les 869 Mo en une trentaine de minutes.

---

## 7. Utilisation

### Installation

```bash
pip install -r requirements.txt
ollama pull glm-ocr                 # meilleur modèle (2,7 Go)
ollama pull qwen3-vl:4b-instruct    # optionnel, pour la comparaison (3,3 Go)
```

### Extraction

Placer les photos dans le dossier du projet, puis :

```bash
python extract.py                   # toutes les images du dossier
python extract.py page1.jpg page2.jpg
```

Résultats dans `extractions/` :

- `<image>.json` : pour chaque page, la date, le n° de page, la liste des mentions (`numero`, `heure`, `texte`) et le texte brut ;
- `<image>.md` : la même chose en Markdown lisible.

Avec `glm-ocr`, les annotations de marge (« Vu », « Restitution… ») restent dans le texte de la mention : le modèle les lit dans l'ordre de la page sans distinguer les colonnes.

### Réglages (variables d'environnement)

| Variable | Défaut | Rôle |
|---|---|---|
| `OCR_MODEL` | `glm-ocr` | modèle Ollama ; avec `qwen3-vl:4b-instruct` la sortie est un JSON imposé par schéma (`numero`, `heure`, `texte`, `annotations`) |
| `OCR_CTX` | 8192 (glm) / 6144 (qwen) | taille du contexte ; trop petit, la réponse est tronquée ; trop grand, le modèle déborde des 6 Go |
| `OCR_GPU_LAYERS` | 99 | force toutes les couches sur le GPU |
| `OCR_SIDE` | 1600 | taille (px) du plus grand côté de chaque page envoyée au modèle |
| `OLLAMA_URL` | `http://localhost:11434` | adresse du serveur Ollama |

Exemple sous PowerShell :

```powershell
$env:OCR_MODEL = "qwen3-vl:4b-instruct"; python extract.py
```

### Refaire la comparaison

```bash
python benchmark/compare.py glm-ocr
python benchmark/compare.py qwen3-vl:4b-instruct
```

Les transcriptions et les scores sont écrits dans `benchmark/resultats/`. Le script attend la photo `WhatsApp Image 2026-11-25 at 20.38.11.jpeg` à la racine du projet ; elle n'est pas publiée dans ce dépôt.

---

## 8. Limites et pistes

- **Qualité de la photo** : c'est la principale limite. Une photo WhatsApp de 1280 px pour deux pages laisse environ 27 px par ligne manuscrite. Des photos à pleine résolution, une page par photo, bien éclairées et à plat, devraient nettement améliorer les résultats.
- **Relecture obligatoire** : environ 40 % des mots restent faux. Les noms propres, numéros de pièce d'identité et plaques sont les moins fiables.
- **Référence imparfaite** : la transcription de référence est elle-même une lecture humaine de la même photo ; quelques noms y sont incertains.
- **Marge et colonnes** : `glm-ocr` ne sépare pas les annotations de marge du texte de la mention.
- **Pistes** :
  - découper chaque mention en bande horizontale avant l'OCR (plus de pixels par mot) ;
  - essayer `qwen3-vl:8b` : il ne tient pas entièrement dans 6 Go, donc il sera plus lent ;
  - une post-correction par un modèle de texte : attention, elle risque d'inventer comme `qwen3-vl`.

---

## 9. Structure du dépôt

```text
.
├── README.md                     ce rapport
├── extract.py                    extraction (glm-ocr par défaut), sorties JSON + Markdown
├── requirements.txt
├── extractions/                  résultat de extract.py avec glm-ocr
└── benchmark/
    ├── compare.py                comparaison d'un modèle contre la référence
    ├── reference.json            transcription manuelle + 29 repères
    └── resultats/                transcriptions brutes et scores de chaque modèle
```

Les photos (`*.jpg`, `*.jpeg`, `*.png`…) placées à la racine pour l'extraction ne sont pas versionnées (voir `.gitignore`).
