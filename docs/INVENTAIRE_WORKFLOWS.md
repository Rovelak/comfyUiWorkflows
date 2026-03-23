# Inventaire des workflows ComfyUI

## Vue d'ensemble

Tous les workflows JSON du dossier ont ete renommes avec une convention explicite:

`<famille>__<variante-modele>__<usage>__<format>__vNN.json`

Formats detectes:

- `UI`: JSON avec cle `nodes`, destine a l'editeur graphique ComfyUI
- `API`: JSON avec clefs numeriques et `class_type`, destine a l'API ou a l'automatisation

## Arborescence actuelle

Les fichiers ont ete ranges dans les sous-dossiers suivants:

```text
comfyUiWorkflows/
  docs/
    INVENTAIRE_WORKFLOWS.md
  scripts/
    batch-img.py
    batch-video.py
  ghibli/
    simple/
    wd14/
    controlnet/
  flux/
    base/
    lora/
  video/
    ltx23/
  archive/
    duplicates/
    experiments/
```

Regle appliquee:

- workflows actifs dans leurs familles respectives
- doublons exacts dans `archive/duplicates`
- workflow de test isole dans `archive/experiments`
- scripts Python dans `scripts`
- documentation dans `docs`

## Noms actuels par famille

### Ghibli simple / img2img

- `ghibli__illustration-juaner__img2img__ui__v01.json`
- `ghibli__illustration-juaner__img2img__ui__condzero__v02.json`
- `ghibli__illustrious__img2img__api__v01.json`
- `ghibli__illustrious__img2img__ui__v01.json`

### Ghibli + WD14

- `ghibli__wd14-basic__img2img__api__v01.json`
- `ghibli__wd14-basic__img2img__ui__v01.json`
- `ghibli__wd14-basic__img2img__ui__v02.json`
- `ghibli__wd14-combined__img2img__api__v01.json`
- `ghibli__wd14-combined__img2img__ui__v01.json`

### Ghibli + ControlNet

- `ghibli__illustrious-controlnet-canny__img2img__api__v01.json`
- `ghibli__illustrious-controlnet-canny__img2img__ui__v01.json`
- `ghibli__illustrious-controlnet-canny__img2img__ui__v02.json`
- `ghibli__illustrious-controlnet-canny__img2img__ui__v02b.json`

### Flux

- `flux__base__img2img__ui__v01.json`
- `flux__base__img2img__ui__v01__duplicate.json`
- `flux__ghibli-lora__img2img__ui__v01.json`
- `flux__ghibli-lora__img2img__ui__v01__duplicate.json`

### Video LTX 2.3

- `ltx23__image-to-video__api__experimental__v01.json`
- `ltx23__image-to-video__api__stable-fast-noaudio__v01.json`

### Test technique isole

- `zimage-turbo__pencil-sketch-lora__txt2img__ui__techpractice__v01.json`

## Classement fonctionnel

### 1. Ghibli simple / img2img

But: styliser une image source en rendu Ghibli sans module de controle additionnel.

Fichiers:

- `ghibli__illustration-juaner__img2img__ui__v01.json`
- `ghibli__illustration-juaner__img2img__ui__condzero__v02.json`
- `ghibli__illustrious__img2img__api__v01.json`
- `ghibli__illustrious__img2img__ui__v01.json`

Details:

- `ghibli__illustration-juaner__img2img__ui__v01.json`: variante UI avec `UNETLoader` + `DualCLIPLoader`
- `ghibli__illustration-juaner__img2img__ui__condzero__v02.json`: meme famille avec `ConditioningZeroOut`
- `ghibli__illustrious__img2img__api__v01.json`: version API minimaliste basee sur un checkpoint Illustrious/Ghibli
- `ghibli__illustrious__img2img__ui__v01.json`: pendant UI du workflow precedent

### 2. Ghibli + WD14 auto-tagging

But: analyser l'image source avec WD14 pour injecter des tags automatiques dans le prompt.

Fichiers:

- `ghibli__wd14-basic__img2img__api__v01.json`
- `ghibli__wd14-basic__img2img__ui__v01.json`
- `ghibli__wd14-basic__img2img__ui__v02.json`
- `ghibli__wd14-combined__img2img__api__v01.json`
- `ghibli__wd14-combined__img2img__ui__v01.json`

Recommendation:

- garder en reference UI `ghibli__wd14-basic__img2img__ui__v02.json`
- garder en reference API `ghibli__wd14-combined__img2img__api__v01.json`

### 3. Ghibli + ControlNet

But: mieux conserver la composition de l'image source via Canny + ControlNet.

Fichiers:

- `ghibli__illustrious-controlnet-canny__img2img__api__v01.json`
- `ghibli__illustrious-controlnet-canny__img2img__ui__v01.json`
- `ghibli__illustrious-controlnet-canny__img2img__ui__v02.json`
- `ghibli__illustrious-controlnet-canny__img2img__ui__v02b.json`

Etat du groupe:

- les variantes partagent la meme logique generale
- `v02` et `v02b` restent proches mais pas assez documentees pour fusionner sans test

### 4. Flux / img2img personnel

But: pipeline img2img base sur Flux.

Fichiers:

- `flux__base__img2img__ui__v01.json`
- `flux__base__img2img__ui__v01__duplicate.json`

Etat du groupe:

- le fichier suffixe `__duplicate` est un doublon exact

### 5. Flux + LoRA

But: pipeline Flux avec LoRA Ghibli dediee.

Fichiers:

- `flux__ghibli-lora__img2img__ui__v01.json`
- `flux__ghibli-lora__img2img__ui__v01__duplicate.json`

Etat du groupe:

- le fichier suffixe `__duplicate` est un doublon exact

### 6. Video / animation LTX 2.3

But: animer une image fixe en video courte.

Fichiers:

- `ltx23__image-to-video__api__experimental__v01.json`
- `ltx23__image-to-video__api__stable-fast-noaudio__v01.json`

Recommendation:

- garder `ltx23__image-to-video__api__stable-fast-noaudio__v01.json` comme version de production

### 7. Test technique isole

Fichier:

- `zimage-turbo__pencil-sketch-lora__txt2img__ui__techpractice__v01.json`

Interpretation:

- workflow de test ou de tutoriel, separe des workflows Ghibli principaux

## Doublons exacts detectes

- `flux__base__img2img__ui__v01.json` = `flux__base__img2img__ui__v01__duplicate.json`
- `flux__ghibli-lora__img2img__ui__v01.json` = `flux__ghibli-lora__img2img__ui__v01__duplicate.json`

## Scripts Python du dossier

Ces fichiers ne sont pas des workflows, mais des outils d'automatisation autour de ComfyUI. Ils se trouvent maintenant dans `scripts/`:

- `batch-img.py`: script batch image, configure par defaut pour `ghibli__wd14-combined__img2img__api__v01.json`
- `batch-video.py`: script batch video, configure pour `ltx23__image-to-video__api__stable-fast-noaudio__v01.json`

## Reorganisation conseillee du dossier

Structure cible conseillee:

```text
comfyUiWorkflows/
  docs/
    INVENTAIRE_WORKFLOWS.md
  ghibli/
    simple/
    wd14/
    controlnet/
  flux/
    base/
    lora/
  video/
    ltx23/
  archive/
    duplicates/
    experiments/
  scripts/
```

## Tri minimal recommande

Base a garder en priorite:

- `ghibli__wd14-basic__img2img__ui__v02.json`
- `ghibli__wd14-combined__img2img__api__v01.json`
- `ghibli__illustrious-controlnet-canny__img2img__ui__v01.json`
- `flux__base__img2img__ui__v01.json`
- `flux__ghibli-lora__img2img__ui__v01.json`
- `ltx23__image-to-video__api__stable-fast-noaudio__v01.json`
- `zimage-turbo__pencil-sketch-lora__txt2img__ui__techpractice__v01.json`

A archiver si tu veux nettoyer vite:

- `flux__base__img2img__ui__v01__duplicate.json`
- `flux__ghibli-lora__img2img__ui__v01__duplicate.json`

## Conclusion

Le dossier est maintenant coherent au niveau des noms: chaque workflow indique sa famille, sa variante, son usage, son format et sa version.
