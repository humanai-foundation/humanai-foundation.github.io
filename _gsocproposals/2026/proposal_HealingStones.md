---
title: Healing Stones - Reconstructing digitized cultural heritage artifacts with artificial intelligence
layout: gsoc_proposal
project: HealingStones
year: 2026
organization:
  - Alabama
---

## Description

All ancient architectural features, monuments, and smaller artifacts inevitably break. The breaks may be accidental or deliberate, stemming from ritual practices, political iconoclasm, or actions of modern looters and collectors. Ancient Maya would ‘de-activate’ building facades, carved monuments, and ceramic vessels by breaking and caching them in multiple locations. Other Maya monuments were broken as acts of political and ritual violence. Political and religious upheaval was responsible for breaking and dispersing of European medieval sculptures including those of the Notre Dame cathedral in Paris. Identifying and matching broken pieces is often an essential and highly labor-intensive component of research and conservation projects in Archaeology and Art History. The challenge becomes especially daunting when fragments of the same sculpture end up dispersed across various private and institutional repositories. We believe that new methods of digital documentation combined with machine learning can greatly enhance our ability to reconstruct ancient monuments and artifacts.

This is the second phase of the project. It builds on the outcomes of the first phase that identified several promising approaches to identifying and refitting matching sculpture fragments. At the same time, the first phase highlighted several challenging tasks such as finding matches between segments of vastly different sizes, classifying fragments surfaces, and dealing with match imperfections caused by documentation errors and surface erosion. We hope to resolve these issues while working with the same test dataset: fragments of a single ancient Maya monument that was deliberately broken and cut by modern looters. The dataset offers a wide range of match types, from closely fitting breaks to fragments sharing surface topology yet separated by gaps.

## Duration

Total project length: 175 hours

## Task ideas

- Data reduction / re-topologizing
- Surface classification and detection of break surfaces.
- Geometric feature extraction
- Finding matches between break surfaces
- Aligning matched break surfaces accounting for data errors and gaps

## Expected results

- Incorporate machine learning into the initial stages of the work pipeline (surface classification, fragment relationship prediction)
- Detect and match breaks with at least 80% accuracy
- Detect and match discontinuous areas of shared topology with at least 80% accuracy
- Orient the matching fragments without overfitting

## Requirements

- Python and some previous experience in Machine Learning.
- Ability work with 3D scan data (e.g. .PLY and .OBJ files)

## Project difficulty level

Medium

## Links

- [Notre Dame in color](https://adhc1.ua.edu/notre_dame_in_color/)
- [Visual Documentation Lab](https://sites.ua.edu/atokovinine/3d-lab/)

## Mentors

- [Jennifer Feltman](https://art.ua.edu/people/jennifer-m-feltman/) (University of Alabama)
- [Alexandre Tokovinine](https://anthropology.ua.edu/people/alexandre-tokovinine/) (University of Alabama)
- [Emanuele Usai](mailto:human-ai@cern.ch) (University of Alabama)
- [Sergei Gleyzer](mailto:human-ai@cern.ch) (University of Alabama)

<!--
## Test
Please use [this link](https://docs.google.com/document/d/e/2PACX-1vRgBwVAVn_XZkecZ_axHryevSbVue_nLG6uVJK5nY_l69JdzqENoYCfAo20kR361-aPDtsO640X9vN9/pub) to access the test for this project.
-->

Please **DO NOT** contact mentors directly by email. Instead, please email [human-ai@cern.ch](mailto:human-ai@cern.ch) with Project Title and **include your CV** and **test results**. The mentors will then get in touch with you.
