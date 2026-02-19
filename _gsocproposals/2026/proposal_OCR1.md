---
title: Automating text recognition and transcription of historical documents with weighted convolutional - recurrent architectures and LLM integration
layout: gsoc_proposal
project: RenAIssance
year: 2026
organization:
  - Alabama
  - Delhi
  - Knoxville
  - Yale
---

## Description

Transliteration of text from centuries-old works represents a research area that is underserved by current tools, such as Adobe Acrobat’s OCR. While these resources can perform text recognition from clearly printed modern sources, they are incapable of extracting textual data from early forms of print, much less manuscripts. This project will focus on the application of hybrid end-to-end models based on weighted convolutional - recurrent architectures (CNN-RNN) and LLM models as a late-stage step to recognize text in Spanish printed sources from the seventeenth century.

## Duration

Total project length: 175 hours

## Task ideas

- Creation of a hybrid end-to-end model based on convolutional - recurrent architectures (CNN-RNN) capable of performing text recognition
- Implement weighted learning techniques to improve the model’s ability to recognize rare letterforms, diacritics, or symbols specific to renaissance printed Spanish sources.
- Introduce constrained beam search decoding with a renaissance Spanish lexicon to reduce hallucinated outputs and enhance word level accuracy.
- Integrate LLM models such as Gemini3 as an integral step for transcription accuracy

## Expected results

- Machine learning models will be trained to perform text recognition of non-standard printed text
- AI should be able to extract text with at least 90% accuracy

## Requirements

Python and some previous experience in Machine Learning.

## Difficulty level

Medium

## Tests

Please use [this link](/assets/GSoC%202026%20tests.pdf) to access the test for this project.

## Mentors

- [Sergei Gleyzer](mailto:human-ai@cern.ch) (University of Alabama)
- [Xabier Granja](mailto:human-ai@cern.ch) (University of Alabama)
- [Nicholas Jones](mailto:human-ai@cern.ch) (Yale University)
- [Harrison Meadows](mailto:human-ai@cern.ch) (University of Tennessee Knoxville)
- [Emanuele Usai](mailto:human-ai@cern.ch) (University of Alabama)

Please **DO NOT** contact mentors directly by email. Instead, please email [human-ai@cern.ch](mailto:human-ai@cern.ch) with Project Title and **include your CV** and **test results**. The mentors will then get in touch with you.

<!-- ## Links
  * [Paper 1](https://arxiv.org/abs/1807.11916)
  * [Paper 2](https://arxiv.org/abs/1902.08276) -->
