---
title: Audio-to-text transcription for second/additional language learner data
layout: gsoc_proposal
project: AutoEIT
year: 2026
organization:
  - Alabama
  - NIU
---

## Description

This project aims to develop a robust system for converting audio files from second and additional language learners who have completed a Spanish Elicited Imitation Task (EIT) into accurate, usable text transcriptions for linguistic analysis. The EIT is a widely used sentence‑repetition task that provides valuable data on global language proficiency, but its research potential is currently limited by the time‑intensive process of manually transcribing learner responses.

While commercial voice‑to‑text tools can reliably transcribe native speaker production, they struggle when processing speech from non‑native speakers with diverse language backgrounds, varying accents, lower/intermediate proficiency levels, and inconsistent speech patterns. These systems are typically trained on large datasets of native or near-native speech, making them poorly optimized for learner language that may feature transfer effects, phonological variation, disfluencies, or partially accurate sentence repetitions.

The primary goal is to create a scalable, high‑accuracy transcription pipeline that enables researchers to efficiently evaluate EIT responses, even when dealing with large datasets or highly diverse learner populations. This project offers a unique opportunity to push the limits of speech-to-text technology by working with one of the most challenging types of input data: multilingual learner speech.

## Duration

Total project length: 175 hours

## Task ideas

- Preprocessing raw audio for clarity, segmentation, and noise reduction
- Customizing or fine-tuning existing speech recognition models to handle variable proficiency levels
- Developing post-processing pipelines to correct predictable transcription errors common in learner language
- Outputting accurate transcriptions suitable for automatic or human scoring

## Expected results

- Convert audio recordings into written transcriptions with 90% agreement with experienced human transcribers

## Requirements

Python, Pytorch or Tensorflow, and some previous experience in Machine Learning.

## Difficulty level

Medium

## Tests

Please use [this link](/assets/GSoC%202026%20AutoEIT%20Tests.pdf) to access the test for this project.

## Mentors

- [Mandy Faretta-Stutenberg](mailto:human-ai@cern.ch) (Northern Illinois University)
- [Xabier Granja](mailto:human-ai@cern.ch) (University of Alabama)

Please **DO NOT** contact mentors directly by email. Instead, please email [human-ai@cern.ch](mailto:human-ai@cern.ch) with Project Title and **include your CV** and **test results**. The mentors will then get in touch with you.

<!-- ## Links
  * [Paper 1](https://arxiv.org/abs/1807.11916)
  * [Paper 2](https://arxiv.org/abs/1902.08276) -->
