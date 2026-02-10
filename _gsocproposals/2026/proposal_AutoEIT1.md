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

<!--
## Test
Please use [this link](https://bama365-my.sharepoint.com/:w:/g/personal/xgranja_ua_edu/EeSz8D6iYPxHhzfQD3GGzsYBARpsSkbEDZWzoQH7hIH4lg?e=gMOaR4&xsdata=MDV8MDJ8ZXVzYWlAdWEuZWR1fDIzZDVmYjNmYjYzYjQ0YzljYTU0MDhkZDU3ZjE1MDZhfDJhMDA3MjhlZjBkMDQwYjRhNGU4Y2U0MzNmM2ZiY2E3fDB8MHw2Mzg3NjM0MTYxNDQxNDUxMTR8VW5rbm93bnxUV0ZwYkdac2IzZDhleUpGYlhCMGVVMWhjR2tpT25SeWRXVXNJbFlpT2lJd0xqQXVNREF3TUNJc0lsQWlPaUpYYVc0ek1pSXNJa0ZPSWpvaVRXRnBiQ0lzSWxkVUlqb3lmUT09fDB8fHw%3d&sdata=RUVjT2J4U2N1cjlyNzl2YXd4RkVTV3pkZ1UvWkJhYWVOSjltVVJwYkhIOD0%3d) to access the test for this project.-->

## Mentors

- [Mandy Faretta-Stutenberg](mailto:human-ai@cern.ch) (Northern Illinois University)
- [Xabier Granja](mailto:human-ai@cern.ch) (University of Alabama)

Please **DO NOT** contact mentors directly by email. Instead, please email [human-ai@cern.ch](mailto:human-ai@cern.ch) with Project Title and **include your CV** and **test results**. The mentors will then get in touch with you.

<!-- ## Links
  * [Paper 1](https://arxiv.org/abs/1807.11916)
  * [Paper 2](https://arxiv.org/abs/1902.08276) -->
