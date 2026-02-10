---
title: Automated scoring system for elicited imitation task responses
layout: gsoc_proposal
project: AutoEIT
year: 2026
organization:
  - Alabama
  - NIU
---

## Description

This proposal focuses on creating a reliable, standardized, fully automated scoring system for the Spanish Elicited Imitation Task (EIT), using the transcriptions generated from learner audio. The EIT scoring process is currently labor-intensive and fully dependent on trained human raters. Although large language models can assist with scoring, they often produce inconsistent results, awarding different scores to the exact same sentence across sessions or prompts. This lack of standardization makes automated scoring using existing tools unsuitable for research purposes.

The goal of this project is to eliminate scoring variability by designing a system that applies the EIT scoring rubric in a consistent, rule-driven, and reproducible way. This tool will support research involving second/additional language learners by providing high-quality, standardized, automated scoring for large datasets. By developing both the core scoring logic and a web-based interface, the project delivers a practical tool that will significantly streamline research on second and additional language learning.

## Duration

Total project length: 175 hours

## Task ideas

- Develop a scoring engine: rule-based, machine-learning-supported, rubric application system.
- Test and validate: Compare automated scores with those of experienced human raters. Quantify consistency and identify where rubric interpretation differs.
- Optimize accuracy and consistency: Revise scoring engine to ensure transparent, interpretable, replicable scoring decisions.
- Eventual goal: Build a user-facing web-interface: Allow researchers to upload transcriptions for automatic scoring.

## Expected results

- Consistent scoring engine that produces 90% agreement with experienced human raters (measured at the sentence level, comparing automated vs human rubric application); less than 10-point different in total EIT scores (across 120-point scale)
- Fully documented scoring logic to ensure transparency, replicability, and long-term maintenance of scoring engine.

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
