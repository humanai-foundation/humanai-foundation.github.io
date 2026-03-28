---
project: AutoEIT
layout: default
logo: AutoEIT.png
description: |
   AutoEIT is an applied machine learning project focused on automating the scoring of the Elicited Imitation Task (EIT), a research tool used to measure global language proficiency. The EIT is widely respected and is available for free in several languages, but the current workflow requires manual audio transcription and human scoring—slow, tedious, and error‑prone. This project aims to build an end‑to‑end system that will: Process raw audio files, perform accurate voice‑to‑text transcription, and automatically evaluate responses using a standardized scoring rubric.

---
## Evaluation Strategy

To ensure the effectiveness of the AutoEIT system, the following evaluation approach can be used:

- **Transcription Accuracy**
  - Measure Word Error Rate (WER) to evaluate speech-to-text performance

- **Scoring Accuracy**
  - Compare automated scores with human-evaluated scores

- **Robustness**
  - Test system performance on noisy and diverse audio inputs

- **Consistency**
  - Ensure the system produces stable and repeatable results across multiple runs
 
    ---
    ## Dataset Considerations

- Use multilingual speech datasets
- Include diverse accents and speaking speeds
- Ensure balanced dataset for fair evaluation

{% include gsoc_project.ext %}
