---
project: AutoEIT
layout: default
logo: AutoEIT.png
description: |
   AutoEIT is an applied machine learning project focused on automating the scoring of the Elicited Imitation Task (EIT), a research tool used to measure global language proficiency. The EIT is widely respected and is available for free in several languages, but the current workflow requires manual audio transcription and human scoring—slow, tedious, and error‑prone. This project aims to build an end-to-end system that will: Process raw audio files, perform accurate voice-to-text transcription, and automatically evaluate responses using a standardized scoring rubric.

---

## Metrics Dashboard and Results Visualization

AutoEIT can include a dashboard to visualize transcription and scoring results:

1. **Score Distribution**
   - Visualize the range of scores for all participants
   - Identify patterns in common errors

2. **Transcription Accuracy Metrics**
   - Word Error Rate (WER) per audio file
   - Highlight low-accuracy transcriptions for review

3. **Real-Time Updates**
   - Display ongoing processing results as they are scored
   - Color-coded alerts for scores below threshold

4. **Export Options**
   - Export results as CSV, JSON, or PDF
   - Shareable reports for researchers or instructors

---

{% include gsoc_project.ext %}
