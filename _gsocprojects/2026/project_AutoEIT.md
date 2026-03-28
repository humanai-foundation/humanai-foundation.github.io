---
project: AutoEIT
layout: default
logo: AutoEIT.png
description: |
   AutoEIT is an applied machine learning project focused on automating the scoring of the Elicited Imitation Task (EIT), a research tool used to measure global language proficiency. The EIT is widely respected and is available for free in several languages, but the current workflow requires manual audio transcription and human scoring—slow, tedious, and error‑prone. This project aims to build an end‑to‑end system that will: Process raw audio files, perform accurate voice‑to‑text transcription, and automatically evaluate responses using a standardized scoring rubric.

---

## Error Handling and Logging

To ensure reliable operation and easy debugging, AutoEIT implements structured error handling and logging:

1. **Error Handling**
   - Handles missing or corrupted audio files
   - Detects transcription failures and retries automatically
   - Validates scoring inputs to avoid crashes

2. **Logging**
   - Logs all audio inputs, preprocessing steps, transcription outputs, and scores
   - Provides timestamps for each step
   - Stores logs in `/logs/` directory for audit and review

3. **Notifications**
   - Optional email or console alerts for critical errors
   - Helps developers quickly identify and resolve issues

---

{% include gsoc_project.ext %}
