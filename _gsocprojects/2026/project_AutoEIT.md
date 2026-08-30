---
project: AutoEIT
layout: default
logo: AutoEIT.png
description: |
   AutoEIT is an applied machine learning project focused on automating the scoring of the Elicited Imitation Task (EIT), a research tool used to measure global language proficiency.

   The EIT is widely used and available in multiple languages. However, the current workflow relies on manual audio transcription and human scoring, which is slow, tedious, and prone to errors.

   This project aims to develop an end-to-end system with the following features:

   - Process raw audio files
   - Perform accurate voice-to-text transcription
   - Automatically evaluate responses
   - Apply a standardized scoring rubric

---

## System Workflow

The proposed AutoEIT system can follow this workflow:

1. **Audio Input**
   - User provides recorded speech response

2. **Preprocessing**
   - Noise reduction and audio normalization

3. **Speech-to-Text Conversion**
   - Convert audio into text using a transcription model

4. **Text Analysis**
   - Compare transcribed text with expected response

5. **Scoring Mechanism**
   - Apply EIT scoring rules to evaluate correctness

6. **Output**
   - Generate score and feedback

---

## Future Improvements

- Support multiple languages  
- Improve accuracy using advanced models (e.g., Whisper)  
- Enable real-time transcription and scoring  

---

{% include gsoc_project.ext %}
