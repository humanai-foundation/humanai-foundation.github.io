---
title:  Text recognition with transformer models
layout: gsoc_proposal
project: RenAIssance
year: 2025
organization:
  - Alabama
  - Delhi

---

## Description

Transcription of text from centuries-old works represents a research area that is underserved by current tools, such as Adobe Acrobat’s OCR. While these resources can perform text recognition from clearly printed modern sources, they are incapable of extracting textual data from early forms of print, much less manuscripts. This project will focus on the application of hybrid end-to-end models based on transformers (e.g. VIT-RNN or CNN-TF or VIT-TF) to recognize text in Spanish printed sources from the seventeenth century. Transformer-based models have been fine-tuned to improve transcription accuracy, particularly for degraded and complex historical texts. Training on a diverse dataset that combines expert transcriptions and synthetic data has enabled better generalization across various typographical styles.

For GSoC 2025, this project aims to expand the dataset, to help the model finetune to handle handwritten documents as well. The goal is to increase our fine-tuning on larger datasets incorporating diverse typographical styles both printed and handwritten. 

## Duration

Total project length: 175 hours

## Task ideas
 * Creation of an hybrid end-to-end model based on transformers (e.g. VIT-RNN or CNN-TF or VIT-TF) capable of performing text recognition.
 * Implement Language Modeling & Contextual Understanding for post-processing, allowing for contextual corrections based on 17th-century grammar to further enhancing the OCR accuracy.
 * Develop and deploy a web or mobile-based annotation tool for historians, researchers, and institutions to validate and refine OCR outputs.

## Expected results
 * Machine learning models will be trained to perform text recognition of non-standard printed text
 * AI should be able to extract text with at least 80% accuracy



## Requirements
Python and some previous experience in Machine Learning.

## Difficulty level
Advanced

## Test
Please use [this link](https://bama365-my.sharepoint.com/:w:/g/personal/xgranja_ua_edu/EeSz8D6iYPxHhzfQD3GGzsYBARpsSkbEDZWzoQH7hIH4lg?e=gMOaR4&xsdata=MDV8MDJ8ZXVzYWlAdWEuZWR1fDIzZDVmYjNmYjYzYjQ0YzljYTU0MDhkZDU3ZjE1MDZhfDJhMDA3MjhlZjBkMDQwYjRhNGU4Y2U0MzNmM2ZiY2E3fDB8MHw2Mzg3NjM0MTYxNDQxNDUxMTR8VW5rbm93bnxUV0ZwYkdac2IzZDhleUpGYlhCMGVVMWhjR2tpT25SeWRXVXNJbFlpT2lJd0xqQXVNREF3TUNJc0lsQWlPaUpYYVc0ek1pSXNJa0ZPSWpvaVRXRnBiQ0lzSWxkVUlqb3lmUT09fDB8fHw%3d&sdata=RUVjT2J4U2N1cjlyNzl2YXd4RkVTV3pkZ1UvWkJhYWVOSjltVVJwYkhIOD0%3d) to access the test for this project.

## Mentors
  * [Sergei Gleyzer](mailto:human-ai@cern.ch) (University of Alabama)
  * [Xabier Granja](mailto:human-ai@cern.ch) (University of Alabama)
  * [Emanuele Usai](mailto:human-ai@cern.ch) (University of Alabama)



Please **DO NOT** contact mentors directly by email. Instead, please email [human-ai@cern.ch](mailto:human-ai@cern.ch) with Project Title and **include your CV** and **test results**. The mentors will then get in touch with you.



<!-- ## Links
  * [Paper 1](https://arxiv.org/abs/1807.11916)
  * [Paper 2](https://arxiv.org/abs/1902.08276) -->
