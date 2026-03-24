---
title: Machine Learning for Narrative Voice Classification and Retrieval in Interactive Storytelling Systems
layout: gsoc_proposal
project: TableTalk
year: 2026
organization:
  - Alabama
---

## Description

Tabletop role-playing games and narrative board games rely heavily on storytelling, atmosphere, and narration to create immersive player experiences. In many games, the person leading the story must deliver long passages of descriptive text, control pacing, and establish emotional tone while guiding players through unfolding narrative events. Although digital tools exist to assist with gameplay mechanics, very few systems integrate recorded voice performance, narrative scripting, and environmental effects into a unified storytelling platform. As a result, most tabletop narration remains dependent on a single human storyteller rather than a system that can dynamically support narrative delivery and atmosphere.

The TableTalk project aims to develop a system that allows recorded narrative voice performances to be triggered dynamically during gameplay while coordinating atmospheric effects such as sound, lighting, or other environmental cues. The system consists of a mobile application that stores and organizes narrative voice recordings paired with a tabletop device capable of producing immersive sensory feedback corresponding with story events. Together these components form a narrative artifact that enhances tabletop storytelling by coordinating narration and environmental responses within a shared interactive experience.

The Human-AI component of the project focuses on developing machine-learning tools that assist with organizing, analyzing, and retrieving narrative voice recordings used within this system. The intended dataset will consist of recorded narrative passages drawn from tabletop storytelling systems and narrative games. These recordings will vary in emotional tone, pacing, character voice, and narrative function such as exposition, environmental description, tension building, or dramatic dialogue. Because narrative recordings must be retrieved quickly during gameplay, organizing large libraries of recorded narration becomes an important technical challenge.

Because the initial set of narrative recordings is still in development, the project will begin using publicly available speech and emotional voice datasets. These datasets will allow the Human-AI contributor to develop and test the machine-learning pipeline that will later process narrative recordings produced for the TableTalk system. During the project, the contributor will design tools for audio preprocessing, classification, and metadata tagging using these datasets. As original narrative recordings become available, they can be incorporated into the system and used to refine and evaluate the models developed during the project.

The primary objective of the project is to develop AI-assisted methods that automatically classify and tag narrative recordings so they can be easily retrieved during gameplay. Rather than manually searching through a large audio library, users should be able to locate recordings based on narrative characteristics such as emotional tone, pacing, or narrative context. These classification tools would allow narration to be triggered more efficiently and could help create smoother storytelling experiences within interactive gameplay environments.

A secondary objective of the project is to explore accessibility tools that allow narrative audio content to be converted into alternative formats. Machine-learning tools may be used to generate transcripts, captions, or other forms of metadata that allow narrative content to remain accessible to players with different sensory needs. These accessibility tools may also support alternative sensory cues that can supplement narration within immersive gameplay environments.

From a computational perspective, the project explores how machine learning can assist with organizing and retrieving narrative voice performances within interactive storytelling environments. The central technical challenge is developing methods that can classify voice recordings according to narrative function and emotional tone while generating searchable metadata that allows recordings to be retrieved in response to gameplay events. The Human-AI contributor will experiment with audio feature extraction, speech embeddings, and classification models to determine which approaches best support narrative audio organization and retrieval. The resulting system will provide a prototype pipeline for managing large libraries of narrative voice recordings used in interactive storytelling systems. Initial models will be developed using publicly available speech datasets before being adapted to narrative recordings produced during the development of the TableTalk system.

## Duration

Total project length: 175 hours

## Task ideas

- Dataset preparation: Organizing and preparing speech datasets for model training, including publicly available speech and emotional voice datasets.
- Audio preprocessing: Cleaning and formatting recorded voice data through normalization, segmentation, and noise reduction.
- Narrative classiﬁcation: Developing machine-learning models that classify voice recordings by narrative function (for example exposition, suspense building, environmental description, or character dialogue).
- Emotion and tone detection: Training models that detect emotional tone in narration such as suspense, calm description, urgency, or dramatic emphasis.
- Audio feature extraction: Extracting acoustic features from recordings that can support classiﬁcation and retrieval tasks.
- Metadata generation: Automatically generating searchable metadata tags for each voice recording.
- Audio retrieval system: Building a prototype system that retrieves narrative recordings based on descriptive queries or gameplay events.
- Accessibility tools: Generating automated transcripts and captions from voice recordings.
- Prototype integration: Testing how the classiﬁcation and tagging system could organize narrative recordings for use within interactive storytelling systems.

## Expected results

- Develop a working pipeline for processing and analyzing narrative voice recordings.
- Create a labeled dataset derived from public speech and emotional voice datasets that can serve as a prototype training corpus.- - Develop a machine-learning model capable of classifying narrative recordings by tone and narrative function.
- Automatically generate searchable metadata tags for voice recordings.
- Develop a prototype system that retrieves narrative recordings based on descriptive keywords.
- Produce transcripts of narrative recordings with at least 85% transcription accuracy.
- Demonstrate how AI-generated tagging improves the speed and accuracy of locating narrative audio clips.
- Produce a working prototype pipeline capable of incorporating newly recorded narrative audio as it becomes available.

## Requirements

- Experience with Python.
- Basic familiarity with machine learning frameworks such as PyTorch or TensorFlow.
- Experience working with audio datasets or signal processing tools.
- Ability to work with labeled datasets and perform model evaluation.
- Interest in storytelling, gaming, or interactive media.
- (Expertise) Mobile app development/design

## Difficulty level

Medium

## Tests

Please use [this link](/assets/TableTalk%20Technical%20Test.pdf) to access the test for this project.

## Mentors

- [Sergei Gleyzer](mailto:human-ai@cern.ch) (University of Alabama)
- [Matthew Davis](mailto:human-ai@cern.ch) (University of Alabama)
- [Mark Barry](mailto:human-ai@cern.ch) (University of Alabama)

Please **DO NOT** contact mentors directly by email. Instead, please email [human-ai@cern.ch](mailto:human-ai@cern.ch) with Project Title and **include your CV** and **test results**. The mentors will then get in touch with you.

<!-- ## Links
  * [Paper 1](https://arxiv.org/abs/1807.11916)
  * [Paper 2](https://arxiv.org/abs/1902.08276) -->
