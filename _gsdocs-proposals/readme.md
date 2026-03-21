

title: Painting in a Painting - AI-driven Hidden Image Reconstruction without Multispectral Dependency
layout: gsoc_proposal
project: ArtExtraction(Srishti-1806)
year: 2026
organization:

* Alabama



## Description

Traditional approaches to uncover hidden images beneath paintings rely heavily on multispectral and X-ray imaging. While effective, these methods require expensive hardware, controlled environments, and are not scalable for widespread use.

X Ray based machines are not available to many researchers and are quite uncommon too. My solutopn focuses on:
1) Shifting database dependency to Generative AI solutions.
2) Making this solution scalable and availabe to all by refining the outputs using existing imprinting models using existing ML tools.

This proposal introduces a **software-first, AI-driven reconstruction pipeline** that aims to infer hidden structures and compositions directly from standard RGB images, eliminating the dependency on specialized imaging equipment.

The core idea is to leverage recent advancements in **generative AI, diffusion models, and structure-preserving networks** to reconstruct underlying visual information from degraded or modified paintings. Instead of explicitly detecting hidden layers through physical imaging, this system performs **intelligent reconstruction and inference**, enabling a broader and more accessible approach to hidden art exploration.

The pipeline operates in multiple stages:

* **Denoising and Preprocessing** to clean and normalize the input
* **Damage and Region Detection** using semantic segmentation and edge-based masking
* **AI Inpainting using Diffusion Models** to reconstruct missing or altered regions
* **Structure Preservation via ControlNet** to maintain geometric and compositional consistency
* **Enhancement and Super-Resolution** for high-quality output generation

This approach not only restores damaged artworks but also provides a **probabilistic reconstruction of hidden compositions**, making it a powerful tool for historians, researchers, and digital archivists.


## Why this proposal is an improvement

This proposal enhances the original idea in several key ways:

### 1. Accessibility and Scalability

Unlike traditional methods that require multispectral or X-ray data, this system works with **standard RGB images**, making it deployable in low-resource environments and scalable across large datasets.

### 2. Software-first Approach

By shifting from hardware-dependent imaging to AI-driven reconstruction, the project reduces cost and increases usability. This allows broader adoption across institutions, researchers, and independent analysts.

### 3. End-to-End Automated Pipeline

The proposed system integrates multiple stages—denoising, segmentation, inpainting, enhancement, and upscaling—into a **single cohesive pipeline**, reducing manual intervention.

### 4. Generative Reconstruction instead of Detection

Rather than only identifying whether a hidden image exists, this approach attempts to **reconstruct plausible underlying visuals**, providing richer insights.

### 5. Deployment-ready Architecture

The system is designed with **FastAPI and Docker-based deployment**, ensuring reproducibility, scalability, and ease of integration into real-world applications.


## Duration

Total project length: 175 hours


## Task ideas

* Design and implement a multi-stage AI pipeline for artifact reconstruction
* Develop segmentation-based masking techniques for identifying altered regions
* Integrate diffusion-based inpainting for reconstructing missing or hidden content
* Apply ControlNet for structure preservation during generation
* Implement enhancement and super-resolution modules
* Optimize pipeline performance for CPU/GPU environments


## Expected results

* A fully functional AI pipeline capable of reconstructing and enhancing damaged paintings
* A system that can infer and visualize potential hidden compositions
* Modular architecture enabling future extensions (e.g., multispectral integration)
* Deployment-ready API for real-world usage
* Documentation covering architecture, implementation, and optimization


## Tech stack

* **Programming Language:** Python
* **Deep Learning Framework:** PyTorch
* **Computer Vision:** OpenCV, NumPy
* **Models and Libraries:**

  * Hugging Face Transformers (Mask2Former for segmentation)
  * Diffusers (Stable Diffusion Inpainting)
  * ControlNet (structure preservation)
  * RealESRGAN (super-resolution)
* **Backend:** FastAPI
* **Deployment:** Docker
* **Utilities:** Python-dotenv, Hugging Face Hub


## Why my proposal stands out

This proposal stands out due to its **practicality, innovation, and real-world applicability**:

* It transforms a research-heavy concept into a **deployable engineering solution**
* It removes reliance on costly imaging techniques, making the solution **democratized and scalable**
* It combines multiple state-of-the-art AI models into a **cohesive, production-ready system**
* It focuses not just on detection but on **meaningful reconstruction and visualization**
* It is built with deployment and usability in mind, ensuring impact beyond experimentation

Overall, this project bridges the gap between **academic research and practical implementation**, offering a novel approach to exploring hidden art using modern AI techniques.


## Requirements

* Python
* PyTorch
* Computer vision fundamentals
* Basic understanding of deep learning and generative models


## Project difficulty level

Medium to High


## Mentors

* Emanuele Usai (University of Alabama)
* Sergei Gleyzer (University of Alabama)
