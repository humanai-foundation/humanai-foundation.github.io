title: Painting in a Painting - AI-driven Hidden Image Reconstruction without Multispectral Dependency
layout: gsoc_proposal
project: ArtExtraction(Srishti-1806)
year: 2026
organization:

* Alabama

---

## Description

Traditional approaches to uncover hidden images beneath paintings rely heavily on multispectral and X-ray imaging. While effective, these methods require expensive hardware, controlled environments, and expert handling, making them difficult to scale and inaccessible to many researchers.

X-ray-based systems are not widely available, especially to smaller institutions and independent historians. This proposal addresses these limitations by introducing a **software-first, AI-driven reconstruction pipeline** that operates solely on standard RGB images.

The core objective is not just detection, but **probabilistic reconstruction of plausible underlying compositions**, using modern generative AI techniques.

---

## Problem Statement & Proposed Solution

### 1. Hardware Dependency & Lack of Scalability

**Challenge:**
Hidden image detection relies on costly imaging techniques (X-ray, multispectral), limiting accessibility.

**Solution:**
An **RGB-based AI inference pipeline** that leverages deep learning to infer underlying structures directly from standard images. This removes the need for specialized equipment and enables scalable deployment.

---

### 2. Manual Identification of Layered History

**Challenge:**
Identifying overpaintings and layered compositions requires manual inspection and expert analysis.

**Solution:**
A **Generative Reconstruction Pipeline** combining:

* Semantic Segmentation (Mask2Former)
* Diffusion-based Inpainting (Stable Diffusion)

This allows automatic identification and reconstruction of altered regions.

---

### 3. Data Complexity & Information Extraction

**Challenge:**
Analyzing damage, pigment variation, and restorations from raw data is complex and time-intensive.

**Solution:**
An **end-to-end automated pipeline** integrating:

* Edge detection (Canny)
* Structure guidance (ControlNet)

This enables structured extraction and reconstruction without manual intervention.

---

## Proposed Approach

The system leverages **diffusion models, segmentation networks, and structure-preserving conditioning** to reconstruct underlying visual content.

### Pipeline Overview

```
Input Image (RGB)
        ↓
Denoising & Preprocessing
        ↓
Segmentation (Mask2Former)
        ↓
Mask Generation (Altered Regions)
        ↓
Edge Extraction (Canny)
        ↓
ControlNet (Structure Conditioning)
        ↓
Stable Diffusion (Inpainting)
        ↓
Super Resolution (RealESRGAN)
        ↓
Final Reconstruction
```

---

## Training Strategy (Key Innovation)

A major challenge is the lack of ground truth hidden images.

To address this, I propose a **synthetic data generation pipeline**:

* Start with clean paintings
* Simulate overpainting using:

  * occlusions
  * texture overlays
  * noise and degradation

This creates supervised pairs:

| Input (Modified Painting) | Target (Original Painting) |
| ------------------------- | -------------------------- |

This enables effective training of reconstruction models while maintaining realism.

---

## Evaluation Strategy

Since reconstruction is inherently uncertain, evaluation is performed across multiple dimensions:

### Quantitative Metrics

* **SSIM (Structural Similarity Index)**
* **PSNR (Peak Signal-to-Noise Ratio)**
* **LPIPS (Perceptual Similarity)**

### Structural Consistency

* Edge similarity (Canny overlap)
* Feature similarity using CLIP embeddings

### Retrieval-based Validation

* Compare reconstructed outputs with nearest neighbors in embedding space

### Qualitative Evaluation

* Visual inspection
* Comparison with known restored artworks

---

## Uncertainty Estimation

The system models reconstruction as a **probabilistic process**:

* Generate multiple reconstructions per image
* Measure variance across outputs
* Highlight low-confidence regions

This ensures transparency and avoids overconfident interpretations.

---

## Why this proposal is an improvement

### 1. Accessibility and Scalability

Works entirely on RGB images → eliminates dependency on specialized hardware.

### 2. Software-first Approach

Transforms a hardware-intensive problem into a scalable AI solution.

### 3. End-to-End Automated Pipeline

Fully integrated pipeline reduces manual effort.

### 4. Generative Reconstruction

Moves beyond detection → reconstructs plausible hidden compositions.

### 5. Deployment-ready Architecture

Designed for real-world usage using FastAPI + Docker.

---

## Duration

Total project length: 175 hours

---

## Task ideas

* Design and implement multi-stage reconstruction pipeline
* Develop segmentation-based masking techniques
* Train synthetic data generation pipeline
* Integrate diffusion-based inpainting
* Apply ControlNet for structure preservation
* Implement enhancement and super-resolution
* Optimize for CPU/GPU environments
* Develop evaluation and uncertainty estimation modules

---

## Expected results

* Functional AI pipeline for artifact reconstruction
* Ability to generate plausible hidden compositions
* Quantitative + qualitative evaluation framework
* Modular architecture for future extensions
* Deployment-ready API

---

## Tech stack

* **Programming Language:** Python
* **Framework:** PyTorch
* **Computer Vision:** OpenCV, NumPy

### Models & Libraries:

* Hugging Face Transformers (**Mask2Former**)
* Diffusers (**Stable Diffusion Inpainting**)
* **ControlNet**
* **RealESRGAN**

### Backend & Deployment:

* FastAPI
* Docker
* Hugging Face Hub

---

## Why my proposal stands out

This proposal distinguishes itself through:

* **Bridging research and engineering**
* **Eliminating dependency on expensive imaging hardware**
* **Introducing synthetic supervision for training**
* **Combining multiple SOTA models into a unified pipeline**
* **Providing uncertainty-aware reconstruction**
* **Focusing on deployability and real-world usability**

Rather than only detecting hidden images, this system provides **interpretable, visual reconstructions**, enabling deeper insights into artistic processes and history.

---

## Requirements

* Python
* PyTorch
* Computer vision fundamentals
* Understanding of deep learning and generative models

---

## Project difficulty level

Medium to High

---

## Mentors

* Emanuele Usai (University of Alabama)
* Sergei Gleyzer (University of Alabama)
