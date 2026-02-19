---
title: AI-Powered Funding Intelligence (FOA Ingestion + Semantic Tagging)
layout: gsoc_proposal
project: ISSR
year: 2026
organization:
  - Alabama
---

## Description

Funding Opportunity Announcements (FOAs) are distributed across many agencies, vary widely in structure, and require significant manual effort to interpret and circulate. This project will build an open-source pipeline that automatically ingests FOAs from public sources, extracts structured fields, and applies ontology-based semantic tags to support institutional research discovery and grant matching.

## Motivation

Research development teams often lose time and opportunities due to the manual process of finding, parsing, and categorizing FOAs. Automating FOA ingestion and tagging creates structured, queryable funding intelligence that can support investigator discovery, proposal development workflows, and institutional strategy.

## Project Goals

The contributor will design and implement a modular pipeline that:

- Ingests FOAs from at least **two public sources**
- Normalizes FOAs into a consistent schema (JSON + CSV)
- Applies semantic tags using a controlled ontology
- Produces clean, reproducible outputs suitable for downstream grant matching

## Scope of Work

**1) FOA Ingestion**

Build ingestion modules that scrape or retrieve FOAs from at least two sources (e.g., Grants.gov + NSF).
Requirements:

- Handle HTML and/or PDF formats
- Normalize raw text for downstream extraction

**2) Structured Extraction + Normalization**

Extract key FOA fields including:

- FOA ID (generated if missing)
- Title
- Agency
- Open/Close dates (ISO format)
- Eligibility text
- Program description
- Award range (if available)
- Source URL

Output must conform to a standardized JSON + CSV schema.

**3) Semantic Tagging**

Implement semantic tagging aligned to a controlled ontology including:

- Research domains
- Methods/approaches
- Populations
- Sponsor themes
  Tagging should use hybrid approaches:
- rule-based tagging
- embedding similarity
- optional LLM-assisted classification (stretch goal)

**4) Storage + Export**

Deliver clean, reproducible outputs including:

- JSON export
- CSV export
- update workflow for ingesting new FOAs
- documentation for reproducibility

**5) Basic Evaluation**

Create a small evaluation set to test tagging consistency and demonstrate baseline accuracy.
Deliver:

- evaluation dataset
- summary metrics (precision/recall or agreement)

## Deliverables

By the end of the project, the contributor will deliver:

- Working FOA ingestion + normalization pipeline
- Structured FOA dataset (JSON + CSV export)
- Semantic tagging module aligned to a defined ontology
- Documentation and reproducible setup instructions
- Basic evaluation of tagging consistency
- A foundation for future integration into a grant-matching system

## Stretch Goals (If Time Allows)

- Add additional FOA sources (e.g., NIH)
- Add a lightweight search interface (CLI or minimal UI)
- Add vector indexing (FAISS or Chroma) for similarity search

## Required Skills

- Strong Python programming skills
- Web scraping and text processing
- Familiarity with NLP tools (spaCy, sentence-transformers, Hugging Face, etc.)
- Understanding of embeddings and semantic similarity
- APIs and structured data formats (JSON/CSV)
- Git/GitHub

## Recommended Technical Stack

- Python
- requests + BeautifulSoup
- PyPDF / PDFMiner
- sentence-transformers
- Optional: FAISS or Chroma
- Optional (stretch): LLM API with documented usage

## Screening Task (2-4 Hours)

Build a minimal script that ingests a single FOA URL (Grants.gov or NSF), extracts fields into the required schema, applies deterministic rule-based tags, and outputs:

- foa.json
- foa.csv
  The program must run as:
  python main.py --url "<FOA_URL>" --out_dir ./out

Applicants should submit the following, along with a current CV or resume:

- main.py
- requirements.txt
- README.md
- out/foa.json
- out/foa.csv

## Mentors

- [Andrya Allen](mailto:human-ai@cern.ch) (University of Alabama)
- [Dr. Xinyue Ye](mailto:human-ai@cern.ch) (University of Alabama)
- [Dr. Andrea Underhill](mailto:human-ai@cern.ch) (University of Alabama)

Please **DO NOT** contact mentors directly by email. Instead, please email [human-ai@cern.ch](mailto:human-ai@cern.ch) with Project Title and **include your CV** and **test results**. The mentors will then get in touch with you.
