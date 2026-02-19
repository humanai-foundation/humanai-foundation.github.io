---
title: Humanlike AI Systems and Trust Attribution
layout: gsoc_proposal
project: ISSR
year: 2026
organization:
  - Alabama
---

## Description

This project will build an open-source, modular experimentation engine for studying **trust calibration** in AI-assisted decision systems. The platform will allow researchers to systematically manipulate humanlike **and authority-signaling interface cues** (e.g., assistant name, tone, confidence framing) and log user behavior at high temporal resolution to measure reliance vs. override decisions.

The end product will be a reusable research infrastructure for human–AI trust, calibration, and adoption studies.

## Motivation

As AI assistants increasingly adopt humanlike names, conversational tone, avatars, and confidence framing, users infer competence, agency, and intentionality from interface design alone. These inferences can lead to appropriate reliance, underuse, or overtrust.

Most existing research relies heavily on self-report trust scales. This project focuses on **behavioral trust metrics**, grounded in observable decision behavior and structured logging.

## Project Goals

The contributor will build a modular web-based experimental environment that supports:

- Controlled manipulation of humanlike and authority-signaling cues
- One structured decision task that produces reliance vs. override outcomes
- Fine-grained behavioral instrumentation (decisions + latency + interaction sequence)
- Exportable, analysis-ready datasets (CSV + JSON)
- A reproducible analysis notebook (Python or R)
- Documentation and deployment instructions for reuse

## Scope of Work

**1) Experimental Web Application**

Build a lightweight experiment platform (React/Next.js or similar) including:

- Participant ID assignment
- Randomized condition assignment (A/B or multi-condition)
- Task presentation interface
- Logging backend
- Dataset export functionality
  Architecture must support modular cue manipulation.

**2. Cue Manipulation System**

Implement a condition management framework enabling systematic manipulation of at least 3 cue dimensions, such as:

- Agent naming (neutral label vs. humanlike name)
- Tone (formal/technical vs. conversational/social)
- Confidence framing (calibrated probability vs. overstated certainty)
  The system should be extensible so additional cues (e.g., visual identity, role assignment source) can be added later.

**3. Behavioral Task Module**

Implement one structured decision task that generates clear behavioral trust outcomes:
Recommendation Acceptance Task

- AI provides a recommendation
- Participant chooses to accept or override
- Optional participant confidence rating
  AI accuracy should be experimentally controlled (e.g., fixed accuracy rate) to evaluate trust calibration.

**4. Instrumentation and Logging**

Design and implement a clean event schema capturing:

- participant_id
- condition
- decision
- timestamp
- latency_ms
- optional trust scale responses

Requirements:

- Chronologically structured logging
- Exportable JSON and CSV formats
- Clear schema documentation

**5. Dataset Export + Analysis Notebook**

Deliver:

- Clean CSV + JSON exports
- A basic analysis notebook (Python or R) demonstrating:
  - reliance rate by condition
  - override rate by condition
  - mean response latency
  - optional trust scale vs. behavior comparison

## Deliverables

By the end of the project, the contributor will provide:

- Functional experimental prototype (local or deployed)
- Condition assignment + cue manipulation module
- Behavioral logging backend
- Structured event schema documentation
- Sample dataset
- Analysis notebook
- "How to Run" documentation
- All code in a public GitHub repository

## Stretch Goals (If Time Allows)

- Add additional cue dimensions (visual identity, role assignment source)
- Add a second task type (e.g., product choice / willingness-to-pay)
- Add optional post-task trust ratings and richer interaction logging

## Required Skills

- JavaScript
- React / Next.js (or similar framework)
- Basic backend logging (API endpoints, file output, or lightweight DB)
- JSON/CSV structuring
- Experimental logic / A/B testing fundamentals
- Basic statistical literacy

## Project difficulty level

Moderate. This project requires integration of frontend development, experimental condition logic, and structured behavioral logging.

## Mentorship Expectations

Contributors will be expected to:

- Participate in weekly mentor check-ins
- Submit incremental pull requests
- Document design decisions
- Maintain reproducible workflows

## Broader Impact

This project supports responsible AI by:

- Distinguishing perceived capability from actual capability
- Measuring when humanlike cues alter trust calibration
- Enabling evidence-based interface design for AI-assisted decision systems

## Screening Test (2-4 Hours)

Applicants must build a minimal working prototype.

Requirements:

- A simple web page with two conditions (A/B) differing in one cue (e.g., name + tone)
- One decision task:
  - Accept/reject AI recommendation OR choose between two products
- Logging to JSON or CSV including:
  - participant_id
  - condition
  - decision
  - timestamp
  - latency_ms

## Submission

- A current CV or resume
- Test results
  - GitHub repo link
  - Short README explaining:
    - condition logic
    - logging implementation
    - how to run locally
    - sample output file

## Mentors

- [Andrya Allen](mailto:human-ai@cern.ch) (University of Alabama)
- [Dr. Xinyue Ye](mailto:human-ai@cern.ch) (University of Alabama)
- [Dr. Kelsey Chappetta](mailto:human-ai@cern.ch) (University of Alabama)
- [Dr. Andrea Underhill](mailto:human-ai@cern.ch) (University of Alabama)

Please **DO NOT** contact mentors directly by email. Instead, please email [human-ai@cern.ch](mailto:human-ai@cern.ch) with Project Title and **include your CV** and **test results**. The mentors will then get in touch with you.
