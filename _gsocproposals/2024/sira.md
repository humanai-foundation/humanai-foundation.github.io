---
title: Learning the Susceptible-Infected-Removed Model
layout: gsoc_proposal
project: SIRA
year: 2024
organization:
  - FSU
  - Tennessee

---

## Description
This project aims to use machine learning to deduce the deterministic form of the classic susceptible-infected-removed (SIR) epidemic model, defined by a set of ordinary differential equations, from a large number of synthetic epidemics, each simulated at a different SIR parameter point using the stochastic version of the SIR model.

## Duration
Total project length: 175 hours

## Task ideas
 * Starting with the stochastic equations, simulate an Epidemic based on the SIR model.
 * Train a ML model to predict the mean counts S, I, R
 * Use auto-differentiation and symbolic methods to approximate S(t), I(t), R(t)

## Expected results
 * Simulated SIR epidemic model
 * Trained machine learning model to predict the mean counts
 * Trained symbolic ML model to approximate the output

## Requirements
Python, Pytorch or Tensorflow, upper-level mathematics such as differential equations and linear algebra preferred

## Project difficulty level
Medium

## Test
Solve the evaluation task(s) for any of the other projects in the HumanAI umbrella organization.  Please send us your CV and a link to all your completed work (github repo, Jupyter notebook + pdf of Jupyter notebook with output) to [human-ai@cern.ch](mailto:human-ai@cern.ch) with Evaluation Test: SIRA in the title. In the email specify which evaluation test(s) you solved. 

## Mentors
  * [Harrison Prosper](human-ai@cern.ch) (Florida State University)
  * [Olivia Prosper](human-ai@cern.ch) (University of Tennessee)
  * [Sergei Gleyzer](human-ai@cern.ch) (University of Alabama)

Please **DO NOT** contact mentors directly by email. Instead, please email [human-ai@cern.ch](mailto:human-ai@cern.ch) with Project Title and **include your CV** and **test results**. The mentors will then get in touch with you.
