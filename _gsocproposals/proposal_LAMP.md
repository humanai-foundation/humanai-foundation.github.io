---
title: Late Antiquity Modelling Project 
layout: gsoc_proposal
project: LAMP
year: 2025
organization:
    - Alabama

---
## Description
The Late Antiquity Modeling Project (LAMP) is an international, interdisciplinary research collective. LAMP uses computational methods to reconstruct the embodied experiences of ancient buildings and landscapes. In so doing, the project crafts more accurate and embodied histories of religious communities across geographic and temporal contexts. Of particular interest to LAMP is how these embodied reconstructions might be leveraged to rewrite the history of the first centuries of Christianity. 

Our current project centers on the late antique necropolis of El Bagawat in Egypt’s Kharga Oasis. We are interested in developing machine learning approaches that will: 1) determine the pathways taken between buildings within the site, as well as recovering pathways no longer visible, and 2) allow to determine the visibility and audibility of any given point at the site from a particular location. 

## Duration
Total project length: 350 hours

## Task Ideas

### 1) Path Tracing Simulations
The current state of the art in architectural path tracing focuses on three kinds of digital tools. For work with GIS and SAR applications, the movements of people are often calculated using algorithms for hydrographic analysis. Human inhabitation patterns are approximated, like water flows, through "least cost paths" derived from the steepness of a given terrain alone. This kind of simulation only takes one variable into account—steepness—and leaves a whole host of other possible influences unaccounted for, including ground density and the direction of the structures’ entrances. Consultants in the architectural industry, depending on their clients' needs, time available, and compute power, either use discrete event simulation or agential modelling to simulate the movements of a building’s or landscape’s future users. Discrete event simulation cannot identify the possible range of possible paths a user might take if paths are unknown. Within complex landscapes that are architecturally and topographically complicated, agential modelling does not successfully take into account existing spatial relationships in the pathway possibilities it generates. 

In light of the limitations of these existing tools, the contributor’s task will be to develop a means of identifying human patterns of moving through the necropolis’ complex topography and building distribution. The solution must take into account the existing known pathways and the “neighborhood” arrangement of buildings and their entrances. 

We are interested in what are the most likely pathways people took between a series of buildings within the site. This requires the determination of the probability of different pathways between any given building at the site. As such, it is necessary to first determine if there any pathways no longer visible today.

### 2) Views and Visibilities
Viewsheds and visibility are critical categories for architectural analyses of the built environment, with modes of seeing and being seen considered core aspects of how people inhabit their spaces. Architects and other design team members deploy ray-traced 3D visualisations to explore particular views of buildings.  Others in the academic sub-field of Space Syntax Analysis have developed software, like depthmapX, for visualising and quantifying "viewsheds," tracking gradients of visibility. GIS software developers have taken up a similar project, developing visibility analysis plugins for GIS and SAR software packages like ERDASImagine and QGIS. However, these digital tools are only effective at analyzing buildings planimetrically, that is in two-dimensions. They do not take into account differences in the height of design elements or buildings on a given site, nor do they account for window openings or translucent partitions. 

In light of the limitations of existing tools, your task is to develop a solution for determining viewsheds on site at the Necropolis of El Bagawat in Egypt, and gradients of visibility. Your solution will need to account for the site's entire three dimensionality. This includes the landscape’s hilly topography, and the building distribution and their heights. The solution will need to determine what a person could see from any given point on site. 


## Expected Results

### Task 1):
    * Train a computer vision model to identify paths between any given buildings on a site, taking into account topography, waking surface types, etc
    * Render the paths as a GIS vector layer

### Task 2):
    * Train a model to project 3D viewsheds from a complex 3D scene
    * Render the viewsheds as a GIS vector layer
    * (If time permits) Render the viewsheds as a 3D volume

## Requirements
    * Experience with GIS (SAR & digital elevation model experience is a plus)
    * Experience with image processing
    * QGIS (with Orfeo Toolbox), ERDASImagine, or GIS/SAR software of your choice

## Project difficulty level
Medium/Hard

## Test
Please find all data at [this link](https://app.box.com/s/6c5tv2nvbm9d1a7bpmryvmc00sdo79op)

## Mentors

### Project Directors
    * [Camille Leon Angelo](mailto:humanai@cern.ch) (University of Alabama)
    * [Joshua Silver](mailto:humanai@cern.ch) (Karlsruhe Institute of Technology)

### Project Collaborators
    * [Rachel Dubose](mailto:humanai@cern.ch) (University of Alabama)
    * [Jefferey Turner](mailto:humanai@cern.ch) (University of Alabama)
    * [Richard Newton](mailto:humanai@cern.ch) (University of Alabama)
