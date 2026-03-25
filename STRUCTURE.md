# Repository Structure Guide

This document provides a detailed overview of how the HumanAI Foundation website is organized.

## Directory Tree Overview

```
humanai-foundation.github.io/
├── Root Content Pages
│   ├── index.html                          # Homepage
│   ├── projects.md                         # Featured projects
│   ├── forums.md                           # Discussion forums
│   ├── get_involved.md                     # Contributing guide
│   ├── future-events.md                    # Upcoming events
│   ├── what_are_activities.md              # Activity areas explained
│   ├── what_are_WGs.md                     # Working groups explained
│   ├── services.md                         # Services offered
│   ├── toolkit.md                          # Community toolkit
│   ├── newsletter.html                     # Newsletter page
│   └── [Other pages]
│
├── Configuration
│   ├── _config.yml                         # Jekyll configuration
│   ├── Gemfile                             # Ruby dependencies
│   ├── CNAME                               # Domain configuration
│   └── .pre-commit-config.yaml             # Git hooks
│
├── Layout & Styling
│   ├── _layouts/                           # HTML templates
│   │   ├── default.html                    # Main layout
│   │   ├── page.html                       # Centered page layout
│   │   ├── educator.html                   # Profile layout
│   │   ├── event.html                      # Event layout
│   │   ├── gsoc_proposal.html              # GSoC proposal layout
│   │   └── [Other layouts]
│   │
│   ├── _includes/                          # Reusable components
│   │   ├── navbar.ext                      # Navigation bar
│   │   ├── sidebar.ext                     # Sidebar navigation
│   │   ├── profile_header.html             # Profile header
│   │   ├── gsoc_project_list.ext           # GSoC project list
│   │   ├── toc.html                        # Table of contents
│   │   └── [Other includes]
│   │
│   └── css/
│       └── hsf.css                         # Custom styles
│
├── Content Collections (Jekyll)
│   ├── _activities/                        # Activity areas
│   │   ├── gsoc2024.md                     # Google Summer of Code
│   │   ├── gsoc2025.md
│   │   ├── gsoc2026.md
│   │   └── studentblogs.md                 # Student blog posts
│   │
│   ├── _workinggroups/                     # Technical working groups
│   │   ├── dataanalysis.md
│   │   ├── detsim.md
│   │   ├── frameworks.md
│   │   ├── generators.md
│   │   ├── pyhep.md
│   │   ├── recotrigger.md
│   │   ├── toolsandpackaging.md
│   │   └── training.md
│   │
│   ├── _training/                          # Training programs
│   │   └── [Training school entries]
│   │
│   ├── _profiles/                          # Community member profiles
│   │   ├── 000_template.md                 # Template for new profiles
│   │   ├── firstname_lastname.md            # Individual profiles
│   │   └── [100+ member profiles]
│   │
│   ├── _data/                              # YAML data files
│   │   └── training-schools.yml            # Training school data
│   │
│   └── [Other collections - see below]
│
├── Google Summer of Code (GSoC)
│   │
│   ├── _gsocorgs/                          # GSoC Organizations
│   │   ├── 2020/
│   │   ├── 2021/
│   │   ├── 2022/
│   │   ├── 2023/
│   │   ├── 2024/
│   │   ├── 2025/
│   │   └── 2026/                           # Current + future years
│   │       └── org-name.md                 # Organization submissions
│   │
│   ├── _gsocprojects/                      # GSoC Projects
│   │   ├── 2021/
│   │   ├── 2022/
│   │   ├── 2023/
│   │   ├── 2024/
│   │   ├── 2025/
│   │   ├── 2026/
│   │   └── archived/
│   │       └── project-name.md             # Project ideas/proposals
│   │
│   ├── _gsocproposals/                     # GSoC Student Proposals
│   │   ├── 2020/
│   │   ├── 2021/
│   │   ├── 2022/
│   │   ├── 2023/
│   │   ├── 2024/
│   │   ├── 2025/
│   │   ├── 2026/
│   │   └── archived/
│   │       └── student-proposal.md         # Student submissions
│   │
│   └── gsoc/                               # Output pages
│       ├── 2023/summary.md                 # Year summary pages
│       ├── 2024/summary.md
│       ├── 2025/summary.md
│       └── 2026/summary.md
│
├── Legacy Systems (Archived)
│   ├── _gsdocs-orgs/                       # Google Docs export (legacy)
│   │   ├── 2020/
│   │   └── example/
│   │
│   ├── _gsdocs-projects/                   # Google Docs export (legacy)
│   │   └── 2020/
│   │
│   ├── _gsdocs-proposals/                  # Google Docs export (legacy)
│   │   ├── 2020/
│   │   └── example/
│   │
│   ├── _drafts/                            # Unpublished drafts
│   │   └── nodate-*.md                     # Draft content
│   │
│   ├── archive/                            # Old pages
│   │   └── [Archived content]
│   │
│   └── cwp/                                # Community White Papers
│       └── papers/
│
├── Structured Content
│   ├── announcements/
│   │   ├── _posts/                         # News items
│   │   │   └── 2021-01-11-gsoc.md
│   │   └── [Announcement pages]
│   │
│   ├── events/                             # Event pages
│   │   └── event-*.md
│   │
│   ├── newsletter/                         # Newsletter archives
│   │   └── newsletter-*.md
│   │
│   ├── notes/                              # Meeting notes
│   │   └── *.md
│   │
│   ├── organization/                       # Organization pages
│   │   ├── team.html                       # Coordination team
│   │   └── [Org pages]
│   │
│   └── gsdocs/                             # Generated documentation
│       └── [Generated pages]
│
├── Static Assets
│   ├── assets/                             # Web assets
│   │   └── [CSS, JS, etc.]
│   │
│   ├── images/                             # Site images
│   │   ├── humanai.jpg                     # Favicon/logo
│   │   ├── GSoC/                           # GSoC logos
│   │   └── [Other images]
│   │
│   ├── css/                                # Stylesheets
│   │   └── hsf.css                         # Main styles
│   │
│   └── Schools/                            # Training school info
│       └── [School pages]
│
├── Utilities & Scripts
│   ├── scripts/
│   │   ├── add_training_event.py           # Add training events
│   │   └── profile_maintenance_script.py   # Manage profiles
│   │
│   ├── inventory/                          # Project inventory
│   │   └── inventory.md                    # Community projects
│   │
│   └── .github/                            # GitHub specific
│       ├── workflows/                      # CI/CD workflows
│       └── [GitHub config]
│
└── Development Files
    ├── README.md                           # This file (updated)
    ├── CONTRIBUTING.md                     # Contribution guidelines
    ├── STRUCTURE.md                        # This structure guide
    ├── .gitignore                          # Git ignore rules
    ├── Gemfile.lock                        # Locked dependencies
    ├── feed.xml.inactive                   # RSS feed (inactive)
    ├── .pre-commit-config.yaml             # Pre-commit hooks
    ├── .travis-scripts/                    # CI/CD scripts
    └── .jekyll-metadata                    # Jekyll cache (generated)
```

## Key Collections Explained

### Jekyll Collections

Jekyll uses "collections" for organized content. Each collection generates output pages:

| Collection | Stores | Output Pattern | Purpose |
|-----------|--------|----------------|---------|
| `_activities` | Activity definitions | `/activities/:title.html` | Interest groups, GSoC activity |
| `_workinggroups` | WG descriptions | `/workinggroups/:title.html` | Technical working groups |
| `_training` | Training program data | `/training/:path.html` | Schools, courses |
| `_profiles` | Member profiles | `/profiles/:title.html` | Community member directory |
| `_gsocorgs` | Org submissions (by year) | `/gsoc/organizations/:path.html` | GSoC participants |
| `_gsocprojects` | Project proposals (by year) | `/gsoc/projects/:path.html` | GSoC project ideas |
| `_gsocproposals` | Student proposals (by year) | `/gsoc/:path.html` | GSoC applications |