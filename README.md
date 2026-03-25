# HumanAI Foundation Website

![HumanAI](images/humanai.jpg)

A community-driven website built with Jekyll for the HumanAI Foundation, hosting resources for research, education, and community collaboration in AI and related fields.

**Live Website:** [humanai-foundation.github.io](https://humanai-foundation.github.io)

## About This Repository

This is the source code for the HumanAI Foundation website. The site serves as a central hub for:

- **Google Summer of Code (GSoC)** - Annual student coding projects and internships
- **Working Groups** - Technical collaboration areas (Data Analysis, Detector Simulation, Frameworks, Generators, PyHEP, Reconstruction & Triggering, Tools & Packaging, Training)
- **Activity Areas** - Community initiatives and interest groups
- **Training & Education** - Schools, courses, and educational resources
- **Community Profiles** - Member directory with professional backgrounds
- **Forums & Discussions** - Communication channels and knowledge sharing
- **Events** - Meetings, workshops, and community gatherings
- **Project Showcase** - Featured projects and initiatives
- **Newsletter & News** - Regular updates and announcements

## Repository Structure

### Core Content Directories

```
├── Root Pages          # Main navigation pages (projects.md, forums.md, events.html, etc.)
├── _activities/        # Activity areas and initiatives
├── _workinggroups/     # Technical working groups
├── _training/          # Training programs and schools
├── _profiles/          # Community member profiles
├── announcements/      # News and blog posts
├── events/             # Event pages and schedules
├── newsletter/         # Newsletter archives
└── notes/              # Meeting notes and documentation
```

### GSoC Directories (Organized by Year: 2020-2026)

```
├── _gsocorgs/          # GSoC participating organizations
├── _gsocprojects/      # GSoC projects and ideas
└── _gsocproposals/     # GSoC student proposals
```

### Documentation & Data

```
├── _config.yml         # Jekyll configuration file
├── _data/              # YAML data files (training-schools.yml)
├── _layouts/           # HTML templates (default, educator, event, gsoc_proposal, etc.)
├── _includes/          # Reusable HTML components (navbar, sidebars, lists, etc.)
├── css/                # Stylesheets (hsf.css)
├── assets/             # Static assets
├── images/             # Site images and icons
└── scripts/            # Utility scripts (profile maintenance, training events)
```

### Legacy & Archives

```
├── _gsdocs-*/          # Google Docs export system (2020, example, etc.)
├── _drafts/            # Draft content
├── archive/            # Archived pages
├── cwp/                # Community White Papers
└── inventory/          # Community project inventory
```

## Getting Started

### Prerequisites

- **Ruby** 2.7+ (for Jekyll)
- **Bundle** (Ruby gem manager)

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/humanai-foundation/humanai-foundation.github.io.git
   cd humanai-foundation.github.io
   ```

2. **Install dependencies**
   ```bash
   bundle install
   ```

3. **Build and serve locally**
   ```bash
   bundle exec jekyll serve
   ```
   The site will be available at `http://localhost:4000`

4. **Watch for changes**
   ```bash
   bundle exec jekyll serve --watch
   ```

### Using Docker (Optional)

If you prefer Docker:

```bash
docker run --rm -v $(pwd):/srv/jekyll -p 4000:4000 jekyll/jekyll:latest jekyll serve
```

## What's Inside

### Key Pages

- **[index.html](index.html)** - Homepage
- **[projects.md](projects.md)** - Featured projects
- **[forums.md](forums.md)** - Discussion forums
- **[get_involved.md](get_involved.md)** - How to contribute
- **[future-events.md](future-events.md)** - Upcoming meetings and events
- **[what_are_activities.md](what_are_activities.md)** - Activity areas overview
- **[what_are_WGs.md](what_are_WGs.md)** - Working groups overview

### Collections

The site uses Jekyll collections for structured content:

| Collection | Output Path | Purpose |
|-----------|-------------|---------|
| `activities` | `/activities/:title.html` | Activity areas and initiatives |
| `workinggroups` | `/workinggroups/:title.html` | Technical working groups |
| `training` | `/training/:path.html` | Training programs and schools |
| `profiles` | `/profiles/:title.html` | Community member profiles |
| `gsocorgs` | `/gsoc/organizations/:path.html` | GSoC organizations |
| `gsocprojects` | `/gsoc/projects/:path.html` | GSoC project ideas |
| `gsocproposals` | `/gsoc/:path.html` | GSoC proposals |
| `gsdocs-orgs` (legacy) | `/gsdocs/organizations/:path.html` | Exported documentation |

## Contributing

### How to Contribute

1. **Report Issues** - Use GitHub Issues for bugs or suggestions
2. **Edit Pages** - Click "Improve this page" link at the bottom of any page for quick edits
3. **Pull Requests** - Submit changes via pull requests on the `master` branch
4. **Add Content** - Follow the appropriate directory structure and frontmatter format

### File Frontmatter Template

Most markdown files should start with YAML frontmatter:

```yaml
---
title: "Page Title"
author: "Your Name"
layout: default
---
```

### Content Guidelines

- Use descriptive titles and clear descriptions
- Include proper author attribution where applicable
- Use consistent formatting and markdown style
- Keep files organized in appropriate directories
- For new profiles, use the template in [_profiles/000_template.md](_profiles/000_template.md)

## Jekyll Configuration

### Key Settings (_config.yml)

- **Theme**: Custom with Bootstrap 3.3.5 (Flatly)
- **Markdown Engine**: Kramdown
- **Highlighter**: Rouge
- **Plugins**: jekyll-mentions, jekyll-sitemap, jekyll-redirect-from, jekyll-feed

### Build Process

The site is built automatically on push to GitHub Pages. To build locally:

```bash
bundle exec jekyll build
```

Output is generated in the `_site/` directory.

## Maintenance Scripts

### Available Utilities

- **[scripts/profile_maintenance_script.py](scripts/profile_maintenance_script.py)** - Manage community profiles
- **[scripts/add_training_event.py](scripts/add_training_event.py)** - Add training school events

## Important Notes

### Image Management

- Favicon: `images/humanai.jpg`
- GSoC logo: `images/GSoC/GSoC-icon-192.png`
- Use relative paths: `/images/filename.ext`

### External Links

- **GitHub Docs**: https://github.com/HSF/documents/blob/master
- **Community White Papers**: https://github.com/HSF/documents/blob/master/CWP/papers/2017-11
- **CERN Indico**: http://indico.cern.ch/category/5816/
- **Project Organization**: https://groups.google.com/forum/#!forum/hsf-forum

## Troubleshooting

### Build Issues

```bash
# Clean build cache
rm -rf _site/ .jekyll-metadata

# Rebuild
bundle exec jekyll build
```

### Dependency Issues

```bash
# Update gems
bundle update

# Reinstall
rm Gemfile.lock
bundle install
```

## Community & Communication

- **Main Forum**: [HSF Forum](https://groups.google.com/forum/#!forum/hsf-forum)
- **Coordination**: hsf-coordination@googlegroups.com
- **Issues**: [GitHub Issues](https://github.com/humanai-foundation/humanai-foundation.github.io/issues)

## License & Attribution

This website builds on the HEP Software Foundation infrastructure. Thanks to:

- **[GitHub Pages](https://pages.github.com/)** - Hosting
- **[Jekyll](http://jekyllrb.com/)** - Static site generator
- **[Bootstrap](http://getbootstrap.com/)** - CSS framework

## Site Organization & Authors

- **Founder/Lead**: Benedikt Hegner
- **Contributions**: Graeme Stewart and community members
- **Last Updated**: March 2026

---

**Want to help?** See [How to Get Involved](get_involved.md) or submit a pull request!
