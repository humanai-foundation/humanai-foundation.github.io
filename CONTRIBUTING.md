# Contributing to HumanAI Foundation Website

Thank you for your interest in contributing! We welcome contributions from all community members, whether you're fixing typos, adding content, or improving the site structure.

## Ways to Contribute

### 1. **Quick Edits (Easy)**
Every page on the live website has an "Improve this page" link at the bottom. Click it to edit directly on GitHub (requires login).

### 2. **Small Changes (Easy)**
- Fix typos or grammar
- Update existing content
- Add links or references
- Improve formatting

### 3. **New Content (Medium)**
- Add activity descriptions
- Create working group pages
- Add training school information
- Publish profiles and news
### 4. **Structural Changes (Advanced)**
- Modify layouts or templates
- Update Jekyll configuration
- Reorganize directory structure
- Create new collections

## Getting Started

### Option A: Edit on GitHub (No Local Setup)

1. Navigate to the file you want to edit
2. Click the pencil icon (Edit)
3. Make your changes
4. Write a commit message
5. Create a pull request

**Best for:** Quick fixes, typos, small content updates

### Option B: Fork & Clone (Full Development)

1. **Fork the repository** on GitHub
2. **Clone your fork locally**
   ```bash
   git clone https://github.com/YOUR-USERNAME/humanai-foundation.github.io.git
   cd humanai-foundation.github.io
   ```
3. **Create a new branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes** (see sections below)
5. **Test locally**
   ```bash
   bundle install
   bundle exec jekyll serve
   ```
   Visit `http://localhost:4000` to preview
6. **Commit and push**
   ```bash
   git add .
   git commit -m "Description of your changes"
   git push origin feature/your-feature-name
   ```
7. **Create a pull request** on GitHub

## Content Guidelines

### General Rules

- **Use clear, concise language** - Write for a diverse audience
- **Include metadata** - All markdown files need YAML frontmatter
- **Use relative links** - Links should work locally and on GitHub Pages
- **Follow existing structure** - Keep content organized in logical directories
- **Check spelling** - Use a spell-checker before submitting

### YAML Frontmatter Template

```yaml
---
title: "Page Title"
author: "Your Name"
layout: default
---
```

**Common layouts:**
- `default` - Standard page with navigation
- `page` - Simple page
- `educator` - For educator profiles
- `event` - For event pages
- `gsoc_proposal` - For GSoC proposals

### File Naming Conventions

- **Pages**: Use hyphens, lowercase: `my-new-page.md`
- **Profiles**: Use format: `firstname_lastname.md`
- **Images**: Descriptive names: `feature-image-2026.png`
- **Dates**: Use ISO format in frontmatter or filenames: `2026-03-25`

### Directory Organization

```
Root pages/              → Main navigation items
_activities/             → Activity areas (GSoC, initiatives)
_workinggroups/          → Technical working groups
_training/               → Training schools and programs
_profiles/               → Community member profiles
_gsocorgs/YEAR/          → GSoC organization submissions
_gsocprojects/YEAR/      → GSoC project ideas
_gsocproposals/YEAR/     → GSoC student proposals
announcements/           → News and blog posts
events/                  → Event pages
newsletter/              → Newsletter content
```

## Specific Content Types

### Adding a Profile

1. Copy [_profiles/000_template.md](_profiles/000_template.md)
2. Rename to `_profiles/firstname_lastname.md`
3. Fill in all fields:
   ```yaml
   ---
   title: "Full Name"
   author: "Your Name"
   layout: educator
   institution: "Institution Name"
   ---
   ```
4. Add biography and highlights
5. Submit pull request

### Adding an Activity

1. Create `_activities/activity-name.md`
2. Use proper frontmatter:
   ```yaml
   ---
   title: "Activity Title"
   layout: default
   ---
   ```
3. Include description, contact info, and links
4. Keep similar format to existing activities

### Adding a Working Group

1. Create `_workinggroups/group-name.md`
2. Include leadership, scope, and meetings
3. Link to relevant resources and mailing lists
4. Add contact information

### Adding a News Item

1. Create `announcements/_posts/YYYY-MM-DD-slug.md`
2. Include proper date and title:
   ```yaml
   ---
   title: "Announcement Title"
   author: "Your Name"
   layout: default
   ---
   ```
3. Keep it concise and relevant

### Adding GSoC Content

**Organizations** (by year):
```
_gsocorgs/2026/organization-name.md
```

**Projects/Ideas** (by year):
```
_gsocprojects/2026/project-name.md
```

**Student Proposals** (by year):
```
_gsocproposals/2026/student-name-project-name.md
```

## Style Guide

### Markdown Format

```markdown
# Main Heading (H1)
## Subheading (H2)
### Smaller Heading (H3)

Regular paragraph text.

**Bold text** for emphasis
*Italic text* for subtle emphasis

- Bullet point
- Another point
  - Nested point

1. Numbered list
2. Second item
   1. Nested numbered

[Link text](URL)

> Blockquote for important notes

| Column 1 | Column 2 |
|----------|----------|
| Value    | Value    |
```

### Links

- **Internal**: Use relative paths: `[GSoC](activities/gsoc.html)`
- **External**: Use full URLs: `[GitHub](https://github.com)`
- **Profiles**: Link to `/profiles/firstname_lastname.html`
- **Working Groups**: Link to `/workinggroups/group-name.html`

### Images

```markdown
![Alt text for image](/images/filename.ext)

<!-- With caption -->
![Alt text](/images/image.png)
*Image caption*
```

**Best Practices:**
- Use descriptive alt text
- Keep file sizes reasonable
- Use appropriate formats (PNG for screenshots, JPG for photos)
- Store in `/images/` directory

## Code of Conduct

We follow the HumanAI Foundation code of conduct. When contributing:

- Be respectful and inclusive
- Give credit to original authors
- Provide constructive feedback
- Respect other people's work
- Report concerns to the coordination team

## Review Process

1. **Automatic Checks** - GitHub Actions runs HTML/Jekyll validation
2. **Team Review** - Maintainers review content and structure
3. **Approval** - Changes are approved or comments requested
4. **Merge** - Approved PRs are merged to `master`
5. **Deployment** - Site rebuilds automatically on GitHub Pages

### What We Look For

✅ Clear, well-written content
✅ Proper file organization
✅ Working links and formatting
✅ Relevant to the community
✅ No copyright violations

❌ Spam or promotional content
❌ Broken links or formatting
❌ Outdated information without updates
❌ Duplicate or conflicting content

## Common Issues & Solutions

### Build Fails Locally

```bash
# Update gems
bundle update

# Clean and rebuild
rm -rf _site .jekyll-metadata
bundle exec jekyll build
```

### Links Don't Work

- Ensure relative paths use `/` not `\`
- Check file actually exists
- Verify `layout` name is correct
- Test locally before submitting

### Images Don't Display

- Verify image file exists in `/images/`
- Use correct path: `/images/filename.ext`
- Test locally first
- Check file format is supported

### Content Not Showing

- Verify YAML frontmatter is correct
- Check `layout` value matches actual layout file
- Ensure file is in correct directory
- Look for Jekyll build errors in terminal

## Questions?

- **Report Issues**: [GitHub Issues](https://github.com/humanai-foundation/humanai-foundation.github.io/issues)
- **Ask Questions**: [HSF Forum](https://groups.google.com/forum/#!forum/hsf-forum)
- **Contact Team**: hsf-coordination@googlegroups.com

## Recognition

Contributors will be recognized in:
- Pull request discussions
- Site commit history
- Community profiles (if created)

Thank you for making this community better! 🎉
