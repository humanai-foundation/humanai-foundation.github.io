# Repository Improvement Suggestions

This document outlines recommended improvements for the HumanAI Foundation website repository.

## Priority Issues (Fix First)

### 1. ⚠️ Update Site Metadata & Branding
**Status**: Critical  
**Issue**: Repository and site branding inconsistencies (ML4SCI vs HSF vs HumanAI)

**Action Items**:
- [ ] Update `_config.yml` to reflect current organization name/focus
- [ ] Review all markdown files for outdated references
- [ ] Standardize domain/URL throughout documentation
- [ ] Update footer links and organization branding
- [ ] Review `CNAME` file for correct domain

**Files to Check**:
- `_config.yml` - Description says "HEP Software Foundation"
- `index.html` - Says "HumanAI"
- Multiple markdown files reference different organizations

### 2. 🔗 Fix Broken/Outdated Links
**Status**: High  
**Issue**: Many links may be outdated

**Action Items**:
- [ ] Run link checker: `bundle exec htmlproofer ./_site/`
- [ ] Update external links (CERN Indico, forum links)
- [ ] Fix internal cross-references
- [ ] Update email addresses in contact sections
- [ ] Verify all collection links resolve correctly

**Files to Check**:
- `get_involved.md` - Check forum and meeting links
- `forums.md` - Verify all communication channels
- Layout files - Check footer links

### 3. 📝 Add Missing Documentation
**Status**: Done (Created)  
**Files Added**:
- ✅ Updated README.md - Comprehensive guide
- ✅ Created CONTRIBUTING.md - Contribution guidelines
- ✅ Created STRUCTURE.md - Directory structure guide
- [ ] Create DEPLOYMENT.md - How the site is deployed
- [ ] Create DEVELOPMENT.md - Developer workflow
- [ ] Create FAQ.md - Common questions

### 4. 🗂️ Organize Legacy/Archived Content
**Status**: Medium  
**Issue**: Old content mixed with current content

**Suggested Actions**:
- [ ] Move pre-2023 GSoC years to `_gsocprojects/archived/` and `_gsocproposals/archived/`
- [ ] Document deprecation of `_gsdocs-*` directories
- [ ] Archive old announcements (move to `archive/` subdirectory)
- [ ] Clean up `_drafts/` - delete or publish
- [ ] Document which sections are "for historical reference only"

**Directory Cleanup**:
```
_gsocprojects/
├── 2023/            → Consider archiving
├── 2024/
├── 2025/
├── 2026/
└── archived/        ← Move old years here

_gsdocs-*/          → Mark as deprecated/legacy
```

## Medium Priority Improvements

### 5. 🏗️ Improve Directory Structure
**Status**: Medium  
**Current Issues**:
- GSoC has 3 separate collections (_gsocorgs, _gsocprojects, _gsocproposals) organized by year
- Unclear purpose of similar-named directories
- Some directories have ambiguous names

**Options for Future Reorganization**:
```
Option A: Merge related GSoC content by year
├── _gsoc/
│   ├── 2024/
│   │   ├── organizations/
│   │   ├── projects/
│   │   └── proposals/
│   └── 2025/

Option B: Keep current structure but add docs
├── _gsocorgs-ORGS/          ← Rename for clarity
├── _gsocprojects-IDEAS/     ← Rename for clarity
└── _gsocproposals-SUBS/     ← Rename for clarity
```

**Recommendation**: Keep current structure (breaking changes risky) but add comprehensive documentation (✅ Done via STRUCTURE.md)

### 6. 🚀 Improve Development Workflow
**Status**: Medium  
**Missing**:
- [ ] GitHub Actions/CI workflows documented
- [ ] Pre-commit hooks configured
- [ ] Automated link checking in CI
- [ ] Spell checking automation
- [ ] HTML validation in build

**Add to CI/CD**:
```yaml
# .github/workflows/jekyll-build.yml
- Run htmlproofer on output
- Check for broken links
- Validate YAML frontmatter
- Spell checking (optional)
```

**Files to Create**:
- `.github/workflows/build-and-test.yml`
- `.github/workflows/deploy.yml`

### 7. 📊 Add Content Audit
**Status**: Medium  
**Suggested Audit**:
- [ ] Review all profiles - remove inactive members or mark as emeritus
- [ ] Update old announcements to show publication date
- [ ] Check all event pages - remove past events
- [ ] Verify all external links still work
- [ ] Update "last updated" dates where relevant

**Script Needed**:
```python
# scripts/audit_content.py
- Find profiles without updates in 2+ years
- Identify broken links
- List pages with old frontmatter
- Report outdated event dates
```

### 8. 🎨 Improve Visual Branding
**Status**: Low  
**Issues**:
- [ ] Logo/favicon could be clearer
- [ ] Navigation could be more intuitive
- [ ] Mobile responsiveness check
- [ ] Color scheme consistency

**Suggestions**:
- Update navbar to show more dynamic links
- Add breadcrumb navigation
- Improve mobile menu
- Better visual hierarchy for collections

## Low Priority Enhancements

### 9. 🔍 SEO & Discoverability
- [ ] Add meta descriptions to all pages
- [ ] Improve page titles for clarity
- [ ] Add structured data (schema.org)
- [ ] Update sitemap.xml
- [ ] Add robots.txt if needed

### 10. 📱 Mobile & Accessibility
- [ ] Test on mobile devices
- [ ] Improve accessibility (WCAG compliance)
- [ ] Add alt text to all images
- [ ] Test with screen readers
- [ ] Improve keyboard navigation

### 11. 🔄 Content Templates
**Create Template Files**:
- [ ] `_profiles/000_template.md` - (exists, could be enhanced)
- [ ] `_activities/template.md` - Activity template
- [ ] `_workinggroups/template.md` - Working group template
- [ ] `announcements/_posts/template.md` - Announcement template
- [ ] `events/template.md` - Event template

### 12. 📚 Knowledge Base
- [ ] Create FAQ.md with common questions
- [ ] Document Jekyll customizations
- [ ] Document Python scripts usage
- [ ] Add troubleshooting guide
- [ ] Create quick-start for contributors

## Script & Automation Improvements

### 13. Enhance Utility Scripts
**Current Scripts**:
- `scripts/add_training_event.py` - Add training events
- `scripts/profile_maintenance_script.py` - Manage profiles

**Recommendations**:
- [ ] Add error handling and validation
- [ ] Document script usage
- [ ] Create `scripts/validate_content.py`
- [ ] Create `scripts/generate_indexes.py`
- [ ] Add `scripts/check_links.py`
- [ ] Create `scripts/archive_old_gsoc_years.py`

### 14. Documentation Generation
```bash
# New script ideas
scripts/
├── validate_frontmatter.py      # Check all YAML frontmatter
├── validate_links.py            # Check internal/external links
├── generate_sitemap.py          # Generate sitemap
├── report_outdated.py           # Find old/unmaintained content
├── bulk_update_profiles.py      # Batch profile updates
└── migrate_content.py           # Migration utilities
```

## Configuration Improvements

### 15. Update _config.yml
**Current Issues**:
- Description is outdated (mentions HEP Software Foundation but site is HumanAI)
- Collections could have better descriptions
- Could add more metadata

**Suggested Updates**:
```yaml
# More accurate description
description: "HumanAI Foundation community hub with working groups, GSoC, training, and projects"

# Add collection descriptions
collections:
  activities:
    output: true
    description: "Interest groups and initiatives"
    permalink: /activities/:title.html
  # ... etc
```

### 16. Add Pre-commit Hooks
**File**: `.pre-commit-config.yaml` (exists but not used)

**Setup**:
```bash
pip install pre-commit
pre-commit install
```

**Add Hooks**:
- YAML validator
- Markdown linter
- Check for broken links
- Spell checker

## Migration & Backward Compatibility

### 17. Deprecation Plan
**For Future Changes**:
- Add deprecation notices to pages being changed
- Create redirects for renamed pages
- Keep old URLs working with `jekyll-redirect-from`
- Document breaking changes
- Maintain changelog

**Example Deprecation Notice**:
```markdown
> ⚠️ **Deprecated**: This page is deprecated as of March 2026.
> Please see [New Location](/new-page.html) instead.
```

## Testing & Quality Assurance

### 18. Add Testing
```bash
# Add to build process
bundle exec jekyll build
bundle exec htmlproofer ./_site/ --allow-hash-href --disable-external

# For future: Add tests for
- Link validity
- YAML frontmatter structure
- Image references
- Collection definitions
```

## Documentation Roadmap

### Already Created (✅)
- Updated main README.md
- Created CONTRIBUTING.md
- Created STRUCTURE.md
- Created this SUGGESTIONS.md

### To Create (📋)
1. DEPLOYMENT.md - How/where site is deployed
2. DEVELOPMENT.md - Developer setup & workflow
3. FAQ.md - Common questions and answers
4. MAINTENANCE.md - Tasks for maintainers
5. CHANGELOG.md - Track major changes
6. ROADMAP.md - Future plans

## Implementation Timeline

### Immediate (This Week)
- ✅ Update README
- ✅ Create CONTRIBUTING.md
- ✅ Create STRUCTURE.md
- [ ] Fix critical broken links
- [ ] Update site metadata in _config.yml

### Short Term (This Month)
- [ ] Create additional documentation (DEPLOYMENT.md, etc.)
- [ ] Organize legacy GSoC content
- [ ] Clean up _drafts directory
- [ ] Run full link audit

### Medium Term (This Quarter)
- [ ] Implement CI/CD improvements
- [ ] Update mobile/accessibility
- [ ] Create content templates
- [ ] Enhance Python scripts

### Long Term (This Year)
- [ ] Major directory restructuring (if needed)
- [ ] Full content audit
- [ ] SEO improvements
- [ ] Complete migration of legacy systems

## Success Metrics

You'll know these improvements are working when:

✅ New contributors can set up locally in under 5 minutes
✅ All links are valid and working
✅ Site metadata is consistent and accurate
✅ Documentation is comprehensive and up-to-date
✅ Contribution process is clear and documented
✅ Build succeeds with no warnings
✅ Mobile experience is good
✅ Outdated content is clearly marked or archived

## Questions & Next Steps

1. **What's the priority**? - Focus on metadata/links first
2. **Who maintains this**? - Document maintainers in MAINTENANCE.md
3. **What should be archived**? - Plan cleanup timeline
4. **How often do GSoC cycles occur**? - Plan automated archival
5. **What's the deployment process**? - Document in DEPLOYMENT.md

---

**Created**: March 25, 2026
**Status**: Recommendations for review and implementation
**Next Review**: After first set of improvements implemented
