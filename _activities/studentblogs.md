---
title: "Blogs"
author: "Emanuele Usai"
layout: default
---


<img src="/images/humanai.jpg" alt="HumanAI" width="200"/>

# Google Summer of Code blog posts

## Introduction

This is a collection of blog posts from GSoC students who worked with HumanAI.

<!-- Student Blogs Enhanced UI -->
<nav class="studentblogs-breadcrumb" aria-label="Breadcrumb">
  <ol>
    <li><a href="/">Home</a></li>
    <li><a href="/activities/">Activities</a></li>
    <li aria-current="page">Blogs</li>
  </ol>
</nav>

<div id="studentblogs-app" class="studentblogs">
  <section class="controls" aria-label="Blog filters and search">
    <div class="search-group">
      <label for="blog-search" class="sr-only">Search blogs</label>
      <input id="blog-search" type="search" placeholder="Search by title, author, or category" aria-label="Search blogs" />
    </div>
    <div class="filter-group">
      <label for="category-filter" class="sr-only">Filter by category</label>
      <select id="category-filter" aria-label="Filter by category">
        <option value="">All categories</option>
      </select>
    </div>
    <div class="sort-group">
      <label for="sort-select" class="sr-only">Sort posts</label>
      <select id="sort-select" aria-label="Sort posts">
        <option value="recent">Most recent</option>
        <option value="alpha">Title A–Z</option>
        <option value="author">Author A–Z</option>
      </select>
    </div>
  </section>

  <section id="blog-grid" class="grid" aria-live="polite" aria-busy="false"></section>

  <div class="pagination">
    <button id="load-more" class="load-more" type="button" aria-label="Load more posts">Load More</button>
  </div>
</div>

<noscript>
  <ul>
    <li><a href="https://medium.com/@domenicolacavalla8/examination-of-the-evolution-of-language-among-dark-web-users-67fd3397e0fb" target="_blank">Examination of the evolution of language among Dark Web users — Domenico Lacavalla</a></li>
    <li><a href="https://medium.com/@yamanko1234/historical-ocr-with-self-supervised-learning-c4f00da6637f" target="_blank">RenAIssance — Yukinori Yamamoto</a></li>
    <li><a href="https://medium.com/@shashankshekharsingh1205/my-journey-with-humanai-in-the-google-summer-of-code24-program-part-2-bb42abce3495" target="_blank">Historical Text Recognition using CRNN Model — Shashank Shekhar Singh</a></li>
    <li><a href="https://medium.com/@soyoungpark.psy/how-i-designed-hidden-art-extraction-tool-with-siamese-networks-part4-gsoc-24-e3387b3ae50b" target="_blank">ArtExtract — Soyoung Park</a></li>
    <li><a href="https://medium.com/@luisvz/duet-choreaigraphy-dance-meets-ai-again-part-2-b8f459a0e3d6" target="_blank">ChoreoAI — Luis Zerkowski</a></li>
    <li><a href="https://wang-zixuan.github.io/posts/2024/gsoc_2024" target="_blank">AI-Generated Choreography - from Solos to Duets — Zixuan Wang</a></li>
    <li><a href="https://medium.com/@rashiguptaofficial/exploring-gender-roles-in-education-a-grade-wise-analysis-cb87db14bc7d" target="_blank">Gender, Roles & Careers: Exploring Congruity Theories — Rashi Gupta</a></li>
    <li><a href="https://medium.com/@sj3192/18c818d77527" target="_blank">Enhancing Program Evaluation Research by Leveraging AI for Integrated Analysis of Mixed-Methods Data — Shao Jin</a></li>
    <li><a href="https://medium.com/@aditya.arvind97/fatigue-detection-and-driver-distraction-monitoring-b895a5ee287c" target="_blank">Fatigue detection — Aditya Arvind</a></li>
    <li><a href="https://medium.com/@khanarsh0124/gsoc-2024-with-humanai-text-recognition-with-transformer-models-de86522cdc17" target="_blank">Text Recognition using Transformer Models — Arsh Ahmed Faisal Khan</a></li>
    <li><a href="https://utsavrai.substack.com/p/decoding-history-advancing-text-recognition" target="_blank">Decoding History: Advancing Text Recognition — Utsav Rai</a></li>
  </ul>
</noscript>

<style>
  /* Scoped styles for Student Blogs */
  .studentblogs {
    --brand-bg: #0b1e3b;
    --brand-primary: #2e79ff;
    --brand-accent: #ff5a7a;
    --brand-surface: #ffffff;
    --text-color: #0d121c;
    --muted-color: #5c667a;
    --border-color: #e6e9f0;
    --card-shadow: 0 8px 24px rgba(0,0,0,0.08);
    --radius-lg: 16px;
    --radius-md: 12px;
    --radius-sm: 10px;
  }
  .studentblogs-breadcrumb {
    margin: 1rem 0 1.25rem;
    font-size: 0.875rem;
  }
  .studentblogs-breadcrumb ol {
    list-style: none;
    padding: 0;
    margin: 0;
    display: flex;
    gap: 0.5rem;
    color: var(--muted-color);
  }
  .studentblogs-breadcrumb a {
    color: var(--muted-color);
    text-decoration: none;
  }
  .studentblogs-breadcrumb li+li::before {
    content: "→";
    margin: 0 0.5rem;
    color: var(--muted-color);
  }

  .studentblogs .controls {
    display: grid;
    grid-template-columns: 1fr;
    gap: 0.75rem;
    margin: 1rem 0 1.5rem;
  }
  .studentblogs .controls input,
  .studentblogs .controls select {
    width: 100%;
    border: 1px solid var(--border-color);
    border-radius: var(--radius-sm);
    padding: 0.625rem 0.75rem;
    font-size: 0.95rem;
    background: var(--brand-surface);
    color: var(--text-color);
  }
  @media (min-width: 720px) {
    .studentblogs .controls { grid-template-columns: 2fr 1fr 1fr; }
  }

  .studentblogs .grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 1rem;
  }
  @media (min-width: 540px) { .studentblogs .grid { grid-template-columns: repeat(2, 1fr); } }
  @media (min-width: 960px) { .studentblogs .grid { grid-template-columns: repeat(3, 1fr); } }

  .studentblogs .card {
    display: flex;
    flex-direction: column;
    background: var(--brand-surface);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg);
    overflow: hidden;
    box-shadow: var(--card-shadow);
    transition: transform 200ms ease, box-shadow 200ms ease;
  }
  .studentblogs .card:focus-within,
  .studentblogs .card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 28px rgba(0,0,0,0.12);
  }
  .studentblogs .thumb {
    position: relative;
    aspect-ratio: 16 / 9;
    background: linear-gradient(135deg, rgba(46,121,255,0.12), rgba(255,90,122,0.12));
  }
  .studentblogs .thumb img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .studentblogs .badge-new {
    position: absolute;
    top: 10px;
    left: 10px;
    background: var(--brand-accent);
    color: #fff;
    padding: 4px 8px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
  }
  .studentblogs .content { padding: 0.875rem 1rem 1rem; display: grid; gap: 0.5rem; }
  .studentblogs .title {
    font-size: 1.05rem;
    line-height: 1.35;
    margin: 0;
  }
  .studentblogs .title a { color: var(--text-color); text-decoration: none; }
  .studentblogs .title a:hover { color: var(--brand-primary); text-decoration: underline; }
  .studentblogs .meta { display: flex; flex-wrap: wrap; gap: 0.5rem 0.75rem; color: var(--muted-color); font-size: 0.85rem; }
  .studentblogs .meta a { color: var(--muted-color); text-decoration: none; }
  .studentblogs .meta a:hover { color: var(--brand-primary); text-decoration: underline; }
  .studentblogs .excerpt {
    color: var(--text-color);
    font-size: 0.95rem;
    line-height: 1.5;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    min-height: 3.0em;
  }
  .studentblogs .read-more { color: var(--brand-primary); text-decoration: none; font-weight: 600; }
  .studentblogs .read-more:hover { text-decoration: underline; }

  .studentblogs .pagination { display: flex; justify-content: center; margin: 1.25rem 0; }
  .studentblogs .load-more {
    background: var(--brand-primary);
    color: #fff;
    border: none;
    padding: 0.6rem 1rem;
    border-radius: 999px;
    cursor: pointer;
    font-weight: 600;
  }
  .studentblogs .load-more:hover { filter: brightness(0.95); }

  .sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0; }
</style>

<script>
  (function() {
    const BRAND = {
      bg: getComputedStyle(document.documentElement).getPropertyValue('--brand-bg') || '#0b1e3b',
      primary: getComputedStyle(document.documentElement).getPropertyValue('--brand-primary') || '#2e79ff',
      accent: getComputedStyle(document.documentElement).getPropertyValue('--brand-accent') || '#ff5a7a'
    };

    const posts = [
      {
        title: "Examination of the evolution of language among Dark Web users",
        url: "https://medium.com/@domenicolacavalla8/examination-of-the-evolution-of-language-among-dark-web-users-67fd3397e0fb",
        author: "Domenico Lacavalla",
        authorUrl: "https://medium.com/@domenicolacavalla8",
        year: 2024,
        categories: ["NLP", "Social Analysis"],
        excerpt: "Exploring language evolution among Dark Web communities using NLP.",
        thumbnail: null
      },
      {
        title: "RenAIssance",
        url: "https://medium.com/@yamanko1234/historical-ocr-with-self-supervised-learning-c4f00da6637f",
        author: "Yukinori Yamamoto",
        authorUrl: "https://medium.com/@yamanko1234",
        year: 2024,
        categories: ["Text Recognition", "Self-Supervised"],
        excerpt: "Historical OCR with self-supervised learning approaches.",
        thumbnail: null
      },
      {
        title: "Historical Text Recognition using CRNN Model",
        url: "https://medium.com/@shashankshekharsingh1205/my-journey-with-humanai-in-the-google-summer-of-code24-program-part-2-bb42abce3495",
        author: "Shashank Shekhar Singh",
        authorUrl: "https://medium.com/@shashankshekharsingh1205",
        year: 2024,
        categories: ["Text Recognition", "CRNN"],
        excerpt: "CRNN-based pipeline for historical document text recognition.",
        thumbnail: null
      },
      {
        title: "ArtExtract",
        url: "https://medium.com/@soyoungpark.psy/how-i-designed-hidden-art-extraction-tool-with-siamese-networks-part4-gsoc-24-e3387b3ae50b",
        author: "Soyoung Park",
        authorUrl: "https://medium.com/@soyoungpark.psy",
        year: 2024,
        categories: ["Computer Vision", "Siamese Networks"],
        excerpt: "Designing a hidden art extraction tool with Siamese networks.",
        thumbnail: null
      },
      {
        title: "ChoreoAI",
        url: "https://medium.com/@luisvz/duet-choreaigraphy-dance-meets-ai-again-part-2-b8f459a0e3d6",
        author: "Luis Zerkowski",
        authorUrl: "https://medium.com/@luisvz",
        year: 2024,
        categories: ["Choreography AI", "Generative"],
        excerpt: "Dance meets AI again: duet choreography generation.",
        thumbnail: null
      },
      {
        title: "AI-Generated Choreography - from Solos to Duets",
        url: "https://wang-zixuan.github.io/posts/2024/gsoc_2024",
        author: "Zixuan Wang",
        authorUrl: "https://wang-zixuan.github.io",
        year: 2024,
        categories: ["Choreography AI", "Motion"],
        excerpt: "From solo routines to duets: generating choreography with AI.",
        thumbnail: null
      },
      {
        title: "Gender, Roles & Careers: Exploring Congruity Theories",
        url: "https://medium.com/@rashiguptaofficial/exploring-gender-roles-in-education-a-grade-wise-analysis-cb87db14bc7d",
        author: "Rashi Gupta",
        authorUrl: "https://medium.com/@rashiguptaofficial",
        year: 2024,
        categories: ["Social Science", "Education"],
        excerpt: "Grade-wise analysis of gender roles in education using AI insights.",
        thumbnail: null
      },
      {
        title: "Enhancing Program Evaluation Research by Leveraging AI for Integrated Analysis of Mixed-Methods Data",
        url: "https://medium.com/@sj3192/18c818d77527",
        author: "Shao Jin",
        authorUrl: "https://medium.com/@sj3192",
        year: 2024,
        categories: ["Program Evaluation", "Mixed Methods"],
        excerpt: "Integrating AI into mixed-methods program evaluation research.",
        thumbnail: null
      },
      {
        title: "Fatigue detection",
        url: "https://medium.com/@aditya.arvind97/fatigue-detection-and-driver-distraction-monitoring-b895a5ee287c",
        author: "Aditya Arvind",
        authorUrl: "https://medium.com/@aditya.arvind97",
        year: 2024,
        categories: ["Computer Vision", "Safety"],
        excerpt: "Fatigue detection and driver distraction monitoring with AI.",
        thumbnail: null
      },
      {
        title: "Text Recognition using Transformer Models",
        url: "https://medium.com/@khanarsh0124/gsoc-2024-with-humanai-text-recognition-with-transformer-models-de86522cdc17",
        author: "Arsh Ahmed Faisal Khan",
        authorUrl: "https://medium.com/@khanarsh0124",
        year: 2024,
        categories: ["Text Recognition", "Transformers"],
        excerpt: "Transformer-based models for historical text recognition.",
        thumbnail: null
      },
      {
        title: "Decoding History: Advancing Text Recognition",
        url: "https://utsavrai.substack.com/p/decoding-history-advancing-text-recognition",
        author: "Utsav Rai",
        authorUrl: "https://utsavrai.substack.com/",
        year: 2024,
        categories: ["Text Recognition", "Research"],
        excerpt: "Advancing text recognition to decode historical documents.",
        thumbnail: null
      }
    ];

    const state = {
      pageSize: 6,
      rendered: 0,
      query: '',
      category: '',
      sort: 'recent'
    };

    const app = document.getElementById('studentblogs-app');
    const grid = document.getElementById('blog-grid');
    const search = document.getElementById('blog-search');
    const categoryFilter = document.getElementById('category-filter');
    const sortSelect = document.getElementById('sort-select');
    const loadMoreBtn = document.getElementById('load-more');

    function uniqueCategories() {
      const set = new Set();
      posts.forEach(p => p.categories && p.categories.forEach(c => set.add(c)));
      return Array.from(set).sort();
    }

    function buildCategoryFilter() {
      uniqueCategories().forEach(cat => {
        const opt = document.createElement('option');
        opt.value = cat; opt.textContent = cat;
        categoryFilter.appendChild(opt);
      });
    }

    function matchesQuery(post, query) {
      if (!query) return true;
      const q = query.toLowerCase();
      return [post.title, post.author, (post.categories || []).join(' ')].some(v => (v || '').toLowerCase().includes(q));
    }

    function matchesCategory(post, category) {
      if (!category) return true;
      return (post.categories || []).includes(category);
    }

    function sortPosts(list, mode) {
      const copy = list.slice();
      if (mode === 'alpha') {
        copy.sort((a,b) => a.title.localeCompare(b.title));
      } else if (mode === 'author') {
        copy.sort((a,b) => a.author.localeCompare(b.author));
      } else {
        copy.sort((a,b) => (b.year||0) - (a.year||0));
      }
      return copy;
    }

    function isNew(post) {
      return false; // No precise dates available; disable until dates are added
    }

    function svgPlaceholder(titleText) {
      const text = (titleText || 'AI').slice(0, 1).toUpperCase();
      const svg = `<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360" viewBox="0 0 640 360" role="img" aria-label="Placeholder thumbnail">\n  <defs>\n    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">\n      <stop offset="0%" stop-color="${BRAND.primary}" stop-opacity="0.9"/>\n      <stop offset="100%" stop-color="${BRAND.accent}" stop-opacity="0.9"/>\n    </linearGradient>\n  </defs>\n  <rect width="640" height="360" fill="url(#g)"/>\n  <text x="50%" y="54%" text-anchor="middle" fill="#ffffff" font-family="Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif" font-size="200" font-weight="800">${text}</text>\n</svg>`;
      return 'data:image/svg+xml;charset=UTF-8,' + encodeURIComponent(svg);
    }

    function renderCard(post) {
      const article = document.createElement('article');
      article.className = 'card';
      article.setAttribute('itemscope', '');
      article.setAttribute('itemtype', 'https://schema.org/BlogPosting');

      const thumb = document.createElement('div');
      thumb.className = 'thumb';
      const img = document.createElement('img');
      img.loading = 'lazy';
      img.width = 640; img.height = 360;
      img.alt = `${post.title} thumbnail`;
      img.src = post.thumbnail || svgPlaceholder(post.title);
      thumb.appendChild(img);
      if (isNew(post)) {
        const badge = document.createElement('div');
        badge.className = 'badge-new';
        badge.textContent = 'New';
        thumb.appendChild(badge);
      }

      const content = document.createElement('div');
      content.className = 'content';

      const h2 = document.createElement('h2');
      h2.className = 'title';
      const a = document.createElement('a');
      a.href = post.url; a.target = '_blank'; a.rel = 'noopener';
      a.textContent = post.title;
      a.setAttribute('itemprop', 'headline');
      a.setAttribute('aria-label', `Read ${post.title}`);
      h2.appendChild(a);

      const meta = document.createElement('div');
      meta.className = 'meta';
      const by = document.createElement('span');
      by.innerHTML = `by <a href="${post.authorUrl}" target="_blank" rel="noopener" itemprop="author" itemscope itemtype="https://schema.org/Person"><span itemprop="name">${post.author}</span></a>`;
      const year = document.createElement('span');
      year.textContent = post.year ? String(post.year) : '';
      const cats = document.createElement('span');
      cats.textContent = (post.categories || []).join(' • ');
      meta.appendChild(by);
      if (post.year) meta.appendChild(year);
      if (post.categories && post.categories.length) meta.appendChild(cats);

      const excerpt = document.createElement('p');
      excerpt.className = 'excerpt';
      excerpt.setAttribute('itemprop', 'description');
      excerpt.textContent = post.excerpt || '';

      const cta = document.createElement('a');
      cta.className = 'read-more';
      cta.href = post.url; cta.target = '_blank'; cta.rel = 'noopener';
      cta.setAttribute('aria-label', `Read more: ${post.title}`);
      cta.textContent = 'Read More →';

      content.appendChild(h2);
      content.appendChild(meta);
      content.appendChild(excerpt);
      content.appendChild(cta);

      article.appendChild(thumb);
      article.appendChild(content);

      // microdata URL
      const linkMeta = document.createElement('meta');
      linkMeta.setAttribute('itemprop', 'mainEntityOfPage');
      linkMeta.content = post.url;
      article.appendChild(linkMeta);

      return article;
    }

    function applyFilters() {
      const filtered = posts.filter(p => matchesQuery(p, state.query) && matchesCategory(p, state.category));
      return sortPosts(filtered, state.sort);
    }

    function render(reset = false) {
      grid.setAttribute('aria-busy', 'true');
      if (reset) { grid.innerHTML = ''; state.rendered = 0; }
      const list = applyFilters();
      const next = list.slice(state.rendered, state.rendered + state.pageSize);
      next.forEach(p => grid.appendChild(renderCard(p)));
      state.rendered += next.length;
      loadMoreBtn.style.display = state.rendered < list.length ? 'inline-flex' : 'none';
      grid.setAttribute('aria-busy', 'false');
    }

    function attachEvents() {
      search.addEventListener('input', (e) => { state.query = e.target.value || ''; render(true); });
      categoryFilter.addEventListener('change', (e) => { state.category = e.target.value || ''; render(true); });
      sortSelect.addEventListener('change', (e) => { state.sort = e.target.value || 'recent'; render(true); });
      loadMoreBtn.addEventListener('click', () => render(false));
    }

    function init() {
      buildCategoryFilter();
      attachEvents();
      render(true);
      addJsonLd();
    }

    function addJsonLd() {
      try {
        const data = {
          '@context': 'https://schema.org',
          '@type': 'Blog',
          'name': 'HumanAI GSoC Blogs',
          'url': location.href,
          'blogPost': posts.map(p => ({
            '@type': 'BlogPosting',
            'headline': p.title,
            'author': { '@type': 'Person', 'name': p.author, 'url': p.authorUrl },
            'url': p.url,
            'datePublished': p.year ? `${p.year}-01-01` : undefined,
            'description': p.excerpt
          }))
        };
        const script = document.createElement('script');
        script.type = 'application/ld+json';
        script.textContent = JSON.stringify(data);
        document.body.appendChild(script);
      } catch (_) {}
    }

    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', init);
    } else { init(); }
  })();
</script>


## Contacts

*HumanAI GSoC Admins* [human-ai@cern.ch](mailto:human-ai@cern.ch)

