/* Compare page — side-by-side source vs translation, paginated by source page. */
(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const TOKEN_KEY = 'pdf-translation.token';

  function getToken() { return localStorage.getItem(TOKEN_KEY) || ''; }
  function escapeHtml(s) {
    const d = document.createElement('div');
    d.textContent = s ?? '';
    return d.innerHTML;
  }

  if (!getToken()) {
    $('auth-required').hidden = false;
    return;
  }

  const params = new URLSearchParams(window.location.search);
  const jobId = params.get('job_id');
  if (!jobId) {
    $('missing-job').hidden = false;
    return;
  }

  $('status-card').hidden = false;

  async function api(path) {
    const res = await fetch(path, { headers: { Authorization: `Bearer ${getToken()}` } });
    if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
    return res.json();
  }

  const state = {
    pages: [],   // CompareResponse.pages
    pageIdx: 0,  // 0-based current page index
    syncScroll: true,
  };

  async function load() {
    try {
      const data = await api(`/api/jobs/${encodeURIComponent(jobId)}/compare`);
      state.pages = data.pages || [];
      $('page-title').textContent = data.original_filename || 'Compare';
      const sub = data.blocks_total
        ? `${data.blocks_translated} of ${data.blocks_total} blocks translated · ${data.pages_total} page(s)`
        : 'no translatable text yet';
      $('page-sub').textContent = sub;

      if (!state.pages.length) {
        $('status-line').textContent =
          'No translation available yet. Translation may still be running — refresh once it finishes.';
        $('status-line').className = 'status warn';
        return;
      }

      $('status-card').hidden = true;
      $('page-controls').hidden = false;
      $('compare-grid').hidden = false;
      $('page-jump').max = String(state.pages.length);
      renderPage();
    } catch (e) {
      $('status-line').textContent = e.message;
      $('status-line').className = 'status err';
    }
  }

  function renderPage() {
    const page = state.pages[state.pageIdx];
    if (!page) return;
    const src = $('src-pane');
    const tgt = $('tgt-pane');
    src.innerHTML = '';
    tgt.innerHTML = '';

    for (const block of page.blocks) {
      const tag = block.type === 'heading' ? 'h2' : 'p';
      const klass = block.type === 'heading' ? 'cmp-heading' : 'cmp-block';

      const srcEl = document.createElement(tag);
      srcEl.className = klass;
      srcEl.dataset.blockId = block.block_id;
      srcEl.textContent = block.source;
      src.appendChild(srcEl);

      const tgtEl = document.createElement(tag);
      tgtEl.className = klass + (block.translated ? '' : ' cmp-untranslated');
      tgtEl.dataset.blockId = block.block_id;
      tgtEl.textContent = block.translated || '(not translated yet)';
      tgt.appendChild(tgtEl);

      // Cross-pane hover sync — highlight the matching block in the other pane.
      const onEnter = () => {
        srcEl.classList.add('cmp-active');
        tgtEl.classList.add('cmp-active');
      };
      const onLeave = () => {
        srcEl.classList.remove('cmp-active');
        tgtEl.classList.remove('cmp-active');
      };
      srcEl.addEventListener('mouseenter', onEnter);
      srcEl.addEventListener('mouseleave', onLeave);
      tgtEl.addEventListener('mouseenter', onEnter);
      tgtEl.addEventListener('mouseleave', onLeave);
    }

    $('page-indicator').textContent = `Page ${page.page_number} (${state.pageIdx + 1} of ${state.pages.length})`;
    $('prev-page').disabled = state.pageIdx === 0;
    $('next-page').disabled = state.pageIdx >= state.pages.length - 1;
    $('page-jump').value = page.page_number;

    // Reset scroll position when changing page so the user starts at the top.
    src.scrollTop = 0;
    tgt.scrollTop = 0;
  }

  // Sync-scroll: when one pane scrolls, mirror the proportional scroll
  // position on the other. We avoid the ping-pong loop with a guard flag.
  let scrollGuard = false;
  function attachScrollSync() {
    const src = $('src-pane');
    const tgt = $('tgt-pane');
    function mirror(from, to) {
      if (!state.syncScroll || scrollGuard) return;
      scrollGuard = true;
      const ratio = from.scrollTop / Math.max(1, from.scrollHeight - from.clientHeight);
      to.scrollTop = ratio * Math.max(0, to.scrollHeight - to.clientHeight);
      requestAnimationFrame(() => { scrollGuard = false; });
    }
    src.addEventListener('scroll', () => mirror(src, tgt));
    tgt.addEventListener('scroll', () => mirror(tgt, src));
  }

  $('prev-page').onclick = () => {
    if (state.pageIdx > 0) { state.pageIdx -= 1; renderPage(); }
  };
  $('next-page').onclick = () => {
    if (state.pageIdx < state.pages.length - 1) { state.pageIdx += 1; renderPage(); }
  };
  $('page-jump-go').onclick = () => {
    const target = parseInt($('page-jump').value, 10);
    if (!target) return;
    const idx = state.pages.findIndex(p => p.page_number === target);
    if (idx >= 0) { state.pageIdx = idx; renderPage(); }
  };
  $('page-jump').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') $('page-jump-go').click();
  });
  $('sync-scroll').onchange = (e) => { state.syncScroll = e.target.checked; };

  // Keyboard nav: ← / → to flip pages while focus isn't in an input.
  document.addEventListener('keydown', (e) => {
    if (e.target.matches('input, textarea, select')) return;
    if (e.key === 'ArrowLeft') $('prev-page').click();
    else if (e.key === 'ArrowRight') $('next-page').click();
  });

  attachScrollSync();
  load();
})();
