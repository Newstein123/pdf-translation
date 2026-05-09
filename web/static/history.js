/* History page — full table of all translation jobs with cost + actions. */
(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const TOKEN_KEY = 'pdf-translation.token';
  const JOB_KEY = 'pdf-translation.currentJobId';

  function getToken() {
    return localStorage.getItem(TOKEN_KEY) || '';
  }

  // Bail early if no token saved — direct user back to home to set one.
  if (!getToken()) {
    $('auth-required').hidden = false;
    return;
  }
  $('filters').hidden = false;
  $('history-card').hidden = false;

  const state = {
    rows: [],          // every job row we've fetched, newest first
    pageSize: 50,
    skip: 0,
    hasMore: false,
    costsById: new Map(), // job_id → actual usd, lazily filled
    filterText: '',
    filterStatus: '',
  };

  async function api(path) {
    const res = await fetch(path, { headers: { Authorization: `Bearer ${getToken()}` } });
    if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
    return res.json();
  }

  function escapeHtml(s) {
    const d = document.createElement('div');
    d.textContent = s ?? '';
    return d.innerHTML;
  }

  function humanSize(n) {
    if (n == null) return '—';
    const u = ['B', 'KB', 'MB', 'GB'];
    let i = 0;
    while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
    return `${n.toFixed(i ? 1 : 0)} ${u[i]}`;
  }

  async function fetchPage() {
    // The current backend doesn't paginate by skip; we fetch up to 100 and
    // filter client-side. For local single-user use this is plenty.
    const limit = 100;
    const rows = await api(`/api/jobs?limit=${limit}`);
    state.rows = rows;
    state.hasMore = rows.length >= limit;
    $('load-more').hidden = !state.hasMore;
  }

  // Lazy-fetch the actual spend for any COMPLETED / TRANSLATING / etc rows.
  // The cost-estimate endpoint also returns the actual receipt summed from
  // chunk_translation_passes. We avoid hitting it for UPLOADED rows where
  // there'd be nothing to sum.
  async function fetchCostFor(jobId) {
    if (state.costsById.has(jobId)) return state.costsById.get(jobId);
    try {
      const e = await api(`/api/jobs/${jobId}/cost-estimate`);
      const usd = (e.actual && e.actual.usd) || 0;
      state.costsById.set(jobId, usd);
      return usd;
    } catch (_) {
      state.costsById.set(jobId, null);
      return null;
    }
  }

  function applyFilters(rows) {
    const t = state.filterText.toLowerCase();
    const s = state.filterStatus;
    return rows.filter(r => {
      if (s && r.status !== s) return false;
      if (t && !(r.original_filename || '').toLowerCase().includes(t)) return false;
      return true;
    });
  }

  function statePill(s) {
    return `<span class="pill pill-state-${s}">${s}</span>`;
  }

  async function render() {
    const tbody = $('history-table').querySelector('tbody');
    const filtered = applyFilters(state.rows);
    const summary = $('summary');
    summary.textContent = filtered.length === state.rows.length
      ? `${filtered.length} job(s)`
      : `${filtered.length} of ${state.rows.length} job(s)`;
    summary.className = 'status';
    tbody.innerHTML = '';
    if (!filtered.length) {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td colspan="8" class="hint" style="text-align:center;padding:1.4rem">No jobs match.</td>`;
      tbody.appendChild(tr);
      return;
    }

    const dateFmt = new Intl.DateTimeFormat(undefined, {
      year: '2-digit', month: 'short', day: '2-digit',
      hour: '2-digit', minute: '2-digit',
    });
    for (const r of filtered) {
      const created = dateFmt.format(new Date(r.created_at + 'Z'));
      const tr = document.createElement('tr');
      const showCompare = ['COMPLETED', 'TRANSLATING', 'POSTPROCESSING', 'REBUILDING'].includes(r.status);
      const showPdf = r.status === 'COMPLETED';
      tr.innerHTML = `
        <td class="col-file" title="${escapeHtml(r.original_filename)}\n${escapeHtml(r.id)}">${escapeHtml(r.original_filename)}</td>
        <td>${statePill(r.status)}</td>
        <td class="col-profile" title="${escapeHtml(r.style_profile_key || '')}">${escapeHtml(r.style_profile_key || '—')}</td>
        <td>${r.page_count ?? '—'}</td>
        <td>${r.total_tokens ? r.total_tokens.toLocaleString() : '—'}</td>
        <td class="col-cost" data-job="${r.id}">${r.total_tokens > 0 ? '…' : '—'}</td>
        <td title="${escapeHtml(new Date(r.created_at + 'Z').toLocaleString())}">${escapeHtml(created)}</td>
        <td class="col-actions">
          <span class="actions-row">
            <button data-open="${r.id}">Open</button>
            ${showCompare ? `<button class="secondary" data-compare="${r.id}">Compare</button>` : ''}
            ${showPdf ? `<button class="secondary" data-download="${r.id}">PDF</button>` : ''}
          </span>
        </td>`;
      tbody.appendChild(tr);
    }
    tbody.querySelectorAll('button[data-open]').forEach(btn => {
      btn.onclick = () => {
        // Open the home page with this job pre-attached.
        localStorage.setItem(JOB_KEY, btn.dataset.open);
        window.location.href = '/';
      };
    });
    tbody.querySelectorAll('button[data-compare]').forEach(btn => {
      btn.onclick = () => {
        window.location.href = `/compare?job_id=${encodeURIComponent(btn.dataset.compare)}`;
      };
    });
    tbody.querySelectorAll('button[data-download]').forEach(btn => {
      // ?inline=1 flips Content-Disposition to inline so the new tab
      // renders the PDF natively instead of downloading it. User can
      // still File→Save from the browser viewer if they want a local copy.
      btn.onclick = () => {
        const url = `/api/jobs/${encodeURIComponent(btn.dataset.download)}/outputs/pdf?inline=1&token=${encodeURIComponent(getToken())}`;
        window.open(url, '_blank');
      };
    });

    // Lazy-load actual cost cells.
    for (const cell of tbody.querySelectorAll('td.col-cost[data-job]')) {
      const jobId = cell.dataset.job;
      if (cell.textContent === '—') continue;
      fetchCostFor(jobId).then(usd => {
        cell.textContent = usd != null ? `$${usd.toFixed(2)}` : '—';
        cell.removeAttribute('data-job');  // avoid re-fetching on re-render
      });
    }
  }

  $('refresh').onclick = async () => {
    state.costsById.clear();
    await fetchPage();
    render();
  };
  $('filter-text').oninput = (e) => { state.filterText = e.target.value; render(); };
  $('filter-status').onchange = (e) => { state.filterStatus = e.target.value; render(); };

  // Initial load.
  fetchPage().then(render).catch(e => {
    $('summary').textContent = `failed to load history: ${e.message}`;
    $('summary').className = 'status err';
  });
})();
