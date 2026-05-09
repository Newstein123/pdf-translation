/* Minimal single-page client for the PDF→Burmese translator. */
(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const TOKEN_KEY = 'pdf-translation.token';
  const JOB_KEY = 'pdf-translation.currentJobId';

  // Job state — declared before refreshTokenUI() runs on script load so the
  // re-attach branch (`if (savedJob && !currentJobId)`) doesn't hit the
  // temporal dead zone for `currentJobId` when JOB_KEY is set in localStorage.
  let currentJobId = null;
  let eventSource = null;
  // Tracks whether the suggestions table has been populated for the current
  // job. Without this guard, the SSE 'status' poll (~2s) re-renders the
  // table and wipes in-flight edits to the Burmese term cells.
  let suggestionsLoaded = false;
  // Tracks whether the cost estimate has been fetched once for this job.
  let costLoaded = false;
  // M14 — same shape as suggestionsLoaded: load metadata form fields once
  // per job attach, then leave alone so user edits don't get clobbered by
  // status polls. ``post`` form gates separately because it lives in the
  // done-block which only appears once status flips to COMPLETED.
  let metadataLoadedPre = false;
  let metadataLoadedPost = false;
  // M14 — cache-buster bumped after re-export so the preview iframe and
  // Outputs download links don't serve the browser-cached version of the
  // previous render. Stamped onto every output URL as ``&v=N``.
  let outputVersion = Date.now();
  // Suggester state for the current job — drives the suggestions-panel banner
  // and gates the Start button. 'unknown' on attach; transitions on SSE.
  let suggesterStatus = 'unknown';  // 'unknown' | 'running' | 'ready' | 'failed' | 'empty'
  let suggesterFailReason = '';
  // Cache of the latest job snapshot so updateStartButtonState can reason
  // about prerequisites without re-fetching.
  let lastJobSnapshot = null;
  // Latest source-token count from cost-estimate; null until first fetch.
  // 0 means extraction yielded no text (typical: scanned PDF, no OCR).
  let lastSourceTokens = null;

  // ---------- token handling ----------
  const tokenInput = $('token-input');
  const tokenStatus = $('token-status');
  const uploadSection = $('upload-section');

  function getToken() {
    return localStorage.getItem(TOKEN_KEY) || '';
  }
  function setToken(t) {
    if (t) localStorage.setItem(TOKEN_KEY, t);
    else localStorage.removeItem(TOKEN_KEY);
    refreshTokenUI();
  }
  function refreshTokenUI() {
    const t = getToken();
    const historySection = $('history-section');
    if (t) {
      tokenInput.value = '';
      tokenInput.placeholder = '••• saved (click Clear to reset) •••';
      tokenStatus.textContent = 'Token saved.';
      tokenStatus.className = 'status ok';
      uploadSection.hidden = false;
      historySection.hidden = false;
      // Open Recent jobs by default ONLY when there's no active job —
      // otherwise the active-job panel gets pushed below the fold.
      const savedJob = localStorage.getItem(JOB_KEY);
      historySection.open = !savedJob;
      refreshHistory();
      loadStyleProfilesIntoUploadForm();
      // Re-attach to a previously-tracked job after F5 etc.
      if (savedJob && !currentJobId) attachJob(savedJob);
    } else {
      tokenInput.placeholder = 'your-bearer-token';
      tokenStatus.textContent = 'No token set.';
      tokenStatus.className = 'status warn';
      uploadSection.hidden = true;
      historySection.hidden = true;
    }
  }
  $('save-token').onclick = () => {
    const v = tokenInput.value.trim();
    if (!v) {
      tokenStatus.textContent = 'Token cannot be empty.';
      tokenStatus.className = 'status err';
      return;
    }
    setToken(v);
  };
  $('clear-token').onclick = () => setToken('');
  refreshTokenUI();

  // ---------- API helpers ----------
  function authHeaders(extra) {
    const h = { Authorization: `Bearer ${getToken()}` };
    return { ...h, ...(extra || {}) };
  }
  async function api(path, opts = {}) {
    const res = await fetch(path, {
      ...opts,
      headers: { ...(opts.headers || {}), ...authHeaders() },
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`${res.status} ${res.statusText}: ${text}`);
    }
    if (res.status === 204) return null;
    const ct = res.headers.get('content-type') || '';
    if (ct.includes('application/json')) return res.json();
    return res.text();
  }

  // ---------- upload ----------
  // (currentJobId + eventSource are declared at the top of the IIFE — see
  // comment near token handling for why.)

  $('upload-btn').onclick = async () => {
    const file = $('pdf-file').files[0];
    const status = $('upload-status');
    if (!file) {
      status.textContent = 'Pick a PDF first.';
      status.className = 'status err';
      return;
    }
    status.textContent = 'Uploading…';
    status.className = 'status';
    try {
      const fd = new FormData();
      fd.append('file', file);
      const profileKey = $('upload-profile-select').value;
      if (profileKey) fd.append('style_profile_key', profileKey);
      const targetTokens = $('chunk-target-tokens').value;
      if (targetTokens) fd.append('chunk_target_tokens', targetTokens);
      const hardCap = $('chunk-hard-cap-tokens').value;
      if (hardCap) fd.append('chunk_hard_cap_tokens', hardCap);
      const res = await fetch('/api/jobs', {
        method: 'POST',
        headers: authHeaders(),
        body: fd,
      });
      if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
      const job = await res.json();
      status.textContent = `Uploaded. job_id=${job.id}`;
      status.className = 'status ok';
      attachJob(job.id);
    } catch (e) {
      status.textContent = e.message;
      status.className = 'status err';
    }
  };

  // ---------- job tracking ----------
  function attachJob(jobId) {
    currentJobId = jobId;
    localStorage.setItem(JOB_KEY, jobId);
    suggestionsLoaded = false;  // re-fetch once for the new job
    costLoaded = false;
    metadataLoadedPre = false;
    metadataLoadedPost = false;
    suggesterStatus = 'unknown';
    suggesterFailReason = '';
    lastJobSnapshot = null;
    lastSourceTokens = null;
    // Reset PDF preview state so the previous job's iframe doesn't bleed
    // through when the user clicks Preview on a new job.
    const previewFrame = $('pdf-preview-frame');
    const previewContainer = $('pdf-preview-container');
    const previewBtn = $('toggle-preview');
    if (previewFrame) previewFrame.src = '';
    if (previewContainer) previewContainer.hidden = true;
    if (previewBtn) previewBtn.textContent = 'Preview PDF';
    // Collapse the Recent-jobs panel so the active job's section isn't
    // pushed below the fold. User can re-open it manually at any time.
    const hist = $('history-section');
    if (hist) hist.open = false;
    $('job-section').hidden = false;
    $('progress-block').hidden = false;
    $('confirm-block').hidden = true;
    $('done-block').hidden = true;
    $('event-log').innerHTML = '';
    $('chunks-table').querySelector('tbody').innerHTML = '';
    $('progress-bar').value = 0;
    $('progress-label').textContent = '…';
    refreshDetails();
    openEventStream(jobId);
  }

  async function refreshDetails() {
    if (!currentJobId) return;
    try {
      const job = await api(`/api/jobs/${currentJobId}`);
      renderDetails(job);
      adjustSectionsForStatus(job);
    } catch (e) {
      $('progress-label').textContent = `error: ${e.message}`;
      $('progress-label').className = 'status err';
    }
  }

  function renderDetails(job) {
    const dl = $('job-details');
    dl.innerHTML = '';
    const fields = [
      ['id', job.id],
      ['file', job.original_filename],
      ['status', job.status],
      ['category', job.category + (job.category_auto_detected ? ' (auto)' : '')],
      ['profile', job.style_profile_key || '—'],
      ['pages', job.page_count ?? '—'],
      ['size', humanSize(job.file_size_bytes)],
      ['scanned', job.is_scanned ? 'yes' : 'no'],
      ['progress', `${job.progress_percent.toFixed(1)}%`],
    ];
    if (job.total_tokens > 0) fields.push(['tokens', job.total_tokens.toLocaleString()]);
    if (job.estimated_seconds_remaining != null) {
      fields.push(['eta', humanDuration(job.estimated_seconds_remaining)]);
    }
    // Running actual cost from the cost-estimate endpoint (cheap; pulls from
    // chunk_translation_passes). Updated on every refreshDetails() tick.
    if (job.total_tokens > 0) {
      fetchActualCost();
    }
    if (job.error_message) fields.push(['error', job.error_message]);
    for (const [k, v] of fields) {
      const dt = document.createElement('dt');
      dt.textContent = k;
      const dd = document.createElement('dd');
      dd.textContent = v;
      dl.append(dt, dd);
    }
    $('progress-bar').value = job.progress_percent || 0;
  }

  function adjustSectionsForStatus(job) {
    const status = job.status;
    lastJobSnapshot = job;
    if (status === 'AWAITING_CONFIRMATION') {
      $('confirm-block').hidden = false;
      // If the classifier returned UNKNOWN (e.g. low-confidence or no
      // excerpt), show NOVEL as the recommended default in the dropdown
      // AND persist it to the backend so the user can press Start without
      // a manual save click. They can still override the dropdown — that
      // re-saves on change.
      const recommended =
        job.category === 'EDUCATIONAL' ? 'EDUCATIONAL'
          : job.category === 'PERSONAL_DEVELOPMENT' ? 'PERSONAL_DEVELOPMENT'
          : 'NOVEL';
      $('category-select').value = recommended;
      if (job.category !== 'NOVEL' && job.category !== 'EDUCATIONAL' && job.category !== 'PERSONAL_DEVELOPMENT') {
        // Fire-and-forget: persist the recommended value. The next status
        // poll will reflect the updated category.
        saveCategory(recommended);
      }
      loadProfilesForConfirmBlock(job);
      // Only fetch suggestions ONCE per job-attach. Subsequent re-fetches
      // come from explicit triggers (the 'glossary_suggestions_ready' SSE
      // event, the Re-run button, or post-approve/reject actions). Refreshing
      // every status poll would wipe in-flight edits in the Burmese cells.
      if (!suggestionsLoaded) {
        suggestionsLoaded = true;
        // First view of this job — paint the banner now (will update when
        // refreshGlossarySuggestions returns) AND arm the reconcile so we
        // don't sit on 'running' forever if SSE events were already missed.
        renderSuggesterBanner();
        refreshGlossarySuggestions();
        scheduleSuggesterReconcile();
      }
      if (!costLoaded) {
        costLoaded = true;
        refreshCostEstimate();
      }
      // M14 — pre-translation metadata form. Auto-fill once per job-attach
      // so the SSE status poll doesn't clobber user edits in progress.
      if (!metadataLoadedPre) {
        metadataLoadedPre = true;
        fillMetadataInputs(job, 'meta');
      }
      // M16: cover preview reflects whatever's currently saved on the job.
      // Cheap to refresh on every status poll — it's just an <img src>.
      refreshCoverUI(job);
      updateStartButtonState();
    } else {
      $('confirm-block').hidden = true;
    }
    // Show the PDF export settings panel as soon as translation starts,
    // so the user can configure cover / title / fonts while the worker
    // runs. The first rebuild then picks up those settings — no wasted
    // re-export. Outputs subsection still shows only after COMPLETED.
    const showExportPanel = [
      'TRANSLATING',
      'POSTPROCESSING',
      'REBUILDING',
      'COMPLETED',
    ].includes(status);
    $('done-block').hidden = !showExportPanel;
    // Outputs list only meaningful on COMPLETED.
    const outputsSection = $('outputs-section');
    if (outputsSection) {
      outputsSection.hidden = status !== 'COMPLETED';
    }
    // Re-export only works on COMPLETED jobs (the API returns 409 otherwise);
    // grey it out and explain when not eligible. Save still works always —
    // the rebuild stage picks up the saved settings on completion.
    const reExportBtn = $('post-re-export');
    if (reExportBtn) {
      const eligible = status === 'COMPLETED';
      reExportBtn.disabled = !eligible;
      reExportBtn.title = eligible
        ? 'Re-render the PDF using current settings.'
        : `Re-export available after translation completes (current status: ${status}).`;
    }
    if (status === 'COMPLETED') {
      loadOutputs();
    }
    // M14 — load the export-settings form once per job-attach so the
    // SSE status poll doesn't clobber user edits in progress.
    if (showExportPanel && !metadataLoadedPost) {
      metadataLoadedPost = true;
      fillMetadataInputs(job, 'post');
    }
    // Cover preview reflects whatever's currently saved (live across
    // both pre- and post-translation phases).
    if (showExportPanel) {
      refreshCoverUI(job);
    }
    // Quality panel is informative for any post-translate state.
    if (['TRANSLATING', 'POSTPROCESSING', 'REBUILDING', 'COMPLETED', 'FAILED'].includes(status)) {
      refreshQualityPanel();
    }
  }

  function openEventStream(jobId) {
    if (eventSource) eventSource.close();
    const url = `/api/jobs/${jobId}/events?token=${encodeURIComponent(getToken())}`;
    eventSource = new EventSource(url);
    eventSource.addEventListener('hello', () => addLog('hello', 'stream open'));
    eventSource.addEventListener('status', () => refreshDetails());
    eventSource.addEventListener('done', () => {
      addLog('done', 'job finished');
      refreshDetails();
      refreshChunks();
      eventSource.close();
    });
    eventSource.onerror = () => addLog('stream', 'disconnected (will reconnect)');
    // Listen to every event type the backend emits.
    const namedEvents = [
      'extract_started', 'extract_completed', 'extract_failed',
      'classify_started', 'classify_completed', 'classify_failed',
      'translate_started', 'chunks_built', 'translate_chunks_done',
      'glossary_violations', 'translate_completed', 'translate_failed',
      'rebuild_started', 'rebuild_pdf_done', 'rebuild_epub_done',
      'rebuild_completed', 'rebuild_failed', 'uploaded',
      // M7 quality loop:
      'chunk_critiqued', 'chunk_revised', 'revise_skipped',
      'critique_failed', 'revise_failed',
      // M8 glossary suggester:
      'glossary_suggestions_started', 'glossary_suggestions_ready',
      'glossary_suggestions_failed',
      // M9 judge + review queue:
      'chunk_scored', 'low_score_chunk',
    ];
    for (const ev of namedEvents) {
      eventSource.addEventListener(ev, (e) => {
        addLog(ev, e.data);
        refreshChunks();
        if (ev === 'glossary_suggestions_started') {
          // Defense against out-of-order replay: if 'ready'/'empty'/'failed'
          // already arrived (or the table already has rows), don't regress
          // to 'running'. The SSE stream MAY emit a 'started' event after a
          // 'ready' event when both share the same second-precision timestamp.
          if (suggesterStatus === 'unknown') {
            suggesterStatus = 'running';
            renderSuggesterBanner();
            updateStartButtonState();
          }
        }
        if (ev === 'glossary_suggestions_ready') {
          // Mark ready optimistically; the actual count drives the banner
          // ('ready' if rows > 0, 'empty' if zero).
          suggesterStatus = 'ready';
          renderSuggesterBanner();
          updateStartButtonState();
          refreshGlossarySuggestions();
        }
        if (ev === 'glossary_suggestions_failed') {
          suggesterStatus = 'failed';
          try {
            const payload = JSON.parse(e.data || '{}').payload || {};
            suggesterFailReason = payload.error || 'unknown error';
          } catch (_) { suggesterFailReason = 'unknown error'; }
          renderSuggesterBanner();
          updateStartButtonState();
        }
        if (ev === 'chunk_scored' || ev === 'low_score_chunk') {
          refreshQualityPanel();
        }
      });
    }
  }

  function addLog(name, data) {
    const ul = $('event-log');
    const li = document.createElement('li');
    const t = document.createElement('span');
    t.className = 'evt-time';
    t.textContent = new Date().toLocaleTimeString();
    const n = document.createElement('span');
    n.className = 'evt-name';
    n.textContent = name;
    const d = document.createElement('span');
    d.textContent = ' ' + (data || '');
    li.append(t, n, d);
    ul.appendChild(li);
    ul.scrollTop = ul.scrollHeight;
  }

  async function refreshChunks() {
    if (!currentJobId) return;
    try {
      const chunks = await api(`/api/jobs/${currentJobId}/chunks`);
      const tbody = $('chunks-table').querySelector('tbody');
      tbody.innerHTML = '';
      for (const c of chunks) {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td>${c.chunk_index}</td>
          <td>${c.page_range_start}–${c.page_range_end}</td>
          <td>${c.token_count}</td>
          <td class="status-${c.status}">${c.status}</td>
          <td>${c.attempt_count}</td>`;
        tbody.appendChild(tr);
      }
    } catch (_) {
      /* chunks may not exist yet — ignore */
    }
  }

  // ---------- style profiles ----------
  async function loadStyleProfilesIntoUploadForm() {
    const sel = $('upload-profile-select');
    if (!sel) return;
    // Keep the leading "Auto" option; replace the rest each refresh.
    while (sel.options.length > 1) sel.remove(1);
    try {
      const profiles = await api('/api/style-profiles');
      for (const p of profiles) {
        const opt = document.createElement('option');
        opt.value = p.key;
        opt.textContent = `${p.name} (${p.applies_to_category})`;
        sel.appendChild(opt);
      }
    } catch (e) {
      console.warn('failed to load style profiles:', e.message);
    }
  }

  async function loadProfilesForConfirmBlock(job) {
    const sel = $('profile-select');
    if (!sel) return;
    sel.innerHTML = '';
    // Always offer an explicit "no profile" option — the category-specific
    // base prompt + voice card carry the styling guidance on their own,
    // and PERSONAL_DEVELOPMENT has no scoped profiles yet.
    const none = document.createElement('option');
    none.value = '';
    none.textContent = '(none — use category default)';
    if (!job.style_profile_key) none.selected = true;
    sel.appendChild(none);
    try {
      // Filter by the resolved category so the dropdown is tractable.
      const params = job.category && job.category !== 'UNKNOWN'
        ? `?category=${encodeURIComponent(job.category)}`
        : '';
      const profiles = await api(`/api/style-profiles${params}`);
      // Only show profiles whose scope actually matches the category, so
      // PERSONAL_DEVELOPMENT doesn't see NOVEL / EDUCATIONAL profiles
      // that wouldn't help.
      const cat = job.category || '';
      const fits = profiles.filter((p) =>
        cat === 'NOVEL'
          ? ['NOVEL', 'BOTH'].includes(p.applies_to_category)
          : cat === 'EDUCATIONAL'
            ? ['EDUCATIONAL', 'BOTH'].includes(p.applies_to_category)
            : cat === 'PERSONAL_DEVELOPMENT'
              ? ['PERSONAL_DEVELOPMENT', 'BOTH'].includes(p.applies_to_category)
              : true
      );
      for (const p of fits) {
        const opt = document.createElement('option');
        opt.value = p.key;
        opt.textContent = `${p.name}${p.is_default_for_category ? ' (default)' : ''}`;
        if (job.style_profile_key === p.key) opt.selected = true;
        sel.appendChild(opt);
      }
    } catch (e) {
      console.warn('failed to load profiles:', e.message);
    }
  }

  async function saveStyleProfile(key) {
    if (!currentJobId) return;
    const status = $('profile-status');
    // ``key`` may legitimately be the empty string when the user picks
    // "(none — use category default)". Send it through; the API
    // recognizes empty string as "clear the profile".
    try {
      await api(`/api/jobs/${currentJobId}/style-profile`, {
        method: 'PATCH',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ style_profile_key: key || '' }),
      });
      status.textContent = key
        ? `Saved profile: ${key}`
        : 'Cleared profile (using category default).';
      status.className = 'status ok';
      await refreshDetails();
    } catch (e) {
      status.textContent = e.message;
      status.className = 'status err';
    }
  }
  $('save-profile').onclick = () => saveStyleProfile($('profile-select').value);
  $('profile-select').onchange = (e) => saveStyleProfile(e.target.value);

  // ---------- cost estimate ----------
  async function refreshCostEstimate() {
    if (!currentJobId) return;
    const summary = $('cost-summary');
    const tbody = $('cost-scenarios-table').querySelector('tbody');
    summary.textContent = 'Calculating…';
    summary.className = 'status';
    tbody.innerHTML = '';
    try {
      const e = await api(`/api/jobs/${currentJobId}/cost-estimate`);
      lastSourceTokens = e.source_tokens;
      const tokens = e.source_tokens.toLocaleString();
      const chunks = e.estimated_chunks;
      const actualUsd = (e.actual && e.actual.usd) || 0;
      // 0 source tokens = nothing to translate. Common cause: scanned PDF
      // without tesseract installed. Surface the full diagnosis instead of
      // letting the user click Start on an empty book.
      if (e.source_tokens === 0) {
        const isScanned = lastJobSnapshot && lastJobSnapshot.is_scanned;
        if (isScanned) {
          summary.innerHTML = '⚠️ <strong>No text extracted.</strong> This PDF is scanned (image-only pages) and the system needs OCR to read it. Install tesseract: <code>brew install tesseract tesseract-lang</code>, then delete this job and re-upload.';
        } else {
          summary.innerHTML = '⚠️ <strong>No text extracted</strong> from this PDF. The file may be empty, password-protected, or use a font the extractor can\'t read.';
        }
        summary.className = 'status err';
        updateStartButtonState();
        return;
      }
      summary.textContent = `Source: ${tokens} tokens · estimated ${chunks} chunk(s)` +
        (actualUsd > 0 ? ` · spent so far: $${actualUsd.toFixed(2)}` : '');
      summary.className = 'status';
      const order = ['translate_only', 'current_settings', 'full_loop'];
      for (const key of order) {
        const s = e.scenarios[key];
        if (!s) continue;
        const tr = document.createElement('tr');
        const isCurrent = key === 'current_settings';
        tr.innerHTML = `
          <td>${isCurrent ? '<strong>' : ''}${escapeHtml(s.label)}${isCurrent ? '</strong>' : ''}</td>
          <td>${s.tokens.toLocaleString()}</td>
          <td>${isCurrent ? '<strong>' : ''}$${s.usd.toFixed(2)}${isCurrent ? '</strong>' : ''}</td>`;
        tbody.appendChild(tr);
      }
    } catch (err) {
      summary.textContent = err.message;
      summary.className = 'status err';
    }
  }
  $('refresh-cost').onclick = refreshCostEstimate;

  async function fetchActualCost() {
    // Cheap addendum to the details panel — appends a "spent" dt/dd if the
    // actual cost is non-zero. Called every refreshDetails() so it stays
    // current during translation.
    if (!currentJobId) return;
    try {
      const e = await api(`/api/jobs/${currentJobId}/cost-estimate`);
      const usd = (e.actual && e.actual.usd) || 0;
      if (usd <= 0) return;
      const dl = $('job-details');
      // Avoid duplicating the row on each refresh.
      const existing = dl.querySelector('dt[data-spent]');
      if (existing) {
        existing.nextElementSibling.textContent = `$${usd.toFixed(2)}`;
        return;
      }
      const dt = document.createElement('dt');
      dt.textContent = 'spent';
      dt.dataset.spent = '1';
      const dd = document.createElement('dd');
      dd.textContent = `$${usd.toFixed(2)}`;
      dl.append(dt, dd);
    } catch (_) {
      /* non-fatal — details show without cost */
    }
  }

  // ---------- quality panel (M9) ----------
  async function refreshQualityPanel() {
    if (!currentJobId) return;
    const summary = $('quality-summary');
    const tbody = $('review-queue-table').querySelector('tbody');
    try {
      const m = await api(`/api/jobs/${currentJobId}/metrics`);
      const avg = m.avg_overall != null ? m.avg_overall.toFixed(2) : '—';
      const reviseRate = m.revise_rate != null ? `${(m.revise_rate * 100).toFixed(0)}%` : '—';
      summary.textContent =
        `Judged ${m.chunks_judged}/${m.chunks_completed} chunks · avg overall ${avg} · low-score ${m.low_score_count} · revise rate ${reviseRate}`;
      summary.className = 'status';
    } catch (e) {
      summary.textContent = e.message;
      summary.className = 'status err';
      return;
    }
    try {
      const queue = await api(`/api/jobs/${currentJobId}/review-queue`);
      tbody.innerHTML = '';
      for (const r of queue) {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td>${r.chunk_index}</td>
          <td><strong>${r.overall.toFixed(2)}</strong></td>
          <td>${r.naturalness.toFixed(2)}</td>
          <td>${r.fidelity.toFixed(2)}</td>
          <td>${r.register_match.toFixed(2)}</td>
          <td>${r.consistency.toFixed(2)}</td>
          <td>${r.issues_count}</td>`;
        tbody.appendChild(tr);
      }
    } catch (_) {
      /* review queue may be empty — fine */
    }
  }

  // ---------- glossary suggestions ----------
  // Self-healing: if we sit on 'running'/'unknown' too long, the SSE event
  // ordering may have desynced (or the worker crashed). Hit the API
  // directly and derive 'empty' vs 'ready' from the actual rows count.
  let suggesterReconcileTimer = null;
  function scheduleSuggesterReconcile() {
    if (suggesterReconcileTimer) clearTimeout(suggesterReconcileTimer);
    suggesterReconcileTimer = setTimeout(async () => {
      if (suggesterStatus !== 'running' && suggesterStatus !== 'unknown') return;
      if (!currentJobId) return;
      try {
        const rows = await api(`/api/jobs/${currentJobId}/glossary/suggestions`);
        // If the API returns anything, the suggester is definitely done
        // (empty list = it ran and found nothing; non-empty = ready).
        suggesterStatus = rows.length > 0 ? 'ready' : 'empty';
        renderSuggesterBanner();
        updateStartButtonState();
        if (rows.length > 0) refreshGlossarySuggestions();
      } catch (_) {
        /* leave state as-is; next reconcile will retry */
        scheduleSuggesterReconcile();
      }
    }, 12000);
  }

  function renderSuggesterBanner() {
    const status = $('suggestions-status');
    if (suggesterStatus === 'running' || suggesterStatus === 'unknown') {
      status.innerHTML = '⏳ Suggester running… (Gemini takes 5–15s on a typical book)';
      status.className = 'status warn';
      // Arm the self-healing reconcile so we don't sit here forever if
      // the SSE stream is misordered or the worker crashed.
      scheduleSuggesterReconcile();
    } else if (suggesterStatus === 'failed') {
      status.innerHTML = `⚠️ Suggester failed: ${escapeHtml(suggesterFailReason || 'see worker logs')}. You can still upload a manual glossary or skip and start translation.`;
      status.className = 'status err';
    } else if (suggesterStatus === 'empty') {
      // Look at the latest classifier error on the job for a quota hint —
      // when Gemini is quota-blocked, both classifier AND suggester fail
      // the same way, so the empty list is misleading without context.
      const errMsg = (lastJobSnapshot && lastJobSnapshot.error_message) || '';
      const quotaIssue =
        /RESOURCE_EXHAUSTED|spending cap|quota/i.test(errMsg) ||
        /RESOURCE_EXHAUSTED|spending cap|quota/i.test(suggesterFailReason || '');
      if (quotaIssue) {
        status.innerHTML =
          '⚠️ Gemini API quota exhausted — no terms could be surfaced. ' +
          'Bump your project spending cap at <a target="_blank" ' +
          'href="https://ai.studio/spend">ai.studio/spend</a> and click ' +
          '<strong>Re-run</strong>, or skip and translate without a locked glossary.';
        status.className = 'status err';
      } else {
        status.innerHTML =
          'No glossary terms surfaced for this document. ' +
          'Translate without locked terminology, or upload a manual glossary.';
        status.className = 'status';
      }
    } else if (suggesterStatus === 'ready') {
      // Replaced after refreshGlossarySuggestions counts rows.
      status.textContent = 'Loading suggestions…';
      status.className = 'status';
    }
  }

  async function refreshGlossarySuggestions() {
    if (!currentJobId) return;
    const tbody = $('suggestions-table').querySelector('tbody');
    const status = $('suggestions-status');

    // Snapshot any in-flight edits + which input had focus, so that a
    // re-render (post-approve, suggester re-run, etc.) doesn't wipe the
    // user's typing or yank focus mid-edit.
    const liveEdits = new Map();
    for (const inp of tbody.querySelectorAll('input[data-target-edit]')) {
      liveEdits.set(inp.dataset.targetEdit, inp.value);
    }
    const focusedId = document.activeElement && document.activeElement.dataset
      ? document.activeElement.dataset.targetEdit
      : null;
    const focusedSelStart = focusedId ? document.activeElement.selectionStart : null;
    const focusedSelEnd = focusedId ? document.activeElement.selectionEnd : null;

    try {
      const rows = await api(`/api/jobs/${currentJobId}/glossary/suggestions`);
      tbody.innerHTML = '';
      const bulk = $('suggestions-bulk-actions');
      if (!rows.length) {
        // No rows — hide bulk-action buttons; nothing to act on.
        if (bulk) bulk.hidden = true;
        // If we KNOW the suggester finished (ready event seen, or
        // we're seeing this on initial load with no in-flight signal),
        // show "empty". If we don't know yet, leave the banner as-is so
        // the "running…" message stays put until SSE arrives.
        if (suggesterStatus === 'ready') suggesterStatus = 'empty';
        renderSuggesterBanner();
        return;
      }
      // We have rows → suggester is definitely ready.
      suggesterStatus = 'ready';
      status.textContent = `${rows.length} suggestion(s).`;
      status.className = 'status';
      // Show bulk-action buttons only when there are pending rows to act on.
      const anyPending = rows.some((r) => r.state === 'PENDING');
      if (bulk) bulk.hidden = !anyPending;
      updateStartButtonState();
      for (const r of rows) {
        const tr = document.createElement('tr');
        tr.dataset.id = r.id;
        // Prefer the user's in-flight edit over the server value when both
        // exist for this row — the user just typed it, the server hasn't
        // seen it yet.
        const liveValue = liveEdits.has(r.id) ? liveEdits.get(r.id) : r.term_target;
        const targetField = r.state === 'PENDING'
          ? `<input type="text" value="${escapeHtml(liveValue)}" data-target-edit="${r.id}" />`
          : `<span>${escapeHtml(r.term_target)}</span>`;
        const actions = r.state === 'PENDING'
          ? `<div class="actions">
               <button class="approve" data-approve="${r.id}">Approve</button>
               <button class="reject" data-reject="${r.id}">Reject</button>
             </div>`
          : '';
        const conf = Math.max(0, Math.min(1, r.confidence || 0));
        const confClass = conf < 0.6 ? 'low' : (conf < 0.8 ? 'mid' : '');
        const evidenceTitle = escapeHtml((r.evidence_json || []).join(' | '));
        tr.innerHTML = `
          <td class="col-source" title="${evidenceTitle || escapeHtml(r.term_source)}">${escapeHtml(r.term_source)}</td>
          <td>${targetField}</td>
          <td><span class="pill pill-kind">${escapeHtml(r.kind.replace(/_/g, ' ').toLowerCase())}</span></td>
          <td><span class="conf">
            <span class="conf-bar"><span class="conf-bar-fill ${confClass}" style="width:${(conf * 100).toFixed(0)}%"></span></span>
            ${conf.toFixed(2)}
          </span></td>
          <td><span class="pill pill-state-${r.state}">${r.state}</span></td>
          <td>${actions}</td>`;
        tbody.appendChild(tr);
      }
      tbody.querySelectorAll('button[data-approve]').forEach(btn => {
        btn.onclick = () => approveSuggestion(btn.dataset.approve);
      });
      tbody.querySelectorAll('button[data-reject]').forEach(btn => {
        btn.onclick = () => rejectSuggestion(btn.dataset.reject);
      });

      // Restore focus + caret position if the user was mid-edit.
      if (focusedId) {
        const refocus = tbody.querySelector(`input[data-target-edit="${focusedId}"]`);
        if (refocus) {
          refocus.focus();
          try {
            refocus.setSelectionRange(focusedSelStart, focusedSelEnd);
          } catch (_) { /* number inputs etc. don't allow setSelectionRange */ }
        }
      }
    } catch (e) {
      status.textContent = e.message;
      status.className = 'status err';
    }
  }

  async function approveSuggestion(id) {
    const editInput = document.querySelector(`input[data-target-edit="${id}"]`);
    const body = { state: 'APPROVED' };
    if (editInput) body.term_target = editInput.value;
    await api(`/api/jobs/${currentJobId}/glossary/suggestions/${id}`, {
      method: 'PATCH',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    });
    await refreshGlossarySuggestions();
  }

  async function rejectSuggestion(id) {
    await api(`/api/jobs/${currentJobId}/glossary/suggestions/${id}`, {
      method: 'PATCH',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ state: 'REJECTED' }),
    });
    await refreshGlossarySuggestions();
  }

  $('approve-all-suggestions').onclick = async () => {
    // Iterate per-row instead of calling the bulk endpoint, so any edits the
    // user made to the Burmese field are preserved. The bulk endpoint takes
    // no per-row term_target overrides, which would silently drop edits.
    const inputs = document.querySelectorAll('#suggestions-table input[data-target-edit]');
    const status = $('suggestions-status');
    if (!inputs.length) return;
    status.textContent = `Approving ${inputs.length} suggestion(s)…`;
    status.className = 'status';
    for (const input of inputs) {
      const id = input.dataset.targetEdit;
      try {
        await api(`/api/jobs/${currentJobId}/glossary/suggestions/${id}`, {
          method: 'PATCH',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ state: 'APPROVED', term_target: input.value }),
        });
      } catch (e) {
        console.warn('approve failed', id, e.message);
      }
    }
    await refreshGlossarySuggestions();
  };

  $('reject-all-suggestions').onclick = async () => {
    await api(`/api/jobs/${currentJobId}/glossary/suggestions/bulk`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ action: 'reject_all' }),
    });
    await refreshGlossarySuggestions();
  };

  $('rerun-suggestions').onclick = async () => {
    const status = $('suggestions-status');
    status.textContent = 'Running suggester… (may take 5–15s)';
    status.className = 'status';
    try {
      const counts = await api(`/api/jobs/${currentJobId}/glossary/suggestions/run`, {
        method: 'POST',
      });
      status.textContent = `Sampled ${counts.sampled} excerpts → kept ${counts.kept} of ${counts.proposed} proposals.`;
      status.className = 'status ok';
      await refreshGlossarySuggestions();
    } catch (e) {
      status.textContent = e.message;
      status.className = 'status err';
    }
  };

  // ---------- category + glossary + start ----------
  // Auto-save the category whenever the dropdown changes — relying on a
  // separate "Save" click was the #1 footgun (people clicked Start, hit
  // 400 Bad Request "set category before starting translation"). The
  // explicit save-category button stays as a fallback for keyboard users
  // who tab past the change event.
  async function saveCategory(cat) {
    if (!currentJobId || !cat) return;
    try {
      await api(`/api/jobs/${currentJobId}/category`, {
        method: 'PATCH',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ category: cat }),
      });
      await refreshDetails();
    } catch (e) {
      alert(e.message);
    }
  }
  $('save-category').onclick = () => saveCategory($('category-select').value);
  $('category-select').onchange = (e) => saveCategory(e.target.value);

  // -----------------------------------------------------------------------
  // M14 — book metadata + layout-mode (pre-translation form in confirm-block)
  // -----------------------------------------------------------------------
  function fillMetadataInputs(job, prefix) {
    // Title / author are now ONLY in the post-translation panel; the
    // pre-translation 'meta' prefix doesn't have those fields anymore.
    // The lookups guard with ``if`` so the same helper works for both.
    const els = {
      titleSource: $(`${prefix}-title-source`),
      titleTarget: $(`${prefix}-title-target`),
      authorSource: $(`${prefix}-author-source`),
      authorTarget: $(`${prefix}-author-target`),
      layout: $(`${prefix}-layout-mode-select`),
      skip: $(`${prefix}-skip-pages`),
      pageSize: $(`${prefix}-page-size`),
      bodyFontSize: $(`${prefix}-body-font-size`),
      lineHeight: $(`${prefix}-line-height`),
      marginMm: $(`${prefix}-margin-mm`),
      justify: $(`${prefix}-justify`),
    };
    if (els.titleSource) els.titleSource.value = job.title_source || '';
    if (els.titleTarget) els.titleTarget.value = job.title_target || '';
    if (els.authorSource) els.authorSource.value = job.author_source || '';
    if (els.authorTarget) els.authorTarget.value = job.author_target || '';
    if (els.layout) els.layout.value = job.export_layout_mode || 'AUTO';
    if (els.skip) els.skip.value = job.skip_pages || '';
    const ov = job.export_template_overrides_json || {};
    if (els.pageSize) els.pageSize.value = ov.page_size || '';
    if (els.bodyFontSize) els.bodyFontSize.value = ov.body_font_size_pt ?? '';
    if (els.lineHeight) els.lineHeight.value = ov.line_height ?? '';
    if (els.marginMm) els.marginMm.value = ov.margin_mm ?? '';
    if (els.justify) els.justify.checked = !!ov.justify;
  }

  function readMetadataInputs(prefix) {
    const els = {
      titleSource: $(`${prefix}-title-source`),
      titleTarget: $(`${prefix}-title-target`),
      authorSource: $(`${prefix}-author-source`),
      authorTarget: $(`${prefix}-author-target`),
      layout: $(`${prefix}-layout-mode-select`),
      skip: $(`${prefix}-skip-pages`),
      pageSize: $(`${prefix}-page-size`),
      bodyFontSize: $(`${prefix}-body-font-size`),
      lineHeight: $(`${prefix}-line-height`),
      marginMm: $(`${prefix}-margin-mm`),
      justify: $(`${prefix}-justify`),
    };

    // Build template overrides ONLY from non-empty form values. Including
    // an empty key would clobber the saved job-level override; sending
    // null for a key clears it.
    //
    // Each numeric field has a sane range. Out-of-range values (e.g.
    // ``line_height = 15`` if the user mis-typed and missed the dot)
    // get clamped here so a typo doesn't render an absurd PDF. The
    // alternative — sending the raw value and letting the renderer
    // produce an unreadable book — is much worse.
    const overrides = {};
    const numField = (raw, lo, hi, def) => {
      const v = parseFloat(raw);
      if (!Number.isFinite(v)) return null;       // empty / "—" → clear
      if (v < lo || v > hi) return def;           // mistyped → reset to default
      return v;
    };
    if (els.pageSize) overrides.page_size = els.pageSize.value || null;
    if (els.bodyFontSize) {
      overrides.body_font_size_pt = numField(els.bodyFontSize.value, 6, 24, 10);
    }
    if (els.lineHeight) {
      overrides.line_height = numField(els.lineHeight.value, 1.0, 2.5, 1.5);
    }
    if (els.marginMm) {
      overrides.margin_mm = numField(els.marginMm.value, 4, 60, 14);
    }
    if (els.justify) overrides.justify = !!els.justify.checked;

    // Drop keys that didn't even have a form element (so an
    // `meta`-prefixed call doesn't send `page_size: null` and clear the
    // post-prefix saved value).
    const overridesPayload = Object.keys(overrides).length
      ? overrides
      : undefined;

    const out = {};
    if (els.titleSource) out.title_source = els.titleSource.value || '';
    if (els.titleTarget) out.title_target = els.titleTarget.value || '';
    if (els.authorSource) out.author_source = els.authorSource.value || '';
    if (els.authorTarget) out.author_target = els.authorTarget.value || '';
    if (els.layout) out.export_layout_mode = els.layout.value;
    if (els.skip) out.skip_pages = els.skip.value || '';
    if (overridesPayload !== undefined) {
      out.export_template_overrides_json = overridesPayload;
    }
    return out;
  }

  async function saveMetadata(prefix, statusElId) {
    if (!currentJobId) return;
    const status = $(statusElId);
    status.textContent = 'Saving…';
    status.className = 'status';
    try {
      const updated = await api(`/api/jobs/${currentJobId}/metadata`, {
        method: 'PATCH',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(readMetadataInputs(prefix)),
      });
      status.textContent = 'Saved.';
      status.className = 'status ok';
      // Refresh both forms so values stay in sync if user opens both panels.
      fillMetadataInputs(updated, 'meta');
      fillMetadataInputs(updated, 'post');
    } catch (e) {
      status.textContent = e.message;
      status.className = 'status err';
    }
  }

  $('save-metadata').onclick = () => saveMetadata('meta', 'metadata-status');

  // -----------------------------------------------------------------------
  // M16 — cover image upload / preview / delete
  // -----------------------------------------------------------------------
  function refreshCoverUI(job) {
    const preview = $('cover-preview');
    const removeBtn = $('delete-cover');
    if (!preview || !removeBtn) return;
    if (job.cover_image_path) {
      // ?v= cache-buster so a re-uploaded cover replaces the rendered preview.
      preview.src =
        `/api/jobs/${currentJobId}/cover?v=${outputVersion}` +
        `&token=${encodeURIComponent(getToken())}`;
      preview.hidden = false;
      removeBtn.hidden = false;
    } else {
      preview.hidden = true;
      preview.src = '';
      removeBtn.hidden = true;
    }
  }

  $('upload-cover').onclick = async () => {
    if (!currentJobId) return;
    const fileInput = $('cover-file');
    const status = $('cover-status');
    const file = fileInput.files[0];
    if (!file) {
      status.textContent = 'Pick an image first.';
      status.className = 'status err';
      return;
    }
    status.textContent = 'Uploading…';
    status.className = 'status';
    try {
      const fd = new FormData();
      fd.append('file', file);
      const res = await fetch(`/api/jobs/${currentJobId}/cover`, {
        method: 'POST',
        headers: authHeaders(),
        body: fd,
      });
      if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
      const job = await res.json();
      // Bump the output version so the cover-preview <img> refetches.
      outputVersion = Date.now();
      status.textContent = 'Cover uploaded.';
      status.className = 'status ok';
      fileInput.value = '';
      refreshCoverUI(job);
      lastJobSnapshot = job;
    } catch (e) {
      status.textContent = e.message;
      status.className = 'status err';
    }
  };

  $('delete-cover').onclick = async () => {
    if (!currentJobId) return;
    const status = $('cover-status');
    status.textContent = 'Removing…';
    status.className = 'status';
    try {
      const job = await api(`/api/jobs/${currentJobId}/cover`, { method: 'DELETE' });
      outputVersion = Date.now();
      status.textContent = 'Cover removed.';
      status.className = 'status ok';
      refreshCoverUI(job);
      lastJobSnapshot = job;
    } catch (e) {
      status.textContent = e.message;
      status.className = 'status err';
    }
  };

  $('post-save-metadata').onclick = () =>
    saveMetadata('post', 'post-metadata-status');

  // Re-export uses whatever metadata is currently saved on the job. Saves
  // first so user typing in the inputs flows through to the rendered PDF
  // without requiring a separate Save click.
  $('post-re-export').onclick = async () => {
    if (!currentJobId) return;
    const status = $('post-metadata-status');
    status.textContent = 'Saving + re-exporting…';
    status.className = 'status';
    try {
      // Save first so the new title/author/mode flows into the render.
      await api(`/api/jobs/${currentJobId}/metadata`, {
        method: 'PATCH',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(readMetadataInputs('post')),
      });
      const result = await api(`/api/jobs/${currentJobId}/re-export`, {
        method: 'POST',
      });
      status.innerHTML = `Re-exported (${result.mode}, ${result.pages_written} pages, ${(result.size_bytes / 1024).toFixed(0)} KB).`;
      status.className = 'status ok';
      // Bump the cache-buster so the iframe + download links serve the
      // newly-rendered file rather than the browser-cached previous PDF.
      outputVersion = Date.now();
      // Refresh outputs list so the user sees the new file size.
      loadOutputs();
      // If the preview is currently open, force-reload it with the new URL.
      refreshPreviewIfOpen();
    } catch (e) {
      status.textContent = e.message;
      status.className = 'status err';
    }
  };

  $('upload-glossary').onclick = async () => {
    const file = $('glossary-file').files[0];
    const status = $('glossary-status');
    if (!file) {
      status.textContent = 'Pick a file first.';
      status.className = 'status err';
      return;
    }
    try {
      const fd = new FormData();
      fd.append('file', file);
      const res = await fetch(`/api/jobs/${currentJobId}/glossary`, {
        method: 'POST',
        headers: authHeaders(),
        body: fd,
      });
      if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
      const body = await res.json();
      status.textContent = `Saved ${body.saved} term(s).`;
      status.className = 'status ok';
    } catch (e) {
      status.textContent = e.message;
      status.className = 'status err';
    }
  };

  function updateStartButtonState() {
    const btn = $('start-btn');
    if (!btn) return;
    const job = lastJobSnapshot;
    const reasons = [];
    if (!job) {
      btn.disabled = true;
      return;
    }
    // Hard-block: nothing to translate. The cost panel shows the full
    // diagnosis (scanned + OCR-missing, or empty PDF, etc.) — the button
    // hint just nudges them to look there.
    if (lastSourceTokens === 0) {
      reasons.push('No source text was extracted from this PDF. See cost panel for details.');
    }
    if (job.category !== 'NOVEL' && job.category !== 'EDUCATIONAL' && job.category !== 'PERSONAL_DEVELOPMENT') {
      reasons.push('Choose a category (NOVEL or EDUCATIONAL).');
    }
    if (suggesterStatus === 'running' || suggesterStatus === 'unknown') {
      reasons.push('Glossary suggester still running — wait for it, or skip with the "Start anyway" link below.');
    }
    btn.disabled = reasons.length > 0;
    let hint = $('start-blocked');
    if (!hint) {
      hint = document.createElement('p');
      hint.id = 'start-blocked';
      hint.className = 'status warn';
      btn.parentNode.insertBefore(hint, btn.nextSibling);
    }
    if (reasons.length) {
      const force = (suggesterStatus === 'running' || suggesterStatus === 'unknown')
        ? ` <a href="#" id="start-force">Start anyway</a>`
        : '';
      hint.innerHTML = '⚠️ ' + reasons.map(r => escapeHtml(r)).join(' ') + force;
      hint.hidden = false;
      const force_link = $('start-force');
      if (force_link) {
        force_link.onclick = (e) => {
          e.preventDefault();
          // User explicitly wants to start despite suggester pending.
          startTranslation({ force: true });
        };
      }
    } else {
      hint.hidden = true;
    }
  }

  async function startTranslation(opts = {}) {
    if (!currentJobId) return;
    if (!opts.force) {
      // Re-validate gate just before sending — the snapshot might be stale.
      const job = lastJobSnapshot;
      if (!job || (job.category !== 'NOVEL' && job.category !== 'EDUCATIONAL' && job.category !== 'PERSONAL_DEVELOPMENT')) {
        alert('Set the category first.');
        return;
      }
    }
    try {
      await api(`/api/jobs/${currentJobId}/start`, { method: 'POST' });
      await refreshDetails();
    } catch (e) {
      alert(e.message);
    }
  }

  $('start-btn').onclick = () => startTranslation();

  $('cancel-btn').onclick = async () => {
    if (!confirm('Cancel this job? Workers will stop at the next safe point.')) return;
    try {
      await api(`/api/jobs/${currentJobId}/cancel`, { method: 'POST' });
      await refreshDetails();
    } catch (e) {
      alert(e.message);
    }
  };

  $('resume-btn').onclick = async () => {
    try {
      await api(`/api/jobs/${currentJobId}/resume`, { method: 'POST' });
      addLog('resume', 'requested');
      attachJob(currentJobId);  // reopen SSE stream
    } catch (e) {
      alert(e.message);
    }
  };

  $('preview-page-btn').onclick = async () => {
    const status = $('preview-status');
    const page = parseInt($('preview-page-num').value, 10) || 1;
    status.textContent = 'Rendering preview…';
    status.className = 'status';
    try {
      const res = await fetch(
        `/api/jobs/${currentJobId}/preview-page?page=${page}`,
        { method: 'POST', headers: authHeaders() }
      );
      if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      window.open(url, '_blank');
      status.textContent = `Preview opened in new tab.`;
      status.className = 'status ok';
    } catch (e) {
      status.textContent = e.message;
      status.className = 'status err';
    }
  };

  $('open-compare').onclick = () => {
    if (!currentJobId) return;
    window.location.href = `/compare?job_id=${encodeURIComponent(currentJobId)}`;
  };

  function previewUrl() {
    // Cache-bust on outputVersion so re-exports show the new render
    // instead of the browser-cached previous PDF. ?inline=1 flips the
    // response to ``Content-Disposition: inline``; without it the browser
    // would obey the default attachment disposition and trigger a download.
    return (
      `/api/jobs/${encodeURIComponent(currentJobId)}/outputs/pdf` +
      `?inline=1&v=${outputVersion}&token=${encodeURIComponent(getToken())}`
    );
  }

  $('toggle-preview').onclick = () => {
    if (!currentJobId) return;
    const container = $('pdf-preview-container');
    const frame = $('pdf-preview-frame');
    const btn = $('toggle-preview');
    if (container.hidden) {
      const url = previewUrl();
      if (frame.src !== url) frame.src = url;
      container.hidden = false;
      btn.textContent = 'Hide preview';
      // Scroll the preview into view so the user immediately sees it.
      container.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } else {
      container.hidden = true;
      btn.textContent = 'Preview PDF';
    }
  };

  function refreshPreviewIfOpen() {
    const container = $('pdf-preview-container');
    if (!container || container.hidden) return;
    const frame = $('pdf-preview-frame');
    if (!frame || !currentJobId) return;
    // Force the iframe to reload with the new cache-buster.
    frame.src = previewUrl();
  }

  $('forget-job').onclick = () => {
    if (eventSource) eventSource.close();
    eventSource = null;
    currentJobId = null;
    localStorage.removeItem(JOB_KEY);
    $('job-section').hidden = true;
  };

  $('refresh-history').onclick = refreshHistory;

  async function refreshHistory() {
    try {
      const jobs = await api('/api/jobs?limit=20');
      const tbody = $('history-table').querySelector('tbody');
      tbody.innerHTML = '';
      for (const j of jobs) {
        const tr = document.createElement('tr');
        const created = new Date(j.created_at + 'Z').toLocaleString();
        tr.innerHTML = `
          <td>${created}</td>
          <td title="${j.id}">${escapeHtml(j.original_filename)}</td>
          <td class="status-${j.status}">${j.status}</td>
          <td>${j.page_count ?? '—'}</td>
          <td>${j.total_tokens ? j.total_tokens.toLocaleString() : '—'}</td>
          <td><button data-job="${j.id}" class="secondary">Open</button></td>`;
        tbody.appendChild(tr);
      }
      tbody.querySelectorAll('button[data-job]').forEach(btn => {
        btn.onclick = () => attachJob(btn.dataset.job);
      });
    } catch (e) {
      console.error('history fetch failed:', e);
    }
  }

  function escapeHtml(s) {
    const d = document.createElement('div');
    d.textContent = s ?? '';
    return d.innerHTML;
  }

  // ---------- outputs ----------
  async function loadOutputs() {
    try {
      const data = await api(`/api/jobs/${currentJobId}/outputs`);
      const ul = $('outputs-list');
      ul.innerHTML = '';
      for (const o of data.outputs) {
        const li = document.createElement('li');
        const a = document.createElement('a');
        // Cache-bust on outputVersion so a re-export's new file actually
        // downloads instead of the browser-cached previous version.
        a.href = `/api/jobs/${currentJobId}/outputs/${o.format.toLowerCase()}?v=${outputVersion}&token=${encodeURIComponent(getToken())}`;
        a.textContent = `Download ${o.format} (${humanSize(o.size_bytes)})`;
        a.target = '_blank';
        li.appendChild(a);
        ul.appendChild(li);
      }
      $('done-block').hidden = data.outputs.length === 0;
    } catch (e) {
      console.error(e);
    }
  }

  function humanSize(n) {
    if (n == null) return '—';
    const u = ['B', 'KB', 'MB', 'GB'];
    let i = 0;
    while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
    return `${n.toFixed(i ? 1 : 0)} ${u[i]}`;
  }

  function humanDuration(seconds) {
    if (seconds == null) return '—';
    if (seconds < 60) return `~${Math.max(1, Math.round(seconds))}s`;
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.round(seconds % 60);
    if (h > 0) return `~${h}h ${m}m`;
    return s > 0 && m < 10 ? `~${m}m ${s}s` : `~${m}m`;
  }
})();
