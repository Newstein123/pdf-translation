# Track C — Layout Pipeline Audit (M11 Day 1)

Goal: ship translated books with layout that matches the source within ±5%.
This document is an honest read of what the pipeline captures today, what's
missing, and the smallest set of changes needed to hit the target.

Files audited:
- `app/services/extraction.py` (415 lines)
- `app/services/postprocess.py` (43 lines)
- `app/services/rebuild_pdf.py` (403 lines)
- `app/services/rebuild_epub.py` (231 lines)

---

## 1. What's already captured (good news)

Per-block, today, extraction emits this shape (`structure.json`):

```
{ "type": "paragraph" | "heading" | "image" | "ocr",
  "bbox": [x0, y0, x1, y1],
  "dominant_font": { "family", "size", "bold", "italic" } | None,
  "spans": [ {text, bbox, font, size, bold, italic}, ... ] }
```

Concretely that means we already have:

| Element | Captured? | Where |
|---|---|---|
| Per-block bounding box | ✅ | `extraction.py:218,260,278` |
| Dominant font family / size | ✅ | `_dominant_font()` |
| Bold / italic per span | ✅ | PyMuPDF flags (bit 4 / bit 1) |
| Heading detection by font size | ✅ | `heading_threshold = median × multiplier` |
| Page-level background-color sampling | ✅ | for redact-and-fill rebuild |
| Image / OCR blocks distinguished | ✅ | block `type` field |
| Span-level bboxes | ✅ | needed for icon-aware insert rect |
| Document-median font size | ✅ | `_document_median_font_size` |

PDF rebuild path is already block-level: redact original bbox → `insert_htmlbox`
the translation into the same rect with embedded Burmese fonts; shrink-to-fit
with `_starting_font_size`; icon-aware insert rect to avoid clobbering
decorative glyphs. EPUB rebuild flattens to a reflowable structure.

This is a stronger foundation than I assumed before reading the code. Most
of Track C is **completing** what's started, not building from scratch.

---

## 2. Gaps blocking the ±5% goal

### G1. No structural role beyond {paragraph, heading, image, ocr}

**Symptom:** captions, list items, footnotes, table cells, blockquotes,
epigraphs, figure labels all collapse into `paragraph`. Style preservation
(P0 B3) and footnote anchoring (P0 B4) need finer roles.

**Fix:** add a `role` enum on each block:
`BODY | HEADING_{1..6} | LIST_ITEM | QUOTE | CAPTION | FOOTNOTE | FOOTNOTE_ANCHOR | TABLE_CELL | TABLE_HEADER | EPIGRAPH | RUNNING_HEADER | RUNNING_FOOTER | PAGE_NUMBER`.

Heuristics to assign role at extraction time:
- Headings already classified by font size; map to H1–H6 by size buckets relative to median.
- Running headers / footers / page numbers: blocks with stable y-coordinate across pages and short text length.
- Footnotes: blocks whose y-coordinate is in the bottom 15% of page AND font size below body median × 0.85.
- List items: blocks starting with bullet (`•`, `-`, `*`) or numbering pattern (`1.`, `(a)`).
- Captions: short blocks (<160 chars) immediately above or below an `image` block, font size below body median.

This is all heuristic; the linter (Track B) has a similar shape — start
permissive, tighten with real data.

### G2. No table extraction at all

**Symptom:** PyMuPDF's text extraction sees table cells as a stream of
paragraphs. Translation works per cell *by accident* (each cell often
becomes its own block) but column structure, row alignment, merged cells
are lost on rebuild. Educational books rely on tables — this is the
biggest single gap for that genre.

**Fix:** add a table-detection pass after page text extraction using
`page.find_tables()` (PyMuPDF supports it). Each detected table becomes a
block of `type=table` with `cells_json` carrying row/col/text. Rebuild
treats tables as a unit: redact the table bbox, then render an HTML
`<table>` with `insert_htmlbox` inside the redacted area.

### G3. No span-by-span style preservation across translation

**Symptom:** Today, each block is translated as a single string. Bold
emphasis on *one phrase* in the middle of a paragraph is lost — the
translation is plain, even if `dominant_font.bold == False`. Same for
italic. The reconstruction can't put the bold back because the model never
saw the marker.

**Fix:** before sending a block to translation, encode style runs as
inline markers (`«b»…«/b»`, `«i»…«/i»`); ask the model to preserve markers
in the Burmese. Decode on the way back into per-run formatting that the
rebuild's `insert_htmlbox` already supports via inline `<b>`/`<i>` HTML.
Add `style_alignment_quality` per block so we can flag low-confidence
alignments to QA.

### G4. No expansion-compensation strategy

**Symptom:** Burmese text is typically 1.05–1.25× longer in line count
than English at the same font/size. `rebuild_pdf` shrinks font size when
overflow detected (`_starting_font_size`), which is correct, but it's a
single knob. A coordinated policy applied at *page* level would close
the ±5% page-count gap on most books.

**Fix:** new `expansion_policy` per export profile:
- Tighten leading first (1.5 → 1.4).
- Then reduce font size up to 0.5pt (currently the only knob).
- Then allow controlled bbox growth ≤ 5%.
- Then flow to next page (last resort, recorded in QA report).

Policy is a fixed strategy, not free-form: same input + same policy = same
output. Per-block compensation gets logged so we can see what knob was
turned where.

### G5. No layout-fidelity measurement

**Symptom:** We have no way to declare "this book met the ±5% bar"
automatically. Today, ±5% is aspirational; without a measurement loop we
can't tune the expansion compensation against real data.

**Fix:** post-rebuild visual diff (P0 B5):
- Render source and target PDFs to images at the same DPI.
- Compute per-page metrics: page count delta, per-block bbox IoU, image
  position delta, heading/footnote count parity.
- Aggregate into a Layout QA Report (PDF + JSON).
- Gate the export profile on configurable thresholds.

### G6. Burmese typography not enforced consistently

**Symptom:** Fonts are downloaded (`scripts/download_fonts.sh`,
`utils/fonts.get_burmese_fonts`) and embedded in PDF, which is great. But:
- No NFC normalization or Zawgyi detection at extraction time. (Linter
  v1 — Track B — catches Zawgyi residue in *output*; doesn't catch
  Zawgyi-encoded *source*.)
- No ZWSP injection at Burmese word boundaries — justified Burmese gets
  rivers / poor wrap.
- `mm_line_height = 1.6` is set globally but not adjusted per role
  (headings need looser, footnotes tighter).

**Fix:** typography pass between postprocess and rebuild:
- Detect Zawgyi on extracted text via codepoint heuristic; convert
  Zawgyi → Unicode and flag the job.
- NFC-normalize all extracted Burmese text (rare but possible: source
  PDFs with Burmese passages embedded).
- Inject ZWSP at word boundaries on Burmese output via ICU/myWord
  segmentation (or a lighter rule-based segmenter for v1).
- Per-role line-height adjustment in `_build_css`.

### G7. No layout map sidecar

**Symptom:** The structure JSON is shaped per-page-per-block, which is
right. But there's no canonical layout map separate from the per-chunk
`blocks_json` / `translated_blocks_json`. As we add roles, expansion
budgets, and compensation logs, those need a stable home.

**Fix:** introduce a `LayoutMap` JSON document that's the source of truth
for rebuild — page-organized, every block carries source bbox, role,
style flags, expansion-budget hints. `structure.json` becomes the input;
LayoutMap is the version-controlled artifact rebuild reads. Migration is
mechanical: add a build step that derives LayoutMap from structure.json
and persist its path on the existing `document_structure` row.

---

## 3. Proposed Track C order (against the W2–W7 plan)

| Week | Deliverable | Lands |
|---|---|---|
| W2 | LayoutMap sidecar (G7) + role enrichment (G1) | foundation |
| W3 | Burmese typography pass (G6) — Zawgyi conversion + ZWSP + per-role leading | quality |
| W4 | Expansion-compensation policy (G4) — coordinated knobs, recorded per-block | ±5% candidate |
| W5 | Style preservation across translation (G3) — inline markers + decode | bold/italic survive |
| W6 | Tables, figures, captions, footnotes (G2) — find_tables + role-aware rebuild | educational books |
| W7 | Visual-diff regression gate (G5) — measure & enforce | M3 milestone |

Each week ships an increment that's testable on its own. G1+G7 ship together
because role data needs a place to live. G2 (tables) lands later but is the
single biggest deliverable for educational books — flag this as a candidate
to pull forward if a textbook is in the M4 test set.

---

## 4. Risks specific to Track C

| Risk | Mitigation |
|---|---|
| `find_tables()` accuracy is uneven on complex tables (merged cells, nested) | Ship table extraction as opt-in per profile; manual-review flag for low-confidence detections |
| Style markers in translate prompt confuse model | Use a dedicated prompt purpose (`translate_with_styles`) so we can iterate on it without disturbing the main translate prompt |
| Zawgyi conversion has edge cases | Detect-and-flag first; conversion only after we've verified on real Zawgyi inputs |
| Visual diff is slow on long books | Sample-only mode (every 10th page) for routine runs; full-render only on M4 validation |
| LayoutMap sidecar diverges from structure.json | Derive it deterministically from structure.json on every job; never hand-edit |

---

## 5. Decisions needed (before W4 lands)

1. **Per-role line-height table.** Is there a house-style baseline we should
   use, or should I derive defaults from typical published Burmese books?
2. **±5% definition.** Is the bar (a) page-count delta ≤ 5%, (b) per-page
   geometry IoU ≥ 0.95, or (c) both? The compensation knobs we tune
   depend on which.
3. **Educational-book priority.** Pull G2 (tables) forward to W4 if a
   textbook is in the M4 test set.

---

## 6. What does NOT need to change

- PyMuPDF as the extraction backend — it's the right tool, no need to
  swap.
- Redact-and-fill rebuild strategy — it's a sound approach for
  layout-preserving translation; we're refining, not redesigning.
- Per-block translation chunking — block boundaries align with
  layout boundaries, which is what we want.
- Background-color sampling — already produces good results, leave alone.
- Font embedding pipeline — already works, just extend with better
  per-role CSS.
