from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    api_bearer_token: str = Field(default="change-me-local-dev-token")
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    storage_root: Path = Path("./storage")
    database_url: str = "sqlite:///./storage/pdf_translation.db"

    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    max_upload_mb: int = 100
    max_pages: int = 500

    gemini_api_key: str = ""
    gemini_translation_model: str = "gemini-2.5-pro"
    gemini_utility_model: str = "gemini-2.5-flash"
    # Critique runs on every chunk by default; cheap model suffices because
    # critique is rubric-following, not generative. Override per-deployment.
    gemini_critique_model: str = "gemini-2.5-flash"
    # Independent judge for M9 quality scoring. Default to the same family
    # but the admin should override to a *different* model family in prod
    # (cross-family judging reduces self-bias).
    gemini_judge_model: str = "gemini-2.5-flash"

    chunk_target_tokens: int = 1500
    chunk_hard_cap_tokens: int = 2000
    chunk_context_tokens: int = 200
    translation_concurrency: int = 5
    chunk_max_retries: int = 2

    # Quality loop (M7).
    # Critique always runs when enabled; revise only fires when the chunk
    # falls below ``quality_score_threshold`` *or* the critique flagged any
    # high-severity issue. See plan 1-task-summary-convert-mossy-tide.md.
    critique_enabled: bool = True
    quality_score_threshold: float = 4.0  # 1.0–5.0 scale, see critique rubric
    # Hard cap on the sum of (translate + critique + revise) tokens for a
    # single chunk. If pass 3 would push us past this, we skip revise and
    # keep pass 1 — defensive against runaway cost on adversarial chunks.
    max_passes_total_tokens: int = 100_000

    # M9 LLM-as-judge sampling.
    # Per-chunk: every Nth chunk is judged inline at translate time. The
    # first chunk and any *revised* chunk are always judged regardless of
    # the sample rate.
    judge_sample_rate: float = 0.2
    # Threshold for flagging a chunk into the review queue. 1.0–5.0 scale.
    low_score_threshold: float = 3.5

    # M10 typography. Burmese stacked consonants need more vertical space than
    # Latin; 1.6–1.7 reads well with Pyidaungsu/Padauk. Use a single setting
    # so the PDF and EPUB rebuilds stay consistent.
    mm_line_height: float = 1.6

    # Pricing for the cost estimator. USD per 1M tokens. Defaults track Gemini
    # 2.5 published rates at <200k context; admin should override via env if
    # Google adjusts pricing. The estimator multiplies (tokens × rate / 1e6).
    gemini_pro_input_usd_per_m: float = 1.25
    gemini_pro_output_usd_per_m: float = 10.00
    gemini_flash_input_usd_per_m: float = 0.30
    gemini_flash_output_usd_per_m: float = 2.50
    # M11: cached-input pricing. Gemini bills cached input at ~25% of standard
    # input rate. Cache storage is a small per-hour fee (fractions of a cent
    # on a 1-hour TTL) — we don't model it here; the savings come almost
    # entirely from the discounted input rate.
    gemini_pro_cached_input_usd_per_m: float = 0.31  # 25% of 1.25
    gemini_flash_cached_input_usd_per_m: float = 0.075  # 25% of 0.30

    # M13: voice cards. Default translator whose voice card is injected when
    # a job has no per-job override (Job.voice_card_translator). Set to "" to
    # disable voice-card injection globally. The default favors Mya Than Tint
    # because his conversational, modern register fits the broadest range of
    # contemporary novels; switch to "ShweOoDaung" for a more formal default.
    default_voice_card_translator: str = "မြသန်းတင့်"

    # Per-category voice-card defaults. Resolution order is:
    #   1. Job.voice_card_translator (per-job override)
    #   2. Per-category default (these settings)
    #   3. default_voice_card_translator (global fallback)
    # Empty string means "no category-specific default; use the global one".
    voice_card_for_novel: str = "မြသန်းတင့်"
    voice_card_for_educational: str = ""
    voice_card_for_personal_development: str = "ဖေမြင့်"

    # M14: PDF export template defaults.
    # Per-job overrides live in ``Job.export_template_overrides_json``; whatever
    # isn't overridden falls back to these values. All length values are in
    # millimeters unless noted; font sizes are in points.
    #
    # Auto-detect threshold for AUTO export-layout-mode. A page is "white"
    # when ≥ this fraction of its pixels are near-white (RGB > 240); a book
    # is "white" when ≥ 95% of pages meet that bar AND no page has images.
    auto_white_page_pixel_threshold: float = 0.985

    # Page geometry. Margins set tight enough that justified Burmese
    # gets enough characters per line to avoid extreme word-gap stretch.
    template_page_size: str = "A5"  # A5 | A4 | LETTER | 6x9
    template_margin_mm: float = 14.0
    # Body typography.
    # Default = Noto Sans Myanmar so the TEMPLATE export matches the
    # PRESERVE / layout-preserving output that users have already
    # validated. Switch to "Padauk" or "Pyidaungsu" via env / per-job
    # override for a more traditional book typeface.
    template_body_font: str = "Noto Sans Myanmar"
    # 10pt body matches typical Burmese paperback typography. Combined
    # with tighter margins this gives ~28-32 chars per line, which is
    # enough density that justify stretches the few spaces only
    # mildly.
    template_body_font_size_pt: float = 10.0
    template_line_height: float = 1.5
    template_first_line_indent_mm: float = 5.0
    # Justify by default (full page width). Burmese justification
    # would balloon word gaps at larger font sizes; the smaller body
    # font + tighter margins keep gaps reasonable.
    template_justify: bool = True
    # Page furniture.
    # Page-number style: "plain" -> "4"; "page" -> "4 | Page"; "dashed" -> "— 4 —"; "" hidden.
    template_page_number_style: str = "page"
    # Running header content: "title" | "chapter" | "title_and_chapter" | "" hidden
    template_running_header: str = "title_and_chapter"
    # Running header alignment: "alternating" (book title outer, chapter inner)
    # vs "centered" (everything centered).
    template_header_alignment: str = "alternating"
    # Chapter title style: "large_centered" | "large_left" | "smaller" | "" hidden.
    template_chapter_style: str = "large_centered"
    # Section break ornament: "stars" -> "* * *"; "ornament" -> "❦"; "blank" -> blank line; "" hidden.
    template_section_break: str = "stars"

    artifact_retention_hours: int = 72

    # Extraction tuning
    ocr_languages: str = "eng"  # tesseract language codes, e.g. "eng+fra+mya"
    ocr_dpi: int = 200
    scanned_page_char_threshold: int = 50  # below this we treat the page as scanned
    heading_size_multiplier: float = 1.3  # font-size threshold vs median for heading detection
    classification_excerpt_chars: int = 4000

    @property
    def uploads_dir(self) -> Path:
        return self.storage_root / "uploads"

    @property
    def intermediate_dir(self) -> Path:
        return self.storage_root / "intermediate"

    @property
    def outputs_dir(self) -> Path:
        return self.storage_root / "outputs"

    def ensure_dirs(self) -> None:
        for d in (self.storage_root, self.uploads_dir, self.intermediate_dir, self.outputs_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
