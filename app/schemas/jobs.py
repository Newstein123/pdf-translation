from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.db.models import (
    Category,
    ExportLayoutMode,
    JobStatus,
    ProfileCategoryScope,
    SentenceComplexity,
)


class JobCreateResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    original_filename: str
    file_size_bytes: int
    status: JobStatus
    created_at: datetime


class JobStatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    original_filename: str
    status: JobStatus
    category: Category
    category_auto_detected: bool
    page_count: int | None
    file_size_bytes: int
    progress_percent: float
    is_scanned: bool
    error_message: str | None
    style_profile_key: str | None = None
    total_tokens: int = 0  # sum of cost_tokens across all chunks
    estimated_seconds_remaining: int | None = None  # ETA, only set during active stages
    # M14: book metadata + export layout mode (all editable any time).
    title_source: str | None = None
    title_target: str | None = None
    author_source: str | None = None
    author_target: str | None = None
    export_layout_mode: ExportLayoutMode = ExportLayoutMode.AUTO
    export_template_overrides_json: dict | None = None
    # M15: page-range string (e.g. "1-3, 5, 8-10") of pages to skip from
    # both translation and the final PDF.
    skip_pages: str | None = None
    # M16: filesystem path to the user-uploaded cover image, or None.
    # The UI checks this to show a "cover uploaded ✓" indicator and
    # decides whether to render the preview thumbnail.
    cover_image_path: str | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None


class CategoryUpdateRequest(BaseModel):
    category: Category


class JobMetadataUpdateRequest(BaseModel):
    """PATCH body for updating book metadata + export layout mode.

    All fields are optional — only ones explicitly set are applied. Pass
    ``""`` to clear a string field; ``None`` (omit it) leaves it untouched.
    """

    # The ``model_config`` allows ``""`` to flow through as "clear this field"
    # while None means "don't change". Pydantic v2 distinguishes these
    # cleanly without needing a custom marker.
    title_source: str | None = None
    title_target: str | None = None
    author_source: str | None = None
    author_target: str | None = None
    export_layout_mode: ExportLayoutMode | None = None
    export_template_overrides_json: dict | None = None
    # M15: page-range string of pages to skip translation + export.
    # Pass ``""`` to clear, ``None`` (omit) to leave unchanged.
    skip_pages: str | None = None


class ReExportResponse(BaseModel):
    job_id: str
    output_path: str
    pages_written: int
    chapters_rendered: int
    blocks_placed: int
    size_bytes: int
    mode: str  # "TEMPLATE" | "PRESERVE"


class StyleProfileUpdateRequest(BaseModel):
    style_profile_key: str


class StructureSummaryResponse(BaseModel):
    job_id: str
    structured_json_path: str
    total_blocks: int
    total_pages: int
    is_scanned: bool
    scanned_pages: list[int]
    pages_preview: list[dict[str, Any]]  # first few pages, blocks truncated


class GlossaryEntryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    term_source: str
    term_target: str
    case_sensitive: bool


class GlossaryUploadResponse(BaseModel):
    job_id: str
    saved: int


class ChunkSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    chunk_index: int
    page_range_start: int
    page_range_end: int
    token_count: int
    status: str
    attempt_count: int
    cost_tokens: int


class TranslationOutputResponse(BaseModel):
    job_id: str
    chunks_total: int
    chunks_completed: int
    pages: list[dict[str, Any]]  # [{"page_number": int, "translated_text": "..."}]
    plain_text: str


class OutputItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    format: str
    file_path: str
    size_bytes: int


class OutputListResponse(BaseModel):
    job_id: str
    outputs: list[OutputItem]


class StyleProfileItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    key: str
    name: str
    description: str | None
    applies_to_category: ProfileCategoryScope
    sentence_complexity: SentenceComplexity
    is_default_for_category: bool
    version: int


class GlossarySuggestionItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    term_source: str
    term_target: str
    kind: str
    confidence: float
    evidence_json: list[str] | None = None
    state: str


class SuggestionPatchRequest(BaseModel):
    state: str  # "APPROVED" | "REJECTED"
    term_target: str | None = None  # optional edit on approve


class SuggestionsBulkRequest(BaseModel):
    action: str  # "approve_all" | "reject_all" | "approve_ids"
    ids: list[str] | None = None


class SuggestionsRunResponse(BaseModel):
    sampled: int
    proposed: int
    kept: int


class QualityScoreItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    chunk_id: str
    chunk_index: int
    pass_id: str | None
    naturalness: float
    fidelity: float
    register_match: float
    consistency: float
    overall: float
    issues_count: int = 0
    judge_model: str
    created_at: datetime


class JobMetricsSummary(BaseModel):
    job_id: str
    chunks_total: int
    chunks_completed: int
    chunks_judged: int
    avg_overall: float | None
    low_score_count: int
    revise_rate: float | None
    tokens_by_pass_kind: dict[str, int]


class AdminMetricsResponse(BaseModel):
    jobs_total: int
    chunks_total: int
    chunks_judged: int
    avg_overall: float | None
    revise_rate: float | None
    tokens_by_pass_kind: dict[str, int]
    avg_revise_uplift: float | None  # placeholder until we paired-judge pre/post revise


class CostScenario(BaseModel):
    label: str
    tokens: int
    usd: float


class CostEstimateResponse(BaseModel):
    source_tokens: int
    estimated_chunks: int
    scenarios: dict[str, CostScenario]  # translate_only | current_settings | full_loop
    actual: dict[str, Any]  # rolling actual cost from passes — see actual_job_cost


class CompareBlock(BaseModel):
    block_id: str
    type: str  # paragraph | heading | ocr
    source: str
    translated: str | None = None  # None when this block hasn't been translated yet


class ComparePage(BaseModel):
    page_number: int
    blocks: list[CompareBlock]


class CompareResponse(BaseModel):
    job_id: str
    original_filename: str
    pages: list[ComparePage]
    pages_total: int
    blocks_total: int
    blocks_translated: int

