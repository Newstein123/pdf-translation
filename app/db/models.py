from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class JobStatus(str, enum.Enum):
    UPLOADED = "UPLOADED"
    EXTRACTING = "EXTRACTING"
    CLASSIFYING = "CLASSIFYING"
    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"
    TRANSLATING = "TRANSLATING"
    POSTPROCESSING = "POSTPROCESSING"
    REBUILDING = "REBUILDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class Category(str, enum.Enum):
    NOVEL = "NOVEL"
    EDUCATIONAL = "EDUCATIONAL"
    # Self-help / motivational / business / mindset books. Distinct from
    # NOVEL because the register differs (instructional second-person,
    # shorter sentences, examples-driven) and from EDUCATIONAL because
    # the tone is conversational rather than textbook-formal.
    PERSONAL_DEVELOPMENT = "PERSONAL_DEVELOPMENT"
    UNKNOWN = "UNKNOWN"


class ChunkStatus(str, enum.Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    RETRY = "RETRY"


class OutputFormat(str, enum.Enum):
    PDF = "PDF"
    EPUB = "EPUB"


class ExportLayoutMode(str, enum.Enum):
    """How the PDF rebuild stage should produce the output.

    * AUTO — system picks based on the source PDF's color/imagery analysis
      (white-paper books → TEMPLATE, anything with color/decoration → PRESERVE).
    * TEMPLATE — render translated text through a clean Burmese book template
      (single configurable layout: A5 page, generous margins, running headers,
      page numbers). Best for plain-prose novels and non-fiction.
    * PRESERVE — keep the source layout via redact-and-fill (the existing
      ``rebuild_pdf`` path). Best for textbooks, illustrated books, and
      anything with sidebars / colored backgrounds.
    """

    AUTO = "AUTO"
    TEMPLATE = "TEMPLATE"
    PRESERVE = "PRESERVE"


class PassKind(str, enum.Enum):
    TRANSLATE = "TRANSLATE"
    CRITIQUE = "CRITIQUE"
    REVISE = "REVISE"
    JUDGE = "JUDGE"
    HUMAN_EDIT = "HUMAN_EDIT"


class PassStatus(str, enum.Enum):
    OK = "OK"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class PromptPurpose(str, enum.Enum):
    TRANSLATE = "TRANSLATE"
    CRITIQUE = "CRITIQUE"
    REVISE = "REVISE"
    JUDGE = "JUDGE"
    SUGGEST = "SUGGEST"
    CLASSIFY = "CLASSIFY"


class ProfileCategoryScope(str, enum.Enum):
    NOVEL = "NOVEL"
    EDUCATIONAL = "EDUCATIONAL"
    PERSONAL_DEVELOPMENT = "PERSONAL_DEVELOPMENT"
    BOTH = "BOTH"


class SentenceComplexity(str, enum.Enum):
    SIMPLE = "SIMPLE"
    MEDIUM = "MEDIUM"
    COMPLEX = "COMPLEX"


class SuggestionKind(str, enum.Enum):
    PROPER_NOUN = "PROPER_NOUN"
    TERM = "TERM"
    IDIOM = "IDIOM"
    LOCATION = "LOCATION"
    ORG = "ORG"
    OTHER = "OTHER"


class SuggestionState(str, enum.Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    MERGED = "MERGED"  # superseded by an explicit manual entry


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    original_filename: Mapped[str] = mapped_column(String(512))
    file_path: Mapped[str] = mapped_column(String(1024))
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    file_size_bytes: Mapped[int] = mapped_column(Integer)
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, native_enum=False), default=JobStatus.UPLOADED, index=True
    )
    category: Mapped[Category] = mapped_column(
        Enum(Category, native_enum=False), default=Category.UNKNOWN
    )
    category_auto_detected: Mapped[bool] = mapped_column(Boolean, default=False)
    source_language: Mapped[str | None] = mapped_column(String(16), nullable=True)
    target_language: Mapped[str] = mapped_column(String(16), default="my")
    is_scanned: Mapped[bool] = mapped_column(Boolean, default=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    progress_percent: Mapped[float] = mapped_column(Float, default=0.0)
    style_profile_id: Mapped[str | None] = mapped_column(
        ForeignKey("style_profiles.id", ondelete="SET NULL"), nullable=True
    )
    critique_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    quality_score_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    chunk_target_tokens_override: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_hard_cap_tokens_override: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # M13: per-job override for which translator's voice card the prompts
    # should reference. NULL falls back to ``settings.default_voice_card_translator``.
    # Stored as the translator name (matches ``VoiceCard.translator``) so
    # regenerating cards offline doesn't invalidate the link.
    voice_card_translator: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # M14: book metadata + export layout mode.
    # Source/target title and author are editable at any stage. The template
    # export uses them in running headers; PRESERVE export uses them only as
    # PDF metadata. They auto-fill from PDF metadata at upload time.
    title_source: Mapped[str | None] = mapped_column(String(512), nullable=True)
    title_target: Mapped[str | None] = mapped_column(String(512), nullable=True)
    author_source: Mapped[str | None] = mapped_column(String(255), nullable=True)
    author_target: Mapped[str | None] = mapped_column(String(255), nullable=True)
    export_layout_mode: Mapped[ExportLayoutMode] = mapped_column(
        Enum(ExportLayoutMode, native_enum=False), default=ExportLayoutMode.AUTO
    )
    # JSON dict of per-job overrides for template settings (page size, fonts,
    # margins, etc.). NULL means "use the global defaults from settings.";
    # individual keys can be overridden without specifying all of them.
    export_template_overrides_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # M15: page-range string of pages to skip translation AND drop from the
    # final exported PDF. Format: ``"1-3, 5, 8-10"`` (1-indexed, mirroring
    # what the user sees in their PDF reader). Translation pipeline filters
    # blocks before chunking — saves Gemini tokens. Rebuild paths drop the
    # pages so they don't appear in the output. NULL = translate everything.
    skip_pages: Mapped[str | None] = mapped_column(String(512), nullable=True)
    # M16: filesystem path to the user-uploaded cover image (normalized to
    # JPEG). When set, the export pipeline renders the cover as page 1 of
    # the output. NULL = no cover; output starts with TOC / body as usual.
    cover_image_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    structure: Mapped[DocumentStructure | None] = relationship(
        back_populates="job", uselist=False, cascade="all, delete-orphan"
    )
    chunks: Mapped[list[TranslationChunk]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    glossary: Mapped[list[GlossaryEntry]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    names: Mapped[list[NameRegistry]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    outputs: Mapped[list[Output]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    events: Mapped[list[JobEvent]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    style_profile: Mapped[StyleProfile | None] = relationship(
        back_populates="jobs", foreign_keys=[style_profile_id]
    )

    @property
    def style_profile_key(self) -> str | None:
        """Convenience for API responses; ``None`` until classification picks a default."""
        return self.style_profile.key if self.style_profile is not None else None

    @property
    def total_tokens(self) -> int:
        """Sum of Gemini tokens consumed across all chunks. Triggers a lazy
        load of ``self.chunks`` if not already populated — fine for status
        polling, the chunks-by-job index keeps this cheap."""
        return sum((c.cost_tokens or 0) for c in self.chunks)

    @property
    def estimated_seconds_remaining(self) -> int | None:
        """Best-effort ETA for translation completion.

        We project from observed throughput: ``completed_chunks / elapsed``
        gives chunks-per-second; multiply remaining by the inverse. This auto-
        accounts for actual concurrency and Gemini latency variance — better
        than a static per-chunk estimate.

        Returns None when the job isn't actively translating yet, has no
        completed chunks to extrapolate from, or is already done."""
        if self.status not in {
            JobStatus.TRANSLATING,
            JobStatus.POSTPROCESSING,
            JobStatus.REBUILDING,
        }:
            return None

        chunks = self.chunks or []
        if not chunks:
            return None

        completed = [c for c in chunks if c.status == ChunkStatus.COMPLETED and c.completed_at]
        if not completed:
            return None

        started = [c.started_at for c in chunks if c.started_at is not None]
        if not started:
            return None

        from datetime import datetime, timezone

        earliest = min(started)
        latest_completed = max(c.completed_at for c in completed)
        # Anchor at wall-clock if newer chunks are still in-flight; that lets
        # the ETA tick up rather than freeze between chunk completions.
        # Drop tzinfo so we can subtract from naive timestamps consistently.
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        anchor = max(latest_completed, now)
        elapsed_s = (anchor - earliest).total_seconds()
        if elapsed_s <= 0:
            return None

        rate = len(completed) / elapsed_s  # completed chunks / second
        if rate <= 0:
            return None

        remaining = len(chunks) - len(completed)
        if remaining <= 0:
            # Translation done; account for ~20s rebuild headroom on big books.
            return 20 if self.status != JobStatus.COMPLETED else None

        translate_eta = remaining / rate
        # Budget a flat ~20s for the rebuild stage that follows.
        return int(translate_eta + 20)


class DocumentStructure(Base):
    __tablename__ = "document_structure"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), unique=True)
    structured_json_path: Mapped[str] = mapped_column(String(1024))
    total_blocks: Mapped[int] = mapped_column(Integer, default=0)
    extracted_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    job: Mapped[Job] = relationship(back_populates="structure")


class TranslationChunk(Base):
    __tablename__ = "translation_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    chunk_index: Mapped[int] = mapped_column(Integer)
    page_range_start: Mapped[int] = mapped_column(Integer)
    page_range_end: Mapped[int] = mapped_column(Integer)
    source_text: Mapped[str] = mapped_column(Text)
    translated_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    blocks_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    translated_blocks_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    context_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[ChunkStatus] = mapped_column(
        Enum(ChunkStatus, native_enum=False), default=ChunkStatus.PENDING
    )
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    glossary_violations_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    cost_tokens: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    running_context_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    final_pass_id: Mapped[str | None] = mapped_column(
        ForeignKey(
            "chunk_translation_passes.id", use_alter=True, ondelete="SET NULL",
            name="fk_translation_chunks_final_pass_id",
        ),
        nullable=True,
    )

    job: Mapped[Job] = relationship(back_populates="chunks")
    passes: Mapped[list[ChunkTranslationPass]] = relationship(
        back_populates="chunk",
        cascade="all, delete-orphan",
        foreign_keys="ChunkTranslationPass.chunk_id",
        order_by="ChunkTranslationPass.pass_number",
    )
    final_pass: Mapped[ChunkTranslationPass | None] = relationship(
        foreign_keys=[final_pass_id], post_update=True
    )

    __table_args__ = (
        Index("ix_chunks_job_status", "job_id", "status"),
        Index("ix_chunks_job_order", "job_id", "chunk_index"),
    )


class GlossaryEntry(Base):
    __tablename__ = "glossary_entries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    job_id: Mapped[str | None] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), nullable=True
    )
    term_source: Mapped[str] = mapped_column(String(512))
    term_target: Mapped[str] = mapped_column(String(512))
    case_sensitive: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    job: Mapped[Job | None] = relationship(back_populates="glossary")


class GlossarySuggestion(Base):
    __tablename__ = "glossary_suggestions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    term_source: Mapped[str] = mapped_column(String(512))
    term_target: Mapped[str] = mapped_column(String(512))
    kind: Mapped[SuggestionKind] = mapped_column(
        Enum(SuggestionKind, native_enum=False), default=SuggestionKind.OTHER
    )
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    evidence_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    state: Mapped[SuggestionState] = mapped_column(
        Enum(SuggestionState, native_enum=False), default=SuggestionState.PENDING
    )
    approved_glossary_entry_id: Mapped[str | None] = mapped_column(
        ForeignKey("glossary_entries.id", ondelete="SET NULL"), nullable=True
    )
    actioned_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    actioned_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_glossary_suggestions_job_state", "job_id", "state"),
    )


class NameRegistry(Base):
    __tablename__ = "name_registry"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    source_name: Mapped[str] = mapped_column(String(512))
    burmese_phonetic: Mapped[str] = mapped_column(String(512))
    first_seen_chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    occurrence_count: Mapped[int] = mapped_column(Integer, default=1)

    job: Mapped[Job] = relationship(back_populates="names")

    __table_args__ = (Index("ix_names_job_source", "job_id", "source_name", unique=True),)


class Output(Base):
    __tablename__ = "outputs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    format: Mapped[OutputFormat] = mapped_column(Enum(OutputFormat, native_enum=False))
    file_path: Mapped[str] = mapped_column(String(1024))
    size_bytes: Mapped[int] = mapped_column(Integer)
    generated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    job: Mapped[Job] = relationship(back_populates="outputs")


class JobEvent(Base):
    __tablename__ = "job_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    event_type: Mapped[str] = mapped_column(String(64))
    payload_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)

    job: Mapped[Job] = relationship(back_populates="events")


class StyleProfile(Base):
    __tablename__ = "style_profiles"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    key: Mapped[str] = mapped_column(String(64), unique=True)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    applies_to_category: Mapped[ProfileCategoryScope] = mapped_column(
        Enum(ProfileCategoryScope, native_enum=False), default=ProfileCategoryScope.BOTH
    )
    system_prompt_template: Mapped[str] = mapped_column(Text)
    few_shot_examples_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    vocabulary_preferences_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    sentence_complexity: Mapped[SentenceComplexity] = mapped_column(
        Enum(SentenceComplexity, native_enum=False), default=SentenceComplexity.MEDIUM
    )
    version: Mapped[int] = mapped_column(Integer, default=1)
    is_default_for_category: Mapped[bool] = mapped_column(Boolean, default=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    jobs: Mapped[list[Job]] = relationship(
        back_populates="style_profile", foreign_keys="Job.style_profile_id"
    )


class PromptVersion(Base):
    __tablename__ = "prompt_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    purpose: Mapped[PromptPurpose] = mapped_column(Enum(PromptPurpose, native_enum=False))
    key: Mapped[str] = mapped_column(String(128))
    version: Mapped[int] = mapped_column(Integer, default=1)
    hash: Mapped[str] = mapped_column(String(64))
    body: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("ix_prompt_versions_purpose_key_active", "purpose", "key", "active"),
    )


class QualityScore(Base):
    __tablename__ = "quality_scores"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    chunk_id: Mapped[str] = mapped_column(
        ForeignKey("translation_chunks.id", ondelete="CASCADE")
    )
    pass_id: Mapped[str | None] = mapped_column(
        ForeignKey("chunk_translation_passes.id", ondelete="SET NULL"), nullable=True
    )
    naturalness: Mapped[float] = mapped_column(Float)
    fidelity: Mapped[float] = mapped_column(Float)
    register_match: Mapped[float] = mapped_column(Float)
    consistency: Mapped[float] = mapped_column(Float)
    overall: Mapped[float] = mapped_column(Float)
    reasons_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    judge_model: Mapped[str] = mapped_column(String(64))
    judge_prompt_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("prompt_versions.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("ix_quality_scores_chunk", "chunk_id"),
        Index("ix_quality_scores_overall", "overall"),
    )


class ChunkTranslationPass(Base):
    __tablename__ = "chunk_translation_passes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    chunk_id: Mapped[str] = mapped_column(
        ForeignKey("translation_chunks.id", ondelete="CASCADE")
    )
    pass_number: Mapped[int] = mapped_column(Integer)
    pass_kind: Mapped[PassKind] = mapped_column(Enum(PassKind, native_enum=False))
    model: Mapped[str] = mapped_column(String(64))
    prompt_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("prompt_versions.id", ondelete="SET NULL"), nullable=True
    )
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    # Subset of ``input_tokens`` that hit Gemini's context cache. Billed at
    # ~25% of the standard input rate, so cost reports must subtract this
    # from the full-rate bucket. Always ``<= input_tokens``.
    cached_input_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    gemini_cache_id: Mapped[str | None] = mapped_column(
        ForeignKey("gemini_caches.id", ondelete="SET NULL"), nullable=True
    )
    translated_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    translated_blocks_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    critique_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    critique_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[PassStatus] = mapped_column(
        Enum(PassStatus, native_enum=False), default=PassStatus.OK
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    chunk: Mapped[TranslationChunk] = relationship(
        back_populates="passes", foreign_keys=[chunk_id]
    )
    prompt_version: Mapped[PromptVersion | None] = relationship()

    __table_args__ = (
        Index("ix_passes_chunk_pass_number", "chunk_id", "pass_number"),
        Index("ix_passes_chunk_kind", "chunk_id", "pass_kind"),
    )


# ---------------------------------------------------------------------------
# M11 — Production foundation
#
# BookBrief         : whole-book pre-analysis output (POV, register, voice notes).
#                     Injected as a system-prompt block on every per-chunk pass.
# ReferencePassage  : curated source↔Burmese aligned passages used as few-shot
#                     examples retrieved per chunk.
# LinterFinding     : deterministic post-translation lint results
#                     (Zawgyi residue, glossary violations, pronoun drift, …).
# GeminiCache       : tracks Gemini context-cache handles so prompt prefixes
#                     are only paid for once per cache TTL.
# ---------------------------------------------------------------------------


class BriefStatus(str, enum.Enum):
    PENDING = "PENDING"
    READY = "READY"
    FAILED = "FAILED"


class ReferenceMode(str, enum.Enum):
    NARRATION = "NARRATION"
    DIALOGUE = "DIALOGUE"
    INTROSPECTION = "INTROSPECTION"
    EXPOSITION = "EXPOSITION"
    POETRY = "POETRY"
    OTHER = "OTHER"


class LinterSeverity(str, enum.Enum):
    BLOCKER = "BLOCKER"
    WARNING = "WARNING"
    INFO = "INFO"


class BookBrief(Base):
    """Whole-book pre-analysis emitted before chunking starts.

    The brief is the single biggest cross-chunk coherence lever: every pass
    sees the same POV / tense / register / voice notes, instead of inferring
    from a 1500-token slice. Edited briefs are versioned so an in-flight job
    can finish on the brief it started with."""

    __tablename__ = "book_briefs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), unique=True
    )
    status: Mapped[BriefStatus] = mapped_column(
        Enum(BriefStatus, native_enum=False), default=BriefStatus.PENDING
    )
    brief_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    sample_token_count: Mapped[int] = mapped_column(Integer, default=0)
    model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    prompt_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("prompt_versions.id", ondelete="SET NULL"), nullable=True
    )
    edited_by_human: Mapped[bool] = mapped_column(Boolean, default=False)
    version: Mapped[int] = mapped_column(Integer, default=1)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class ReferencePassage(Base):
    """One curated source↔target passage from a published Burmese translation.

    Retrieved per chunk by embedding similarity (filtered by genre / register /
    mode) and injected as a few-shot example. The corpus is the moat — quality
    of curation matters far more than size."""

    __tablename__ = "reference_passages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    source_text: Mapped[str] = mapped_column(Text)
    target_text: Mapped[str] = mapped_column(Text)
    source_lang: Mapped[str] = mapped_column(String(16), default="en")
    target_lang: Mapped[str] = mapped_column(String(16), default="my")
    # Free-form so we can grow new tags without migrations; we filter on
    # exact matches in retrieval.
    genre: Mapped[str | None] = mapped_column(String(64), nullable=True)
    register: Mapped[str | None] = mapped_column(String(64), nullable=True)
    mode: Mapped[ReferenceMode] = mapped_column(
        Enum(ReferenceMode, native_enum=False), default=ReferenceMode.OTHER
    )
    book_title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    translator: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # Embedding stored as JSON array of floats. SQLite has no native vector
    # type and we don't want to require pgvector for local dev; the retrieval
    # service handles cosine similarity in Python. Migrating to pgvector later
    # is a one-shot copy.
    embedding_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    added_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("ix_reference_passages_genre_mode", "genre", "mode", "active"),
    )


class LinterFinding(Base):
    """One deterministic-rule lint finding against a translated chunk.

    Severity drives flow: BLOCKER routes the chunk back to REVISE, WARNING
    surfaces in the QA report, INFO is observability only. ``resolved_in_pass_id``
    lets us see whether a follow-up pass actually fixed the issue."""

    __tablename__ = "linter_findings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    chunk_id: Mapped[str] = mapped_column(
        ForeignKey("translation_chunks.id", ondelete="CASCADE")
    )
    pass_id: Mapped[str | None] = mapped_column(
        ForeignKey("chunk_translation_passes.id", ondelete="SET NULL"), nullable=True
    )
    rule_key: Mapped[str] = mapped_column(String(64), index=True)
    severity: Mapped[LinterSeverity] = mapped_column(
        Enum(LinterSeverity, native_enum=False), default=LinterSeverity.WARNING
    )
    block_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    position_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    position_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    snippet: Mapped[str | None] = mapped_column(Text, nullable=True)
    payload_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    resolved_in_pass_id: Mapped[str | None] = mapped_column(
        ForeignKey("chunk_translation_passes.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("ix_linter_findings_chunk_severity", "chunk_id", "severity"),
        Index("ix_linter_findings_chunk_rule", "chunk_id", "rule_key"),
    )


class VoiceCard(Base):
    """Distilled style brief for a single translator.

    The card is ~200-400 tokens of prose describing the translator's voice
    (sentence rhythm, register, dialogue style, particle preferences,
    vocabulary patterns), generated *once* offline by analyzing the
    translator's reference passages. At translation time, the card is
    injected into the cached system-prompt prefix — cheap per chunk because
    cached input bills at ~25% of standard rate.

    ``baselines_json`` stores empirical numbers extracted from the same
    corpus (avg sentence length, particle density, dialogue ratio) which
    the linter uses as comparison baselines when flagging drift."""

    __tablename__ = "voice_cards"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    translator: Mapped[str] = mapped_column(String(255), unique=True)
    target_lang: Mapped[str] = mapped_column(String(16), default="my")
    genre_tags_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    card_text: Mapped[str] = mapped_column(Text)
    baselines_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    source_passage_count: Mapped[int] = mapped_column(Integer, default=0)
    model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("ix_voice_cards_translator_active", "translator", "active"),
    )


class GeminiCache(Base):
    """A Gemini context-cache handle.

    Caching the static prompt prefix (system instructions + brief + style
    profile + glossary) cuts ~50% off the bill on long jobs. ``prefix_hash``
    lets us reuse caches across jobs that share the same prefix; ``expires_at``
    is the Gemini-side TTL."""

    __tablename__ = "gemini_caches"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    job_id: Mapped[str | None] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), nullable=True
    )
    purpose: Mapped[PromptPurpose] = mapped_column(
        Enum(PromptPurpose, native_enum=False)
    )
    prefix_hash: Mapped[str] = mapped_column(String(64), index=True)
    gemini_cache_name: Mapped[str] = mapped_column(String(255))
    model: Mapped[str] = mapped_column(String(64))
    ttl_seconds: Mapped[int] = mapped_column(Integer, default=3600)
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    cached_token_count: Mapped[int] = mapped_column(Integer, default=0)
    hit_count: Mapped[int] = mapped_column(Integer, default=0)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("ix_gemini_caches_prefix_hash_purpose", "prefix_hash", "purpose"),
        Index("ix_gemini_caches_expires_at", "expires_at"),
    )
