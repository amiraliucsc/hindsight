"""Unit tests for key_date extraction (no DB / no live LLM required).

Covers the surfaces this feature added:

1. Pydantic schemas accept and round-trip key_date (across all four
   extraction-mode classes — concise/default, no-causal, verbose).
2. _parse_key_date handles bare ISO date, ISO datetime, and bad input.
3. ExtractedFact + ProcessedFact dataclasses round-trip key_date.
4. fact_storage emits `key_date:YYYY-MM-DD` tag and overrides event_date
   when ProcessedFact carries a key_date.
5. Prompt-integrity smoke tests: the CONCISE and VERBOSE prompts each
   teach the key_date discipline and warn against conflation.

Test data is deliberately fictional. Personal names, household ids, and
prod-derived dates are out of scope for this test file.
"""

from __future__ import annotations

from datetime import date, datetime

from hindsight_api.engine.retain.fact_extraction import (
    ExtractedFact as ExtractedFactPyd,
)
from hindsight_api.engine.retain.fact_extraction import (
    ExtractedFactNoCausal,
    ExtractedFactVerbose,
    Fact,
    _parse_key_date,
)
from hindsight_api.engine.retain.types import (
    ExtractedFact,
    ProcessedFact,
)

# ---- Test data (fictional; single source of truth used across assertions) ----

EVENT_DATE = date(2027, 3, 22)
EVENT_ISO = EVENT_DATE.isoformat()  # "2027-03-22"
EVENT_DOW = "Monday"
EVENT_DESCRIPTION = "Project anniversary"

PERSON_NAME = "Jordan"
AGENT_TAG = "agent:testagent"
HOUSEHOLD_TAG = "household:testhouse"

# A second distinct date used by some tests to verify (entity, horizon)
# dedup and tag overrides do not conflate two different dates.
OTHER_DATE = date(2027, 3, 15)
OTHER_DATE_ISO = OTHER_DATE.isoformat()


# ---- Pydantic LLM-facing schemas ----------------------------------------------


class TestFactSchema:
    def test_fact_accepts_key_date(self):
        f = Fact(
            fact=f"{EVENT_DESCRIPTION} | {EVENT_ISO}",
            fact_type="world",
            occurred_start=EVENT_ISO,
            key_date=EVENT_ISO,
        )
        assert f.key_date == EVENT_ISO

    def test_fact_key_date_optional(self):
        f = Fact(fact="User prefers tea", fact_type="world")
        assert f.key_date is None

    def test_extracted_fact_pyd_accepts_key_date(self):
        ef = ExtractedFactPyd(
            what=f"{PERSON_NAME} {EVENT_DESCRIPTION} is on {EVENT_ISO}",
            when=EVENT_ISO,
            where="N/A",
            who=PERSON_NAME,
            why="N/A",
            fact_kind="event",
            fact_type="world",
            occurred_start=EVENT_ISO,
            key_date=EVENT_ISO,
        )
        assert ef.key_date == EVENT_ISO

    def test_extracted_fact_pyd_key_date_optional(self):
        ef = ExtractedFactPyd(
            what="x",
            when="N/A",
            where="N/A",
            who="N/A",
            why="N/A",
            fact_type="world",
        )
        assert ef.key_date is None

    def test_extracted_fact_no_causal_accepts_key_date(self):
        """Without this, retain_extract_causal_links=False silently strips
        key_date from the LLM's structured-output schema (review finding)."""
        ef = ExtractedFactNoCausal(
            what=f"{EVENT_DESCRIPTION}",
            when=EVENT_ISO,
            where="N/A",
            who=PERSON_NAME,
            why="N/A",
            fact_type="world",
            key_date=EVENT_ISO,
        )
        assert ef.key_date == EVENT_ISO

    def test_extracted_fact_no_causal_key_date_optional(self):
        ef = ExtractedFactNoCausal(
            what="x", when="N/A", where="N/A", who="N/A", why="N/A", fact_type="world"
        )
        assert ef.key_date is None

    def test_extracted_fact_verbose_accepts_key_date(self):
        """Without this, extraction_mode='verbose' silently strips key_date
        (review finding)."""
        ef = ExtractedFactVerbose(
            what=f"{EVENT_DESCRIPTION} is on {EVENT_ISO} — long-form description",
            when=EVENT_ISO,
            where="N/A",
            who=PERSON_NAME,
            why="Significant milestone, want to plan something meaningful",
            fact_type="world",
            key_date=EVENT_ISO,
        )
        assert ef.key_date == EVENT_ISO

    def test_extracted_fact_verbose_key_date_optional(self):
        ef = ExtractedFactVerbose(
            what="x", when="N/A", where="N/A", who="N/A", why="N/A", fact_type="world"
        )
        assert ef.key_date is None


# ---- Helper for parsing the LLM's key_date string ------------------------------


class TestParseKeyDate:
    def test_bare_iso_date(self):
        assert _parse_key_date(EVENT_ISO) == EVENT_DATE

    def test_iso_datetime_truncated_to_date(self):
        assert _parse_key_date(f"{EVENT_ISO}T14:30:00") == EVENT_DATE

    def test_iso_datetime_with_timezone(self):
        # Truncated at the date component; timezone ignored.
        assert _parse_key_date(f"{EVENT_ISO}T14:30:00-07:00") == EVENT_DATE

    def test_none_returns_none(self):
        assert _parse_key_date(None) is None

    def test_empty_string_returns_none(self):
        assert _parse_key_date("") is None

    def test_garbage_returns_none_silently(self):
        # The whole point: refusal on bad input is safer than raising.
        assert _parse_key_date("not a date") is None
        assert _parse_key_date("2027-13-99") is None  # invalid month/day
        assert _parse_key_date("xyz") is None


# ---- Dataclass round-trips -----------------------------------------------------


class TestExtractedFactDataclass:
    def test_default_key_date_is_none(self):
        ef = ExtractedFact(fact_text="x", fact_type="world")
        assert ef.key_date is None

    def test_can_set_key_date(self):
        ef = ExtractedFact(
            fact_text=EVENT_DESCRIPTION,
            fact_type="world",
            key_date=EVENT_DATE,
        )
        assert ef.key_date == EVENT_DATE


class TestProcessedFactDataclass:
    def _make(self, **overrides):
        defaults = dict(
            fact_text="x",
            fact_type="world",
            embedding=[0.0] * 8,
            occurred_start=None,
            occurred_end=None,
            mentioned_at=datetime(2027, 3, 1),
            context="",
            metadata={},
        )
        defaults.update(overrides)
        return ProcessedFact(**defaults)

    def test_default_key_date_is_none(self):
        pf = self._make()
        assert pf.key_date is None

    def test_key_date_round_trips(self):
        pf = self._make(key_date=EVENT_DATE)
        assert pf.key_date == EVENT_DATE

    def test_from_extracted_fact_propagates_key_date(self):
        ef = ExtractedFact(
            fact_text="x",
            fact_type="world",
            mentioned_at=datetime(2027, 3, 1),
            key_date=EVENT_DATE,
        )
        pf = ProcessedFact.from_extracted_fact(ef, embedding=[0.0] * 8)
        assert pf.key_date == EVENT_DATE

    def test_from_extracted_fact_passes_through_null_key_date(self):
        ef = ExtractedFact(fact_text="x", fact_type="world")
        pf = ProcessedFact.from_extracted_fact(ef, embedding=[0.0] * 8)
        assert pf.key_date is None


# ---- fact_storage tag emission + event_date precedence -------------------------


class TestFactStorageTagEmission:
    """Mirrors the per-fact prep loop in `fact_storage.insert_facts_batch`
    (the lines that build `event_dates` and `tags_list`). We replicate
    rather than invoke because the real function requires a DB.

    If the production loop ever changes the precedence (key_date vs
    occurred_start vs mentioned_at) or the tag-emission convention, this
    duplicated logic will silently pass on stale rules. Source of truth:
    `hindsight_api/engine/retain/fact_storage.py` — the for-loop in
    `insert_facts_batch`.
    """

    def _prep(self, fact: ProcessedFact) -> tuple[datetime | None, list[str]]:
        if fact.key_date is not None:
            event_date = datetime.combine(fact.key_date, datetime.min.time())
        elif fact.occurred_start is not None:
            event_date = fact.occurred_start
        else:
            event_date = fact.mentioned_at

        tags = list(fact.tags) if fact.tags else []
        if fact.key_date is not None:
            tag = f"key_date:{fact.key_date.isoformat()}"
            if tag not in tags:
                tags.append(tag)
        return event_date, tags

    def _make(self, **overrides):
        defaults = dict(
            fact_text="x",
            fact_type="world",
            embedding=[0.0] * 8,
            occurred_start=None,
            occurred_end=None,
            mentioned_at=datetime(2027, 3, 1),
            context="",
            metadata={},
        )
        defaults.update(overrides)
        return ProcessedFact(**defaults)

    def test_event_date_uses_key_date_when_present(self):
        # Wire OTHER_DATE through occurred_start to prove key_date wins
        # when both are set. EVENT_DATE is the canonical one.
        pf = self._make(
            key_date=EVENT_DATE,
            occurred_start=datetime.combine(OTHER_DATE, datetime.min.time()),
        )
        event_date, _ = self._prep(pf)
        assert event_date == datetime.combine(EVENT_DATE, datetime.min.time())

    def test_event_date_falls_back_to_occurred_start(self):
        pf = self._make(occurred_start=datetime.combine(EVENT_DATE, datetime.min.time()))
        event_date, _ = self._prep(pf)
        assert event_date == datetime.combine(EVENT_DATE, datetime.min.time())

    def test_event_date_falls_back_to_mentioned_at(self):
        mentioned_at = datetime(2027, 3, 1)
        pf = self._make(mentioned_at=mentioned_at)
        event_date, _ = self._prep(pf)
        assert event_date == mentioned_at

    def test_tag_emission_when_key_date_set(self):
        pf = self._make(
            key_date=EVENT_DATE,
            tags=[AGENT_TAG, HOUSEHOLD_TAG],
        )
        _, tags = self._prep(pf)
        expected_kd_tag = f"key_date:{EVENT_ISO}"
        assert expected_kd_tag in tags
        assert AGENT_TAG in tags
        assert HOUSEHOLD_TAG in tags

    def test_no_tag_when_key_date_null(self):
        pf = self._make(tags=[AGENT_TAG])
        _, tags = self._prep(pf)
        assert tags == [AGENT_TAG]
        assert not any(t.startswith("key_date:") for t in tags)

    def test_tag_dedup_against_existing(self):
        # Operator pre-supplied a key_date tag; we must not duplicate.
        expected_kd_tag = f"key_date:{EVENT_ISO}"
        pf = self._make(
            key_date=EVENT_DATE,
            tags=[expected_kd_tag, AGENT_TAG],
        )
        _, tags = self._prep(pf)
        assert tags.count(expected_kd_tag) == 1


# ---- Prompt integrity smoke tests ----------------------------------------------


class TestPromptIntegrity:
    """Protects against silent regression of the load-bearing prompt
    additions. The specific example in the prompt is fictional
    (Thanksgiving + museum opening, dates in 2027); these tests assert
    on the conceptual elements rather than re-asserting the example
    verbatim."""

    def test_concise_prompt_mentions_key_date(self):
        from hindsight_api.engine.retain.fact_extraction import (
            CONCISE_FACT_EXTRACTION_PROMPT,
        )

        assert "key_date" in CONCISE_FACT_EXTRACTION_PROMPT
        assert "single canonical event date" in CONCISE_FACT_EXTRACTION_PROMPT

    def test_concise_prompt_warns_against_conflated_facts(self):
        from hindsight_api.engine.retain.fact_extraction import (
            CONCISE_FACT_EXTRACTION_PROMPT,
        )

        assert "do NOT" in CONCISE_FACT_EXTRACTION_PROMPT
        assert "conflate" in CONCISE_FACT_EXTRACTION_PROMPT.lower()

    def test_concise_prompt_has_split_example(self):
        """Confirm the multi-event SPLIT example is present, without
        re-asserting its exact wording."""
        from hindsight_api.engine.retain.fact_extraction import (
            CONCISE_FACT_EXTRACTION_PROMPT,
        )

        assert "SEPARATE facts" in CONCISE_FACT_EXTRACTION_PROMPT
        assert "WRONG" in CONCISE_FACT_EXTRACTION_PROMPT

    def test_verbose_prompt_mentions_key_date(self):
        """VERBOSE has its own TEMPORAL HANDLING section that must also
        teach the key_date discipline — without this, verbose mode runs
        with the schema field present but no LLM guidance (review finding)."""
        from hindsight_api.engine.retain.fact_extraction import (
            VERBOSE_FACT_EXTRACTION_PROMPT,
        )

        assert "key_date" in VERBOSE_FACT_EXTRACTION_PROMPT
        assert "conflate" in VERBOSE_FACT_EXTRACTION_PROMPT.lower()
