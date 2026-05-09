from app.services.glossary import (
    GlossaryViolation,
    deterministic_replace,
    find_violations,
    parse_glossary,
)


def test_parse_json_array() -> None:
    raw = b'[{"term": "Photosynthesis", "translation": "X"}, {"term": "Cell", "translation": "Y"}]'
    terms = parse_glossary(raw, "application/json")
    assert [(t.term_source, t.term_target) for t in terms] == [
        ("Photosynthesis", "X"),
        ("Cell", "Y"),
    ]


def test_parse_json_object() -> None:
    raw = b'{"Photosynthesis": "X", "Cell": "Y"}'
    terms = parse_glossary(raw, "application/json")
    pairs = sorted((t.term_source, t.term_target) for t in terms)
    assert pairs == [("Cell", "Y"), ("Photosynthesis", "X")]


def test_parse_csv() -> None:
    raw = b"term,translation\nPhotosynthesis,X\nCell,Y\n"
    terms = parse_glossary(raw, "text/csv")
    assert [(t.term_source, t.term_target) for t in terms] == [
        ("Photosynthesis", "X"),
        ("Cell", "Y"),
    ]


def test_find_violations_flags_missing_term() -> None:
    src = "Photosynthesis is the process plants use to convert sunlight."
    translated = "အပင်များသည် နေရောင်ခြည်ကိုအသုံးပြုသည်."  # missing Burmese for Photosynthesis
    glossary = parse_glossary(
        '[{"term":"Photosynthesis","translation":"အလင်းအာဟာရ"}]'.encode("utf-8"),
        "application/json",
    )
    violations = find_violations(
        source_text=src, translated_text=translated, glossary=glossary
    )
    assert len(violations) == 1
    assert violations[0].term_source == "Photosynthesis"


def test_find_violations_passes_when_target_present() -> None:
    src = "Photosynthesis converts CO2."
    translated = "အလင်းအာဟာရ သည် CO2 ကိုပြောင်းသည်."
    glossary = parse_glossary(
        '[{"term":"Photosynthesis","translation":"အလင်းအာဟာရ"}]'.encode("utf-8"),
        "application/json",
    )
    assert find_violations(source_text=src, translated_text=translated, glossary=glossary) == []


def test_find_violations_skips_terms_not_in_source() -> None:
    src = "A simple sentence about cats."
    translated = "ကြောင်များအကြောင်း ရိုးရှင်းသော စာကြောင်း"
    glossary = parse_glossary(
        '[{"term":"Photosynthesis","translation":"အလင်းအာဟာရ"}]'.encode("utf-8"),
        "application/json",
    )
    assert find_violations(source_text=src, translated_text=translated, glossary=glossary) == []


def test_deterministic_replace_appends_corrections() -> None:
    out = deterministic_replace(
        "translation here",
        [GlossaryViolation("Cell", "ဆဲလ်", True, False)],
    )
    assert "Cell → ဆဲလ်" in out
    assert "translation here" in out
