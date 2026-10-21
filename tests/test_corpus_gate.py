"""Tests for the Step 1.5 corpus quality gate.

These exercise the encoding + structure checks (which need no network)
and verify validate_corpus aggregation logic. The relevance check is
exercised only via the no-api-key path (skipped) — embedding-API calls
are not made in tests.
"""

from __future__ import annotations

from pipeline.corpus_gate import (
    _encoding_problems,
    _structure_problems,
    check_encoding,
    check_structure,
    validate_corpus,
)


# ---------------------------------------------------------------------------
# Encoding checks
# ---------------------------------------------------------------------------


def test_encoding_clean_abstract_has_no_issues():
    text = "We propose a novel attention mechanism for sequence modeling."
    assert _encoding_problems(text) == []


def test_encoding_flags_mojibake():
    text = "The authorâ€™s previous work showed that café measurements..."
    problems = _encoding_problems(text)
    assert any("mojibake" in p for p in problems)


def test_encoding_flags_html_tags():
    text = "Background: <p>This study examines</p> the effect of..."
    problems = _encoding_problems(text)
    assert any("HTML" in p or "JATS" in p for p in problems)


def test_encoding_flags_html_entities():
    text = "Authors&#8217; results &amp; discussion follow."
    problems = _encoding_problems(text)
    assert any("entit" in p.lower() for p in problems)


def test_encoding_flags_replacement_char():
    text = "Measurement at �0.5K showed a transition."
    problems = _encoding_problems(text)
    assert any("FFFD" in p or "replacement" in p.lower() for p in problems)


def test_encoding_flags_excessive_question_marks():
    text = "??? ??? ??? ??? ??? ??? ??? ??? ???"  # 27 '?' chars out of 35
    problems = _encoding_problems(text)
    assert any("'?'" in p or "question" in p.lower() for p in problems)


def test_encoding_empty_text_flagged():
    assert _encoding_problems("") == ["empty text"]


def test_check_encoding_emits_issue_per_paper():
    papers = [
        {"paper_id": "P1", "openalex_id": "W1", "title": "Clean title",
         "abstract": "We propose X. " * 8},
        {"paper_id": "P2", "openalex_id": "W2", "title": "Bad authorâ€™s paper",
         "abstract": "Normal abstract content here, plenty of words. " * 4},
    ]
    issues = check_encoding(papers)
    assert any(i.paper_id == "P2" and i.check == "encoding" for i in issues)
    assert not any(i.paper_id == "P1" for i in issues)


# ---------------------------------------------------------------------------
# Structure checks
# ---------------------------------------------------------------------------


def test_structure_clean_paper_has_no_issues():
    p = {
        "openalex_id": "W123456789",
        "title": "A reasonable-length paper title",
        "year": 2024,
        "abstract": " ".join(["word"] * 50),
    }
    assert _structure_problems(p) == []


def test_structure_accepts_url_form_openalex_id():
    p = {
        "openalex_id": "https://openalex.org/W123456789",
        "title": "A reasonable-length paper title",
        "year": 2024,
        "abstract": " ".join(["word"] * 50),
    }
    assert _structure_problems(p) == []


def test_structure_flags_bad_openalex_id():
    p = {
        "openalex_id": "not-a-w-id",
        "title": "A reasonable-length paper title",
        "year": 2024,
        "abstract": " ".join(["word"] * 50),
    }
    problems = _structure_problems(p)
    assert any("W-id regex" in pr for pr in problems)


def test_structure_flags_missing_year():
    p = {
        "openalex_id": "W1",
        "title": "A reasonable-length paper title",
        "year": None,
        "abstract": " ".join(["word"] * 50),
    }
    problems = _structure_problems(p)
    assert any("year" in pr.lower() for pr in problems)


def test_structure_flags_out_of_range_year():
    p = {
        "openalex_id": "W1",
        "title": "A reasonable-length paper title",
        "year": 1800,
        "abstract": " ".join(["word"] * 50),
    }
    problems = _structure_problems(p)
    assert any("out of range" in pr for pr in problems)


def test_structure_flags_short_abstract():
    p = {
        "openalex_id": "W1",
        "title": "A reasonable-length paper title",
        "year": 2024,
        "abstract": "too short",
    }
    problems = _structure_problems(p)
    assert any("abstract too short" in pr for pr in problems)


def test_structure_flags_short_title():
    p = {
        "openalex_id": "W1",
        "title": "Short",
        "year": 2024,
        "abstract": " ".join(["word"] * 50),
    }
    problems = _structure_problems(p)
    assert any("title too short" in pr for pr in problems)


def test_check_structure_emits_issue_per_paper():
    papers = [
        {"paper_id": "P1", "openalex_id": "W123", "title": "A good title",
         "year": 2024, "abstract": " ".join(["w"] * 50)},
        {"paper_id": "P2", "openalex_id": "not-a-wid", "title": "Bad",
         "year": "not-a-year", "abstract": "short"},
    ]
    issues = check_structure(papers)
    p2_issues = [i for i in issues if i.paper_id == "P2"]
    assert len(p2_issues) >= 3  # bad oid, bad year, short abstract, short title
    assert not any(i.paper_id == "P1" for i in issues)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def test_validate_corpus_drops_failing_and_keeps_clean(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    papers = [
        {"paper_id": "P1", "openalex_id": "W1",
         "title": "A perfectly fine title",
         "year": 2024, "abstract": " ".join(["w"] * 60)},
        {"paper_id": "P2", "openalex_id": "garbage",  # bad W-id
         "title": "Another fine title",
         "year": 2024, "abstract": " ".join(["w"] * 60)},
        {"paper_id": "P3", "openalex_id": "W3",
         "title": "Mojibake titles authorâ€™s issue",
         "year": 2024, "abstract": " ".join(["w"] * 60)},
    ]
    report = validate_corpus(papers, problem="A problem statement.",
                             max_fail_rate=0.99)
    assert report.total == 3
    assert "P1" not in report.drop_ids
    assert "P2" in report.drop_ids
    assert "P3" in report.drop_ids
    assert "relevance" in report.skipped_checks
    assert not report.aborted


def test_validate_corpus_aborts_when_fail_rate_exceeds_threshold(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    papers = [
        {"paper_id": "P1", "openalex_id": "bad", "title": "Bad",
         "year": "x", "abstract": ""},
        {"paper_id": "P2", "openalex_id": "W2",
         "title": "A perfectly fine title",
         "year": 2024, "abstract": " ".join(["w"] * 60)},
    ]
    report = validate_corpus(papers, problem="p", max_fail_rate=0.30)
    assert report.fail_rate == 0.5
    assert report.aborted
    assert "exceeds max_fail_rate" in report.abort_reason


def test_validate_corpus_relevance_drops_do_not_abort_when_enough_survive(monkeypatch):
    """Broad discovery corpora should be filtered, not vetoed, when enough
    relevant papers remain for composition."""
    from pipeline import relevance
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/codex"
                        if name == "codex" else None)
    monkeypatch.setattr(
        relevance,
        "check_relevance_codex",
        lambda papers, problem, threshold: (
            [(p["paper_id"], 0.1, "off topic") for p in papers[:35]],
            False,
        ),
    )
    papers = [
        {
            "paper_id": f"P{i}",
            "openalex_id": f"W{i}",
            "title": f"A perfectly fine routing title number {i}",
            "year": 2024,
            "abstract": " ".join(["routing"] * 50),
        }
        for i in range(1, 66)
    ]

    report = validate_corpus(
        papers,
        problem="routing solver portfolio",
        relevance_mode="codex",
        max_fail_rate=0.30,
        min_passed=30,
    )

    assert report.failed == 35
    assert report.passed == 30
    assert report.fail_rate > 0.30
    assert report.quality_failed == 0
    assert not report.aborted


def test_validate_corpus_aborts_when_too_few_papers_survive_relevance(monkeypatch):
    from pipeline import relevance
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/codex"
                        if name == "codex" else None)
    monkeypatch.setattr(
        relevance,
        "check_relevance_codex",
        lambda papers, problem, threshold: (
            [(p["paper_id"], 0.1, "off topic") for p in papers[:36]],
            False,
        ),
    )
    papers = [
        {
            "paper_id": f"P{i}",
            "openalex_id": f"W{i}",
            "title": f"A perfectly fine routing title number {i}",
            "year": 2024,
            "abstract": " ".join(["routing"] * 50),
        }
        for i in range(1, 66)
    ]

    report = validate_corpus(
        papers,
        problem="routing solver portfolio",
        relevance_mode="codex",
        max_fail_rate=0.30,
        min_passed=30,
    )

    assert report.passed == 29
    assert report.aborted
    assert "need at least 30" in report.abort_reason


def test_validate_corpus_clean_corpus_passes(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    papers = [
        {"paper_id": f"P{i}", "openalex_id": f"W{i}",
         "title": f"A perfectly fine title number {i}",
         "year": 2024, "abstract": " ".join(["w"] * 50)}
        for i in range(1, 6)
    ]
    report = validate_corpus(papers, problem="p")
    assert report.failed == 0
    assert report.passed == 5
    assert not report.aborted


def test_validate_corpus_report_as_dict_is_serializable(monkeypatch):
    import json
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    papers = [{"paper_id": "P1", "openalex_id": "bad", "title": "Bad",
               "year": "x", "abstract": ""}]
    report = validate_corpus(papers, problem="p", max_fail_rate=0.99)
    json.dumps(report.as_dict())  # should not raise


def test_validate_corpus_strict_aborts_when_relevance_skipped():
    papers = [{
        "paper_id": "P1",
        "openalex_id": "https://openalex.org/W1",
        "title": "An attention mechanism for sequence modeling",
        "year": 2024,
        "abstract": "We propose a novel attention mechanism. " * 30,
    }]
    report = validate_corpus(
        papers, problem="study attention",
        strict=True, api_key=None,
    )
    assert report.aborted
    assert "relevance" in report.abort_reason.lower()
    assert "relevance" in report.skipped_checks


def test_validate_corpus_non_strict_does_not_abort_on_skipped_relevance():
    papers = [{
        "paper_id": "P1",
        "openalex_id": "https://openalex.org/W1",
        "title": "An attention mechanism for sequence modeling",
        "year": 2024,
        "abstract": "We propose a novel attention mechanism. " * 30,
    }]
    report = validate_corpus(
        papers, problem="study attention",
        strict=False, api_key=None,
    )
    assert not report.aborted
    assert "relevance" in report.skipped_checks


def test_print_report_warns_on_skipped_relevance(capsys):
    from pipeline.corpus_gate import (
        CorpusGateReport, print_report,
    )
    r = CorpusGateReport(
        total=5, passed=5, failed=0, fail_rate=0.0,
        by_check={"relevance": 0, "encoding": 0, "structure": 0},
        drop_ids=set(),
    )
    r.skipped_checks.append("relevance")
    print_report(r)
    captured = capsys.readouterr()
    assert "SKIPPED" in captured.out
    assert "relevance" in captured.out.lower()
