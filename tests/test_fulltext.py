"""Full-text mining (pipeline/fulltext.py): OA resolution, excerpting, caching,
and the codegen prompt block. No network, no pdftotext — everything below the
fetch layer is exercised with fakes."""

from types import SimpleNamespace

import pipeline.fulltext as ft
from pipeline import autosearch
from pipeline.fulltext import (
    format_fulltext_block,
    fulltext_for_composition,
    method_excerpt,
    resolve_pdf_url,
)

# ---------------------------------------------------------------- resolve


def test_resolve_prefers_best_oa_pdf():
    work = {
        "best_oa_location": {"pdf_url": "https://x/best.pdf"},
        "primary_location": {"pdf_url": "https://x/primary.pdf"},
    }
    assert resolve_pdf_url(work) == "https://x/best.pdf"


def test_resolve_falls_back_to_primary_then_locations():
    work = {
        "best_oa_location": {"pdf_url": None},
        "primary_location": {"pdf_url": None},
        "locations": [{"pdf_url": None}, {"pdf_url": "https://x/loc.pdf"}],
    }
    assert resolve_pdf_url(work) == "https://x/loc.pdf"


def test_resolve_rewrites_arxiv_landing_page():
    work = {
        "best_oa_location": {
            "pdf_url": None,
            "landing_page_url": "https://arxiv.org/abs/2101.01234v2",
        },
    }
    assert resolve_pdf_url(work) == "https://arxiv.org/pdf/2101.01234v2"


def test_resolve_none_when_closed():
    assert resolve_pdf_url({"best_oa_location": None, "locations": []}) is None


def test_resolve_uses_pdf_suffixed_landing_page():
    work = {"primary_location": {
        "pdf_url": None,
        "landing_page_url": "http://e-collection.ethbib.ethz.ch/eserv/eth:31180/eth-31180-01.pdf",
    }}
    assert resolve_pdf_url(work).endswith("eth-31180-01.pdf")


def test_zenodo_record_id_from_doi_and_record_urls():
    assert ft.zenodo_record_id("https://doi.org/10.5281/zenodo.20753866") == "20753866"
    assert ft.zenodo_record_id("https://zenodo.org/records/123") == "123"
    assert ft.zenodo_record_id("https://zenodo.org/record/456") == "456"
    assert ft.zenodo_record_id("https://example.org/paper") is None


# ---------------------------------------------------------------- excerpt

_FILLER = "The weather in the region has long been considered pleasant. " * 4
_METHODY = (
    "Algorithm 2 gives the canonical form check: we prune any extension whose "
    "canonical relabeling precedes the current prefix, and backtrack on chord "
    "violations detected by the incremental oracle. " * 3
)


def test_method_excerpt_prefers_cue_paragraphs():
    text = f"{_FILLER}\n\n{_METHODY}\n\n{_FILLER}"
    out = method_excerpt(text, max_chars=len(_METHODY) + 10)
    assert "canonical" in out
    assert "weather" not in out


def test_method_excerpt_respects_budget():
    text = "\n\n".join([_METHODY] * 20)
    assert len(method_excerpt(text, max_chars=500)) <= 510


def test_method_excerpt_falls_back_to_head():
    out = method_excerpt(_FILLER, max_chars=100)
    assert out == _FILLER[:100]


# ---------------------------------------------------------------- fetch+cache

_COMP = {"papers": [
    {"paper_id": "P1", "role": "canonical construction"},
    {"paper_id": "P2", "role": "closed paper"},
    {"paper_id": "P9", "role": "not in corpus"},
]}
_CORPUS = [
    {"paper_id": "P1", "openalex_id": "https://openalex.org/W111"},
    {"paper_id": "P2", "openalex_id": "https://openalex.org/W222"},
]


def test_fulltext_caches_and_marks_unavailable(tmp_path, monkeypatch):
    calls: list[str] = []

    def fake_fetch(paper):
        calls.append(paper["openalex_id"])
        return _METHODY if paper["openalex_id"].endswith("W111") else ""

    monkeypatch.setattr(ft, "_fetch_and_extract", fake_fetch)
    monkeypatch.setattr(ft, "FETCH_DELAY_S", 0)

    out = fulltext_for_composition(_COMP, _CORPUS, tmp_path)
    assert set(out) == {"P1"}
    assert "canonical" in out["P1"]
    assert (tmp_path / "fulltext" / "P1.txt").read_text() == _METHODY
    # P2 unavailability is cached as an empty file, P9 skipped entirely
    assert (tmp_path / "fulltext" / "P2.txt").read_text() == ""
    assert not (tmp_path / "fulltext" / "P9.txt").exists()

    # Second call: served from cache, no re-fetch
    out2 = fulltext_for_composition(_COMP, _CORPUS, tmp_path)
    assert out2.keys() == out.keys()
    assert len(calls) == 2


# ---------------------------------------------------------------- arxiv

_ATOM_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1603.05119v2</id>
    <title>New Bounds for  the Snake-in-the-Box Problem</title>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/9999.00001v1</id>
    <title>Some Unrelated Paper</title>
  </entry>
</feed>"""


def test_arxiv_id_from_work_abs_pdf_and_eprint_urls():
    assert ft.arxiv_id_from_work({"primary_location": {
        "landing_page_url": "https://arxiv.org/abs/1603.05119v2"}}) == "1603.05119v2"
    assert ft.arxiv_id_from_work({"primary_location": {
        "pdf_url": "https://arxiv.org/pdf/1603.05119v2.pdf"}}) == "1603.05119v2"
    assert ft.arxiv_id_from_work({"primary_location": {
        "landing_page_url": "https://example.org/x"}}) is None


def test_parse_arxiv_search_matches_normalized_title():
    got = ft.parse_arxiv_search(_ATOM_FEED, "New bounds for the snake-in-the-box problem")
    assert got == "1603.05119v2"
    assert ft.parse_arxiv_search(_ATOM_FEED, "A Different Title") is None


def test_parse_arxiv_search_rejects_doctype():
    evil = '<!DOCTYPE feed [<!ENTITY a "b">]>' + _ATOM_FEED.split("?>")[1]
    assert ft.parse_arxiv_search(evil, "New Bounds for the Snake-in-the-Box Problem") is None


def _targz_with_tex(tex: bytes) -> bytes:
    import gzip as _gzip
    import io as _io
    import tarfile as _tarfile
    buf = _io.BytesIO()
    with _tarfile.open(fileobj=buf, mode="w") as tar:
        info = _tarfile.TarInfo("paper.tex")
        info.size = len(tex)
        tar.addfile(info, _io.BytesIO(tex))
    return _gzip.compress(buf.getvalue())


def test_latex_from_eprint_targz_and_single_gzip_and_pdf():
    tex = b"\\documentclass{article}\\begin{algorithm}canonical DFS\\end{algorithm}"
    assert "canonical DFS" in ft.latex_from_eprint(_targz_with_tex(tex))
    import gzip as _gzip
    assert "canonical DFS" in ft.latex_from_eprint(_gzip.compress(tex))
    assert ft.latex_from_eprint(b"%PDF-1.5 blah") is None


def test_fetch_and_extract_prefers_arxiv_latex_then_title_search(monkeypatch):
    monkeypatch.setattr(ft, "_fetch_work", lambda oid: {
        "title": "T", "primary_location": {"landing_page_url": "https://arxiv.org/abs/12.34"}})
    monkeypatch.setattr(ft, "_arxiv_text", lambda aid: f"latex-for-{aid}")
    assert ft._fetch_and_extract({"openalex_id": "W1"}) == "latex-for-12.34"

    # no locations at all -> title-search fallback
    monkeypatch.setattr(ft, "_fetch_work", lambda oid: {"title": "T"})
    monkeypatch.setattr(ft, "_arxiv_search_id", lambda title: "56.78")
    assert ft._fetch_and_extract({"openalex_id": "W1"}) == "latex-for-56.78"


# ---------------------------------------------------------------- prompt block


def test_format_block_empty_when_no_excerpts():
    assert format_fulltext_block({}, _COMP) == ""


def test_format_block_names_paper_and_role():
    block = format_fulltext_block({"P1": "excerpt body"}, _COMP)
    assert "FULL-TEXT METHOD EXCERPTS" in block
    assert "P1 (canonical construction)" in block
    assert "excerpt body" in block


def test_generate_initial_appends_extra_context(monkeypatch, tmp_path):
    captured: dict[str, str] = {}

    def fake_codegen(prompt, dest, stage):
        captured["prompt"] = prompt
        return dest

    monkeypatch.setattr(autosearch, "_codegen", fake_codegen)
    pack = SimpleNamespace(
        pack=SimpleNamespace(problem_shape=SimpleNamespace(
            objective="obj", constraints=["c1"])),
        evaluator=SimpleNamespace(metric_primary="m", candidate_interface="def solve()"),
        gotchas="",
    )
    autosearch.generate_initial(
        {"synthesis_claim": "claim"}, {}, pack, "problem",
        tmp_path / "c.py", "emphasis", stage="s",
        extra_context="\nFULL-TEXT METHOD EXCERPTS marker\n")
    assert "FULL-TEXT METHOD EXCERPTS marker" in captured["prompt"]
    # the variant directive must survive after the appended block
    assert captured["prompt"].index("marker") < captured["prompt"].index("VARIANT DIRECTIVE")
