"""Full-text mining for composition codegen (roadmap: full-text/methods mode).

The compose/codegen prompts have only ever seen reconstructed abstracts
(~80 words per paper). For record hunting that starves codegen of the exact
machinery the literature used — canonical-DFS pseudocode, pruning rules,
seed constructions live in the METHOD sections, not the abstracts. This
module fetches open-access full text for a composition's constituent papers
and distills method-relevant excerpts for the candidate/refine prompts.

Design constraints:
  - OA-only, best-effort: a paper without a resolvable OA PDF is skipped
    with a note, never an error. Closed papers stay abstract-only.
  - No new Python dependency: PDF extraction shells out to `pdftotext`
    (poppler); arXiv Atom XML is parsed with the stdlib.
  - Cached per compose run under <run_dir>/fulltext/<paper_id>.txt so
    repeated autosearch invocations (resume, more arms) never re-fetch.
  - Polite to hosts: sequential fetches with a small delay; arXiv API
    calls honor the documented 3s courtesy interval
    (https://info.arxiv.org/help/api/index.html).

arXiv is used two ways, both for text quality and coverage rather than
discovery (OpenAlex already indexes arXiv for corpus building):
  - When a work has an arXiv id, the e-print LaTeX source is preferred
    over the PDF — \\begin{algorithm} blocks survive intact instead of
    being mangled by pdftotext.
  - When OpenAlex has no working OA location, an arXiv title search is
    the fallback before giving up.
"""

from __future__ import annotations

import gzip
import io
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

from pipeline._openalex import request_with_retry

FETCH_DELAY_S = 1.0
PDF_TIMEOUT_S = 60
PDFTOTEXT_TIMEOUT_S = 120
MAX_PDF_BYTES = 30_000_000
EXCERPT_CHARS = 4000

# Section/keyword cues that mark method-dense text worth showing codegen.
_METHOD_CUES = re.compile(
    r"algorithm|pseudocode|procedure|canonical|prun|heuristic|backtrack|"
    r"we describe|our method|step 1|initiali[sz]|search strategy",
    re.IGNORECASE,
)

_ARXIV_ABS = re.compile(r"arxiv\.org/abs/([\w.\-/]+)", re.IGNORECASE)
_ZENODO = re.compile(r"(?:doi\.org/10\.5281/zenodo\.|zenodo\.org/records?/)(\d+)",
                     re.IGNORECASE)


def resolve_pdf_url(work: dict) -> str | None:
    """Best OA PDF URL for an OpenAlex work dict, or None. URL rules only.

    Priority: best_oa_location.pdf_url -> primary_location.pdf_url ->
    any locations[].pdf_url -> landing pages that are themselves PDFs ->
    arXiv landing-page rewrite (abs -> pdf). Zenodo landings need an API
    call and are handled separately (_zenodo_pdf_url).
    """
    locations = _locations(work)
    for loc in locations:
        if loc and loc.get("pdf_url"):
            return loc["pdf_url"]
    for loc in locations:
        landing = (loc or {}).get("landing_page_url") or ""
        if landing.lower().endswith(".pdf"):
            return landing
        m = _ARXIV_ABS.search(landing)
        if m:
            return f"https://arxiv.org/pdf/{m.group(1)}"
    return None


def _locations(work: dict) -> list[dict | None]:
    return [work.get("best_oa_location"), work.get("primary_location"),
            *(work.get("locations") or [])]


def zenodo_record_id(landing: str) -> str | None:
    m = _ZENODO.search(landing)
    return m.group(1) if m else None


def _zenodo_pdf_url(work: dict) -> str | None:
    """Resolve a Zenodo landing page to a file download URL via the records API."""
    for loc in _locations(work):
        rec = zenodo_record_id((loc or {}).get("landing_page_url") or "")
        if not rec:
            continue
        try:
            resp = requests.get(f"https://zenodo.org/api/records/{rec}",
                                timeout=PDF_TIMEOUT_S)
            files = resp.json().get("files", []) if resp.status_code == 200 else []
        except (requests.exceptions.RequestException, ValueError):
            continue
        for f in files:
            links = f.get("links") or {}
            if (f.get("key") or "").lower().endswith(".pdf"):
                # InvenioRDM serves bytes at links.content; the legacy API at
                # links.self. Try content first; the %PDF magic check in
                # _download_pdf rejects a metadata response.
                return links.get("content") or links.get("self")
    return None


ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_DELAY_S = 3.0  # documented courtesy interval between arXiv API calls
MAX_SOURCE_BYTES = 30_000_000
_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV_ANY = re.compile(r"arxiv\.org/(?:abs|pdf|e-print)/([\w.\-/]+?)(?:\.pdf)?$",
                        re.IGNORECASE)


def arxiv_id_from_work(work: dict) -> str | None:
    """arXiv id from any OpenAlex location URL (abs, pdf, or e-print form)."""
    for loc in _locations(work):
        for key in ("landing_page_url", "pdf_url"):
            m = _ARXIV_ANY.search((loc or {}).get(key) or "")
            if m:
                return m.group(1)
    return None


def _normalize_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def parse_arxiv_search(atom_xml: str, title: str) -> str | None:
    """arXiv id of the search entry whose title matches `title`, or None."""
    # stdlib ET is DoS-prone on entity expansion; Atom feeds never carry
    # DOCTYPE/ENTITY, so reject any response that does instead of parsing it.
    if re.search(r"<!\s*(DOCTYPE|ENTITY)", atom_xml, re.IGNORECASE):
        return None
    try:
        root = ET.fromstring(atom_xml)
    except ET.ParseError:
        return None
    want = _normalize_title(title)
    if not want:
        return None
    for entry in root.iter(f"{_ATOM}entry"):
        got = _normalize_title(entry.findtext(f"{_ATOM}title") or "")
        if got == want:
            m = _ARXIV_ANY.search(entry.findtext(f"{_ATOM}id") or "")
            if m:
                return m.group(1)
    return None


def _arxiv_search_id(title: str) -> str | None:
    """Title-search fallback against the arXiv API (Atom XML response)."""
    if not title:
        return None
    time.sleep(ARXIV_DELAY_S)
    q = urllib.parse.quote_plus(f'ti:"{title}"')
    try:
        resp = requests.get(f"{ARXIV_API}?search_query={q}&max_results=5",
                            timeout=PDF_TIMEOUT_S,
                            headers={"User-Agent": "congruity-fulltext/1.0"})
    except requests.exceptions.RequestException:
        return None
    if resp.status_code != 200:
        return None
    return parse_arxiv_search(resp.text, title)


def latex_from_eprint(blob: bytes) -> str | None:
    """Concatenated .tex sources from an arXiv e-print blob.

    e-prints are a gzipped tar, a single gzipped file, or (pdf-only
    submissions) a raw PDF. Returns None when there is no LaTeX to read;
    the caller then falls back to the PDF route.
    """
    if blob.startswith(b"%PDF"):
        return None
    try:
        raw = gzip.decompress(blob)
    except (OSError, gzip.BadGzipFile):
        raw = blob
    texts: list[str] = []
    try:
        with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
            for member in tar.getmembers():
                if not member.isfile() or not member.name.lower().endswith(".tex"):
                    continue
                if member.size > MAX_SOURCE_BYTES:
                    continue
                f = tar.extractfile(member)
                if f:
                    texts.append(f.read().decode("utf-8", errors="replace"))
    except tarfile.TarError:
        # single-file source: keep it only when it looks like LaTeX
        head = raw[:2048].decode("utf-8", errors="replace")
        if "\\documentclass" in head or "\\begin" in head:
            texts.append(raw.decode("utf-8", errors="replace"))
    combined = "\n\n".join(texts)
    return combined if combined.strip() else None


def _arxiv_text(arxiv_id: str) -> str:
    """Full text for an arXiv id: e-print LaTeX preferred, PDF fallback."""
    time.sleep(ARXIV_DELAY_S)
    try:
        resp = requests.get(f"https://arxiv.org/e-print/{arxiv_id}",
                            timeout=PDF_TIMEOUT_S,
                            headers={"User-Agent": "congruity-fulltext/1.0"})
        if resp.status_code == 200 and len(resp.content) <= MAX_SOURCE_BYTES:
            latex = latex_from_eprint(resp.content)
            if latex:
                return latex
    except requests.exceptions.RequestException:
        pass
    pdf = _download_pdf(f"https://arxiv.org/pdf/{arxiv_id}")
    return (extract_pdf_text(pdf) or "") if pdf else ""


def _fetch_work(openalex_id: str) -> dict | None:
    short_id = openalex_id.rsplit("/", 1)[-1]
    return request_with_retry(
        f"/works/{short_id}",
        {"select": "id,title,open_access,best_oa_location,primary_location,locations"},
    )


def _download_pdf(url: str) -> bytes | None:
    try:
        resp = requests.get(url, timeout=PDF_TIMEOUT_S,
                            headers={"User-Agent": "congruity-fulltext/1.0"})
    except requests.exceptions.RequestException as e:
        print(f"    fulltext: fetch failed ({type(e).__name__}) {url}", file=sys.stderr)
        return None
    if resp.status_code != 200 or len(resp.content) > MAX_PDF_BYTES:
        print(f"    fulltext: HTTP {resp.status_code}, {len(resp.content)}B for {url}",
              file=sys.stderr)
        return None
    if not resp.content.startswith(b"%PDF"):
        print(f"    fulltext: not a PDF at {url}", file=sys.stderr)
        return None
    return resp.content


def extract_pdf_text(pdf_bytes: bytes) -> str | None:
    """Extract text via pdftotext (poppler). Returns None if unavailable/fails."""
    exe = shutil.which("pdftotext")
    if exe is None:
        print("    fulltext: pdftotext not installed (brew install poppler) — skipping",
              file=sys.stderr)
        return None
    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()
        try:
            proc = subprocess.run(
                [exe, "-enc", "UTF-8", tmp.name, "-"],
                capture_output=True, timeout=PDFTOTEXT_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return None
    if proc.returncode != 0:
        return None
    return proc.stdout.decode("utf-8", errors="replace")


def method_excerpt(text: str, max_chars: int = EXCERPT_CHARS) -> str:
    """Distill the most method-dense stretch of a paper for a codegen prompt.

    Scores paragraphs by method-cue hits, then returns the top-scoring ones
    in document order until the budget is spent. Falls back to the head of
    the document when nothing matches (short papers, OCR noise).
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) > 80]
    if not paragraphs:
        return text[:max_chars]
    scored = sorted(
        ((len(_METHOD_CUES.findall(p)), i, p) for i, p in enumerate(paragraphs)),
        key=lambda t: t[0], reverse=True)
    picked: list[tuple[int, str]] = []
    budget = max_chars
    for hits, i, p in scored:
        if hits == 0 or budget <= 0:
            break
        take = p[:budget]
        picked.append((i, take))
        budget -= len(take) + 2
    if not picked:
        return text[:max_chars]
    return "\n\n".join(p for _i, p in sorted(picked))


def fulltext_for_composition(comp: dict, corpus: list[dict], run_dir: Path,
                             max_chars: int = EXCERPT_CHARS) -> dict[str, str]:
    """Method excerpts for a composition's constituent papers, keyed by paper_id.

    Full extracted text is cached at <run_dir>/fulltext/<paper_id>.txt; a
    cached empty file marks a known-unavailable paper (no refetch storms).
    """
    by_id = {p.get("paper_id"): p for p in corpus}
    cache_dir = run_dir / "fulltext"
    cache_dir.mkdir(parents=True, exist_ok=True)
    excerpts: dict[str, str] = {}
    for entry in comp.get("papers", []):
        pid = entry.get("paper_id")
        paper = by_id.get(pid)
        if not pid or not paper or not paper.get("openalex_id"):
            continue
        cache = cache_dir / f"{pid}.txt"
        if cache.exists():
            text = cache.read_text()
        else:
            text = _fetch_and_extract(paper)
            cache.write_text(text)  # empty string caches "unavailable"
            time.sleep(FETCH_DELAY_S)
        if text:
            excerpts[pid] = method_excerpt(text, max_chars=max_chars)
    return excerpts


def _fetch_and_extract(paper: dict) -> str:
    work = _fetch_work(paper["openalex_id"])
    if not work:
        return ""
    # arXiv LaTeX source beats pdftotext output when the work is on arXiv
    arxiv_id = arxiv_id_from_work(work)
    if arxiv_id:
        text = _arxiv_text(arxiv_id)
        if text:
            return text
    url = resolve_pdf_url(work) or _zenodo_pdf_url(work)
    if url:
        pdf = _download_pdf(url)
        if pdf:
            return extract_pdf_text(pdf) or ""
    # last resort: OpenAlex had no usable OA location — arXiv title search
    arxiv_id = _arxiv_search_id(work.get("title") or paper.get("title") or "")
    return _arxiv_text(arxiv_id) if arxiv_id else ""


def format_fulltext_block(excerpts: dict[str, str], comp: dict) -> str:
    """Prompt block appended to codegen prompts when --fulltext is on."""
    if not excerpts:
        return ""
    roles = {p.get("paper_id"): p.get("role", "") for p in comp.get("papers", [])}
    parts = ["\nFULL-TEXT METHOD EXCERPTS (from the constituent papers — use the "
             "actual algorithmic machinery described here, not just the abstracts):"]
    for pid, text in excerpts.items():
        parts.append(f"\n--- {pid} ({roles.get(pid, '')}) ---\n{text}")
    return "\n".join(parts) + "\n"
