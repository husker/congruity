"""Scrape real HTTP response headers from ~100 diverse public sites.

Polite scraping: batches of 5, >=1s between batches, 5s timeout per request.
Captures: status code + an allowlisted subset of response metadata.
Cookies, authentication headers and opaque session identifiers are excluded.
This changes the benchmark corpus; do not compare it to the historical raw
capture as though they were the same dataset. Keep captures local.

Run:
    uv run python -m experiments.c1_short_string.fetch_real_headers

Output: data/real_http_headers.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import requests

# Only protocol metadata needed for this experiment. An allowlist also excludes
# new vendor-specific authentication and tracking headers by default.
PUBLIC_HEADERS = frozenset({
    "accept-ranges", "cache-control", "content-encoding", "content-language",
    "content-length", "content-type", "date", "expires", "last-modified",
    "vary", "x-content-type-options", "x-frame-options",
})


def public_url(url: str) -> str:
    """Drop userinfo, query parameters and fragments from recorded URLs."""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc.rsplit("@", 1)[-1], parts.path, "", ""))

SITES = [
    # General content / reference
    "https://en.wikipedia.org/wiki/HTTP",
    "https://www.python.org/",
    "https://developer.mozilla.org/en-US/docs/Web/HTTP",
    "https://www.w3.org/",
    "https://www.ietf.org/",
    "https://www.rfc-editor.org/",
    # News
    "https://www.bbc.com/",
    "https://www.theguardian.com/international",
    "https://www.npr.org/",
    "https://www.reuters.com/",
    "https://www.aljazeera.com/",
    "https://apnews.com/",
    # Public APIs (return JSON; headers are real HTTP)
    "https://api.github.com/",
    "https://httpbin.org/get",
    "https://jsonplaceholder.typicode.com/posts/1",
    "https://api.coindesk.com/v1/bpi/currentprice.json",
    "https://restcountries.com/v3.1/all?fields=name",
    # Tech sites
    "https://news.ycombinator.com/",
    "https://stackoverflow.com/",
    "https://github.com/",
    "https://gitlab.com/explore",
    "https://bitbucket.org/product",
    "https://dev.to/",
    "https://lobste.rs/",
    # Cloud providers
    "https://aws.amazon.com/",
    "https://cloud.google.com/",
    "https://azure.microsoft.com/en-us/",
    "https://www.cloudflare.com/",
    "https://www.digitalocean.com/",
    "https://www.linode.com/",
    # SaaS / marketing
    "https://www.salesforce.com/",
    "https://www.shopify.com/",
    "https://stripe.com/",
    "https://www.notion.so/",
    "https://linear.app/",
    "https://vercel.com/",
    "https://www.netlify.com/",
    "https://www.atlassian.com/",
    "https://slack.com/",
    "https://www.dropbox.com/",
    # CDNs / static
    "https://cdn.jsdelivr.net/",
    "https://cdnjs.cloudflare.com/",
    "https://unpkg.com/",
    "https://fonts.googleapis.com/",
    # Reference / language docs
    "https://docs.python.org/3/",
    "https://nodejs.org/en",
    "https://go.dev/",
    "https://www.rust-lang.org/",
    "https://kubernetes.io/",
    "https://www.docker.com/",
    "https://wiki.archlinux.org/",
    # E-commerce
    "https://www.amazon.com/",
    "https://www.ebay.com/",
    "https://www.etsy.com/",
    "https://www.target.com/",
    "https://www.walmart.com/",
    "https://www.bestbuy.com/",
    "https://www.costco.com/",
    # Social / community
    "https://www.reddit.com/",
    "https://medium.com/",
    "https://www.quora.com/",
    "https://substack.com/",
    "https://mastodon.social/",
    # Search / discovery
    "https://www.google.com/",
    "https://www.bing.com/",
    "https://duckduckgo.com/",
    "https://www.startpage.com/",
    "https://www.ecosia.org/",
    # Travel
    "https://www.airbnb.com/",
    "https://www.booking.com/",
    "https://www.tripadvisor.com/",
    "https://www.expedia.com/",
    # Media / streaming
    "https://www.spotify.com/us/",
    "https://www.netflix.com/",
    "https://www.youtube.com/",
    "https://vimeo.com/",
    "https://www.twitch.tv/",
    "https://soundcloud.com/",
    # Tools
    "https://www.figma.com/",
    "https://www.adobe.com/",
    "https://www.canva.com/",
    "https://www.miro.com/",
    # Education
    "https://www.coursera.org/",
    "https://www.edx.org/",
    "https://www.khanacademy.org/",
    "https://www.udemy.com/",
    "https://www.duolingo.com/",
    # Government / nonprofits
    "https://www.usa.gov/",
    "https://www.un.org/en/",
    "https://www.who.int/",
    "https://www.cdc.gov/",
    "https://www.nasa.gov/",
    "https://www.europa.eu/",
    # Finance / business
    "https://www.bloomberg.com/",
    "https://www.wsj.com/",
    "https://www.ft.com/",
    "https://www.cnbc.com/",
    "https://www.forbes.com/",
    # Auto / specialty
    "https://www.tesla.com/",
    "https://www.ford.com/",
    "https://www.honda.com/",
    # Misc
    "https://www.wikipedia.org/",
    "https://archive.org/",
    "https://www.gutenberg.org/",
    "https://openlibrary.org/",
    "https://www.imdb.com/",
    "https://www.goodreads.com/",
    "https://www.tripadvisor.com/Hotels",
    "https://www.yelp.com/",
]


def fetch_one(url: str, timeout: float = 5.0) -> dict | None:
    headers = {
        "User-Agent": "congruity-research/0.1 (https://github.com/; research benchmark)",
        "Accept": "text/html,application/json,*/*;q=0.5",
        "Accept-Language": "en-US,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "close",
    }
    try:
        r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        # Preserve order via list of (name, value)
        raw_headers = r.raw.headers if hasattr(r.raw, "headers") else r.headers
        pairs = [(k, v) for k, v in raw_headers.items() if k.lower() in PUBLIC_HEADERS]
        return {
            "url": public_url(url),
            "final_url": public_url(r.url),
            "status": r.status_code,
            "headers": pairs,
        }
    except Exception as e:
        # Exception messages may contain full URLs or proxy credentials.
        return {"url": public_url(url), "error": type(e).__name__}


def main() -> int:
    out_path = Path("data/real_http_headers.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    BATCH = 5
    DELAY = 1.0
    ok = 0
    for i in range(0, len(SITES), BATCH):
        batch = SITES[i:i + BATCH]
        for url in batch:
            r = fetch_one(url)
            if r is None:
                continue
            if "headers" in r:
                ok += 1
            results.append(r)
            tag = "OK" if "headers" in r else f"ERR ({r.get('error')})"
            print(f"  [{len(results):3d}/{len(SITES)}] {tag:25s} {url}")
        time.sleep(DELAY)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {len(results)} responses ({ok} with headers) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
