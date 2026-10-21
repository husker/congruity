"""Short-string corpora for C1 benchmark.

Each generator returns a list[bytes] of 8-256 byte records. Determinism via
fixed seeds so train/test splits are stable.
"""

from __future__ import annotations

import json
import random
import string
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path


def http_headers(n: int = 5000, seed: int = 1) -> list[bytes]:
    rng = random.Random(seed)
    names = [
        "Content-Type", "Content-Length", "Cache-Control", "Connection",
        "Accept", "Accept-Encoding", "Accept-Language", "User-Agent",
        "Authorization", "Cookie", "Set-Cookie", "Referer", "Origin",
        "Host", "Date", "Server", "X-Forwarded-For", "X-Request-Id",
        "X-Frame-Options", "Strict-Transport-Security", "ETag", "Last-Modified",
        "Vary", "Pragma", "Expires", "Location",
    ]
    content_types = [
        "application/json", "application/json; charset=utf-8",
        "text/html; charset=utf-8", "text/plain", "image/png", "image/jpeg",
        "application/octet-stream", "application/javascript",
        "text/css", "application/x-www-form-urlencoded",
    ]
    cache_vals = [
        "no-cache", "no-store", "max-age=3600", "max-age=0, must-revalidate",
        "public, max-age=86400", "private, max-age=0", "immutable",
    ]
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
        "curl/8.4.0",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_1 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148 Safari/604.1",
        "okhttp/4.12.0",
        "python-requests/2.31.0",
    ]
    hosts = ["api.example.com", "cdn.example.com", "www.example.org", "static.test.io", "auth.svc.local"]
    out: list[bytes] = []
    while len(out) < n:
        n_choice = rng.choice(names)
        if n_choice == "Content-Type":
            v = rng.choice(content_types)
        elif n_choice == "Content-Length":
            v = str(rng.randint(0, 1_000_000))
        elif n_choice == "Cache-Control":
            v = rng.choice(cache_vals)
        elif n_choice == "User-Agent":
            v = rng.choice(user_agents)
        elif n_choice == "Host":
            v = rng.choice(hosts)
        elif n_choice == "Date":
            t = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=rng.randint(0, 365*86400))
            v = t.strftime("%a, %d %b %Y %H:%M:%S GMT")
        elif n_choice == "X-Request-Id":
            v = str(uuid.UUID(int=rng.getrandbits(128)))
        elif n_choice == "ETag":
            v = '"' + "".join(rng.choices("0123456789abcdef", k=16)) + '"'
        elif n_choice == "Authorization":
            tok = "".join(rng.choices(string.ascii_letters + string.digits + "-_", k=rng.choice([32, 40, 64])))
            v = f"Bearer {tok}"
        elif n_choice == "Cookie" or n_choice == "Set-Cookie":
            tok = "".join(rng.choices(string.ascii_letters + string.digits, k=24))
            v = f"sid={tok}; Path=/; HttpOnly"
        else:
            v = "".join(rng.choices(string.ascii_letters + string.digits + " /.-", k=rng.randint(8, 80)))
        rec = f"{n_choice}: {v}".encode("utf-8")
        if 8 <= len(rec) <= 256:
            out.append(rec)
    return out


def syslog_lines(n: int = 5000, seed: int = 2) -> list[bytes]:
    rng = random.Random(seed)
    services = ["sshd", "systemd", "kernel", "nginx", "postfix", "cron", "dhcpd", "named", "ntpd", "snapd"]
    hosts = ["host01", "host02", "edge-3a", "db-master", "worker-7", "router-core"]
    templates = [
        "sshd[{pid}]: Accepted publickey for {user} from {ip} port {port} ssh2",
        "sshd[{pid}]: Failed password for invalid user {user} from {ip} port {port} ssh2",
        "systemd[{pid}]: Started Session {sid} of user {user}.",
        "systemd[{pid}]: Stopped {service}.service.",
        "kernel: [{ts}] eth0: link up, 1000Mbps, full-duplex",
        "nginx: {ip} - - [{date}] \"GET {path} HTTP/1.1\" {status} {size}",
        "cron[{pid}]: ({user}) CMD ({cmd})",
        "postfix/smtpd[{pid}]: connect from unknown[{ip}]",
        "named[{pid}]: client {ip}#{port}: query: {host} IN A",
        "dhcpd: DHCPACK on {ip} to {mac} via eth0",
    ]
    users = ["root", "deploy", "alice", "bob", "www-data", "postgres", "admin"]
    paths = ["/", "/api/v1/users", "/health", "/login", "/static/app.js", "/favicon.ico"]
    out: list[bytes] = []
    while len(out) < n:
        host = rng.choice(hosts)
        t = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=rng.randint(0, 365*86400))
        date_str = t.strftime("%b %d %H:%M:%S")
        tpl = rng.choice(templates)
        msg = tpl.format(
            pid=rng.randint(100, 99999),
            user=rng.choice(users),
            ip=f"{rng.randint(1,254)}.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}",
            port=rng.randint(1024, 65535),
            sid=rng.randint(1, 5000),
            service=rng.choice(services),
            ts=f"{rng.uniform(0, 99999):.6f}",
            date=t.strftime("%d/%b/%Y:%H:%M:%S +0000"),
            path=rng.choice(paths),
            status=rng.choice([200, 200, 200, 301, 302, 404, 500]),
            size=rng.randint(0, 100000),
            cmd="run-parts /etc/cron.hourly",
            host=f"{rng.choice(['www','api','mail','db'])}.example.com",
            mac=":".join(f"{rng.randint(0,255):02x}" for _ in range(6)),
        )
        rec = f"{date_str} {host} {msg}".encode("utf-8")
        if 8 <= len(rec) <= 256:
            out.append(rec)
    return out


def json_records(n: int = 5000, seed: int = 3) -> list[bytes]:
    rng = random.Random(seed)
    actions = ["login", "logout", "view", "click", "purchase", "signup", "search", "share"]
    sources = ["web", "ios", "android", "api"]
    out: list[bytes] = []
    while len(out) < n:
        kind = rng.randint(0, 4)
        t = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=rng.randint(0, 365*86400))
        ts = t.isoformat()
        if kind == 0:
            rec = f'{{"event":"{rng.choice(actions)}","ts":"{ts}","uid":{rng.randint(1,9999999)},"src":"{rng.choice(sources)}"}}'
        elif kind == 1:
            rec = f'{{"id":{rng.randint(1, 9999999)},"status":"{rng.choice(["ok","fail","pending"])}","retries":{rng.randint(0,5)}}}'
        elif kind == 2:
            rec = f'{{"user":{{"id":{rng.randint(1,99999)},"name":"user_{rng.randint(0,999)}"}},"ts":"{ts}"}}'
        elif kind == 3:
            tags = ",".join(f'"{rng.choice(["new","beta","prod","exp"])}"' for _ in range(rng.randint(1, 4)))
            rec = f'{{"sku":"SKU-{rng.randint(10000,99999)}","tags":[{tags}],"price":{rng.uniform(1,999):.2f}}}'
        else:
            rec = f'{{"q":"{rng.choice(["red shoes","blue dress","green hat","laptop","phone"])}","page":{rng.randint(1,20)},"per_page":{rng.choice([10,20,50])}}}'
        b = rec.encode("utf-8")
        if 8 <= len(b) <= 256:
            out.append(b)
    return out


def text_chunks(n: int = 5000, seed: int = 4, chunk_size: int = 128) -> list[bytes]:
    """Synthetic English-ish text chunks (proxy for Calgary/enwik8 chunked).

    Uses a small Markov-ish corpus assembled from common English words. We
    don't claim parity with enwik8 — this is a stylistic proxy. Same data
    flows through all methods so the comparison is fair.
    """
    rng = random.Random(seed)
    words = (
        "the of and to a in is it you that he was for on are with as I his they be "
        "at one have this from or had by hot word but what some we can out other "
        "were all your when up use word how said an each she which do their time if "
        "will way about many then them write would like so these her long make thing "
        "see him two has look more day could go come did number sound no most people "
        "my over know water than call first who may down side been now find any new "
        "work part take get place made live where after back little only round man "
        "year came show every good me give our under name very through just form much "
        "great think say help low line before turn cause same mean differ move right "
        "boy old too does tell sentence set three want air well also play small end"
    ).split()
    out: list[bytes] = []
    while len(out) < n:
        ws: list[str] = []
        total = 0
        while total < chunk_size:
            w = rng.choice(words)
            ws.append(w)
            total += len(w) + 1
        chunk = " ".join(ws)[:chunk_size].encode("utf-8")
        if 8 <= len(chunk) <= 256:
            out.append(chunk)
    return out


REAL_HEADERS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "real_http_headers.json"


def _load_real_blocks() -> list[list[tuple[str, str]]]:
    """Load scraped real HTTP response headers as list-of-blocks.

    Each block = one site's full response header list. Pairs preserved
    in original wire order. Multi-valued names (e.g. set-cookie) appear
    multiple times.
    """
    if not REAL_HEADERS_PATH.exists():
        raise FileNotFoundError(
            f"{REAL_HEADERS_PATH} not found. Run fetch_real_headers.py first."
        )
    data = json.loads(REAL_HEADERS_PATH.read_text())
    blocks: list[list[tuple[str, str]]] = []
    for d in data:
        if "headers" not in d:
            continue
        block = [(str(n), str(v)) for n, v in d["headers"]]
        if block:
            blocks.append(block)
    return blocks


def real_http_lines(n: int = 5000) -> list[bytes]:
    """Real HTTP response headers as `Name: Value` byte records, 8-256 bytes.

    Line-level shuffle. NOTE: this can leak globally-frequent lines into both
    train and test halves. For a leakage-free evaluation see `real_http_lines_by_site_split`.
    """
    blocks = _load_real_blocks()
    out: list[bytes] = []
    for block in blocks:
        for name, value in block:
            rec = f"{name}: {value}".encode("utf-8")
            if 8 <= len(rec) <= 256:
                out.append(rec)
    rng = random.Random(7)
    rng.shuffle(out)
    return out[:n]


def real_http_lines_by_site_split(
    train_sites_frac: float = 0.5,
) -> tuple[list[bytes], list[bytes]]:
    """Split scraped real HTTP headers into train/test by SITE, not by line.

    Returns (train_lines, test_lines). Train sites' lines never appear in
    test. Globally-frequent lines (e.g. 'Connection: close') will appear
    in both halves only by coincidence; site-specific lines won't leak.
    This is the leakage-free fairness check.
    """
    blocks = _load_real_blocks()
    rng = random.Random(11)
    rng.shuffle(blocks)
    cut = int(len(blocks) * train_sites_frac)
    train_blocks, test_blocks = blocks[:cut], blocks[cut:]

    def to_lines(bs: list[list[tuple[str, str]]]) -> list[bytes]:
        out: list[bytes] = []
        for block in bs:
            for name, value in block:
                rec = f"{name}: {value}".encode("utf-8")
                if 8 <= len(rec) <= 256:
                    out.append(rec)
        return out

    return to_lines(train_blocks), to_lines(test_blocks)


def real_http_blocks(n: int = 200) -> list[list[tuple[str, str]]]:
    """Real HTTP response header blocks (one block per site).

    Filters block down to entries whose serialized line is <= 256 bytes so
    direct per-line comparisons remain in scope. Block sizes vary widely.
    """
    blocks = _load_real_blocks()
    out: list[list[tuple[str, str]]] = []
    for block in blocks:
        filtered = [(n_, v) for n_, v in block if 8 <= len(f"{n_}: {v}".encode("utf-8")) <= 256]
        if filtered:
            out.append(filtered)
    rng = random.Random(8)
    rng.shuffle(out)
    return out[:n]


CORPORA = {
    "http_headers": http_headers,
    "syslog": syslog_lines,
    "json": json_records,
    "text128": lambda n=5000, seed=4: text_chunks(n=n, seed=seed, chunk_size=128),
}


def load_all(n: int = 5000) -> dict[str, list[bytes]]:
    return {name: fn(n=n) for name, fn in CORPORA.items()}


def split(records: list[bytes], train_frac: float = 0.5) -> tuple[list[bytes], list[bytes]]:
    cut = int(len(records) * train_frac)
    return records[:cut], records[cut:]


def split_blocks(
    blocks: list[list[tuple[str, str]]], train_frac: float = 0.5
) -> tuple[list[list[tuple[str, str]]], list[list[tuple[str, str]]]]:
    cut = int(len(blocks) * train_frac)
    return blocks[:cut], blocks[cut:]


if __name__ == "__main__":
    import statistics
    for name, fn in CORPORA.items():
        recs = fn(n=2000)
        lens = [len(r) for r in recs]
        print(f"{name:15s}  n={len(recs):5d}  "
              f"len min={min(lens):3d} p50={statistics.median(lens):5.1f} "
              f"p99={sorted(lens)[int(len(lens)*0.99)]:3d} max={max(lens):3d}  "
              f"mean={statistics.mean(lens):5.1f}")
