"""Q9 campaign launcher — validate the ladder config before burning days of
compute on it. These tests read the repo's committed certificates and the
global Q9 WL checkpoint; they run no sampling."""

import math

from experiments.snake_in_box.q9_campaign import (
    OUR_BEST,
    RECORD_LEN,
    REF_MAX,
    build_reference,
    build_windows,
    load_certificate,
)
from tests.test_windowed_wl import _is_snake


def test_load_certificate_seq_key():
    p = load_certificate("data/snake_in_box/certificates/q9_len185_joint.json")
    assert len(p) == 186
    assert _is_snake(9, p)


def test_load_certificate_path_key():
    p = load_certificate("data/snake_in_box/seed6003_q9_181.json")
    assert len(p) == 182
    assert _is_snake(9, p)


def test_ladder_is_launchable():
    ws = build_windows()
    names = [w["name"] for w in ws]
    assert len(set(names)) == len(names)
    ordered = sorted(ws, key=lambda w: w["lmin"])
    # lowest window must overlap the trusted reference for the anchor check
    assert ordered[0]["lmin"] <= REF_MAX
    # consecutive windows must overlap so stitch can chain them
    for a, b in zip(ordered, ordered[1:]):
        assert b["lmin"] <= a["lmax"]
    # every seed is a real snake whose truncated start lands inside its band
    for w in ws:
        seed = w["seed_snake"]
        assert _is_snake(9, seed)
        start_len = min(len(seed) - 1, w["lmax"])
        assert w["lmin"] <= start_len <= w["lmax"]
    # the ladder reaches the record bin
    assert max(w["lmax"] for w in ws) == RECORD_LEN
    assert any(w["lmin"] <= OUR_BEST <= w["lmax"] for w in ws)


def test_build_reference_is_anchored_and_capped():
    ref = build_reference()
    assert min(ref) == 0
    assert max(ref) == REF_MAX
    # anchored at the exactly-known N(0) = 2^9
    assert abs(ref[0] - 9 * math.log(2.0)) < 1e-9
    # trusted region is contiguous — no unvisited holes below the cap
    assert sorted(ref) == list(range(0, REF_MAX + 1))
