"""Mid-body window repair: exact search over interior rewrites of a snake.

Freeze prefix and suffix, free the window between them, and exhaustively
enumerate chord-free replacements of length (window + delta) that bridge
the seams. Unlike end-unwinding (complete_certificate), this performs
surgery on the interior — the operation that end-rigidity theorems say is
the only local move left on near-optimal snakes.

Used 2026-07-04 to prove Wynn-190 and our 185 are mid-body locked
(no +1 repair, all windows w<=40 exhausted; ledgers in data/snake_in_box/).
A True `exhausted` flag makes a not-found result a per-window theorem.
"""

from __future__ import annotations

from experiments.snake_in_box.baselines import validate_snake


def window_repair(n: int, seq: list[int], i: int, j: int, extra: int,
                  node_cap: int = 300_000_000
                  ) -> tuple[list[int] | None, int, bool]:
    """Replace seq[i+1:j] with an exact-searched window of length
    (j-i-1)+extra. Returns (repaired_snake_or_None, nodes, exhausted).
    The repaired snake is pack-validated before being returned."""
    size = 1 << n
    prefix, suffix = seq[: i + 1], seq[j:]
    target_len = (j - i - 1) + extra
    on = [False] * size
    nc = [0] * size
    for v in prefix + suffix:
        on[v] = True
    for v in prefix + suffix:
        for b in range(n):
            nc[v ^ (1 << b)] += 1
    # empty suffix = end-window: no bridge condition, terminal is any legal
    # vertex at the target depth (unifies end surgery with mid surgery)
    sfx0 = suffix[0] if suffix else None
    out: list[list[int]] = []
    st = {"nodes": 0, "exh": True}

    def dfs(cur: int, depth: int) -> None:
        if out:
            return
        if st["nodes"] >= node_cap:
            st["exh"] = False
            return
        st["nodes"] += 1
        for b in range(n):
            v = cur ^ (1 << b)
            if on[v]:
                continue
            if depth == target_len:
                if sfx0 is None:
                    if nc[v] == 1:          # end-window: just a legal cap
                        out.append([v])
                        return
                # mid-window terminal bridges to the suffix: its only snake
                # neighbours must be the current head and suffix[0]
                elif nc[v] == 2 and bin(v ^ sfx0).count("1") == 1:
                    out.append([v])
                    return
                continue
            if nc[v] != 1:
                continue
            on[v] = True
            for bb in range(n):
                nc[v ^ (1 << bb)] += 1
            dfs(v, depth + 1)
            for bb in range(n):
                nc[v ^ (1 << bb)] -= 1
            on[v] = False
            if out:
                out[0].insert(0, v)
                return

    if target_len >= 1:
        dfs(prefix[-1], 1)
    if out:
        cand = prefix + out[0] + suffix
        ok, _ = validate_snake(n, cand)
        if ok:
            return cand, st["nodes"], st["exh"]
    return None, st["nodes"], st["exh"]
